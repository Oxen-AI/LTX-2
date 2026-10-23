"""Sharded state dict with distributed weight backup.
Each rank stores ~1/N of the model weights. Bucketed all-gather rounds
restore weights into a target state dict using only a small,
caller-provided staging buffer.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist


def _stable_owner(key: str, world: int) -> int:
    """Deterministic rank assignment (same across all processes)."""
    h = hashlib.md5(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % world


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


# Staging slots start on a multiple of this many bytes, which keeps the multi-tensor copies' loads aligned.
_SLOT_ALIGN = 256


def _as_bytes(t: torch.Tensor, key: str) -> torch.Tensor:
    if not t.is_contiguous():
        raise RuntimeError(f"ShardedSD: tensor '{key}' is not contiguous")
    return t.view(torch.uint8).view(-1)


@dataclass
class ShardedSD:
    """Sharded state dict with distributed weight backup.
    Distributes model weights across ranks for memory-efficient backup
    and restoration.  Can be used for any scenario where a full state dict
    needs to be restored from sharded storage (e.g. LoRA hot-swap, weight
    rollback, checkpoint recovery).
    - Deterministic ownership: ``MD5(key) % world_size``
    - Local storage only for owned keys (VRAM ≈ 1/world_size of model)
    - Bucketed all-gather rounds through a single small staging buffer
    Usage::
        backup = ShardedSD.from_state_dict(model.state_dict(), group)
        staging = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
        backup.broadcast_shards_into(target_sd, staging)    # cooperative: all ranks must call
    """

    keys: tuple[str, ...]
    """All parameter keys in the original state dict, in insertion order."""
    key_sizes: dict[str, int]
    """Byte size of each parameter tensor (numel * element_size)."""
    owner_of: dict[str, int]
    """Maps each key to the rank that stores it."""
    local_shard: dict[str, torch.Tensor]
    """Tensors owned by this rank (subset of the full state dict)."""
    rank: int
    """This process's rank within the group."""
    world: int
    """Total number of ranks in the group."""
    group: dist.ProcessGroup
    """NCCL process group used for broadcast operations."""
    _owner_groups: dict[int, list[str]]
    """Keys grouped by owning rank, sorted by descending tensor size."""

    @classmethod
    def from_state_dict(
        cls,
        sd: dict[str, torch.Tensor],
        group: dist.ProcessGroup,
        clone: bool = True,
    ) -> ShardedSD:
        rank = dist.get_rank(group)
        world = dist.get_world_size(group)

        keys = tuple(sd.keys())
        # Co-locate .weight_scale with its .weight on the same rank.
        owner_of: dict[str, int] = {}
        for k in keys:
            if k.endswith(".weight_scale"):
                parent = k.replace(".weight_scale", ".weight")
                if parent in sd:
                    owner_of[k] = _stable_owner(parent, world)
                    continue
            owner_of[k] = _stable_owner(k, world)

        key_sizes = {k: _nbytes(v) for k, v in sd.items()}

        local_shard: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if owner_of[k] == rank:
                local_shard[k] = v.clone() if clone else v

        owner_groups: dict[int, list[str]] = {r: [] for r in range(world)}
        for k in keys:
            owner_groups[owner_of[k]].append(k)
        for r in range(world):
            owner_groups[r].sort(key=lambda kk: key_sizes[kk], reverse=True)

        return cls(
            keys=keys,
            key_sizes=key_sizes,
            owner_of=owner_of,
            local_shard=local_shard,
            rank=rank,
            world=world,
            group=group,
            _owner_groups=owner_groups,
        )

    def subset(self, keys: Iterable[str]) -> ShardedSD:
        """The same shards restricted to ``keys`` (no copies): restoring it writes only those keys.
        Every rank must pass the same keys, as for :meth:`broadcast_shards_into`.
        """
        wanted = frozenset(keys)
        return replace(
            self,
            keys=tuple(k for k in self.keys if k in wanted),
            key_sizes={k: v for k, v in self.key_sizes.items() if k in wanted},
            owner_of={k: v for k, v in self.owner_of.items() if k in wanted},
            local_shard={k: v for k, v in self.local_shard.items() if k in wanted},
            _owner_groups={r: [k for k in ks if k in wanted] for r, ks in self._owner_groups.items()},
        )

    def broadcast_shards_into(
        self,
        target_sd: dict[str, torch.Tensor],
        staging: torch.Tensor,
    ) -> None:
        """Restore the stored weights from the sharded backup into *target_sd*.
        This is a **cooperative operation** — all ranks in the process group
        must call it simultaneously.
        *staging* is a caller-owned ``uint8`` scratch buffer: one send slot plus ``world``
        receive slots are carved from it, and tensors larger than a slot split across rounds.
        It may be shared by instances that never broadcast at the same time. In round ``i``
        every rank packs its ``i``-th bucket and one ``all_gather_into_tensor`` ships all of
        them, so every link direction carries data at once; the pack and the unpack of a round
        are single multi-tensor copies. Writes directly into existing tensors in *target_sd*.
        """
        if staging.dtype != torch.uint8 or staging.numel() == 0:
            raise ValueError("staging must be a non-empty uint8 buffer")
        slot = (staging.numel() // (self.world + 1)) // _SLOT_ALIGN * _SLOT_ALIGN
        if slot <= 0:
            raise ValueError(f"staging buffer of {staging.numel()} bytes is too small for {self.world} ranks")
        plans = {owner: self._plan_rounds(keys, slot) for owner, keys in self._owner_groups.items()}
        n_rounds = max((len(rounds) for rounds in plans.values()), default=0)
        send = staging[:slot]
        recv = staging[slot : slot + self.world * slot]
        for i in range(n_rounds):
            mine = plans[self.rank][i] if i < len(plans[self.rank]) else []
            if mine:
                pack_dst: list[torch.Tensor] = []
                pack_src: list[torch.Tensor] = []
                filled = 0
                for key, offset, chunk in mine:
                    pack_dst.append(send[filled : filled + chunk])
                    pack_src.append(_as_bytes(self.local_shard[key], key)[offset : offset + chunk])
                    filled += chunk
                torch._foreach_copy_(pack_dst, pack_src)
            dist.all_gather_into_tensor(recv, send, group=self.group)
            unpack_dst: list[torch.Tensor] = []
            unpack_src: list[torch.Tensor] = []
            for owner, rounds in plans.items():
                if i >= len(rounds):
                    continue
                cursor = owner * slot
                for key, offset, chunk in rounds[i]:
                    dst = target_sd[key]
                    # A zero-copy shard is the target tensor itself: nothing to write back.
                    if owner != self.rank or dst.data_ptr() != self.local_shard[key].data_ptr():
                        unpack_dst.append(_as_bytes(dst, key)[offset : offset + chunk])
                        unpack_src.append(recv[cursor : cursor + chunk])
                    cursor += chunk
            if unpack_dst:
                torch._foreach_copy_(unpack_dst, unpack_src)

    def _plan_rounds(self, keys: list[str], capacity: int) -> list[list[tuple[str, int, int]]]:
        """Build rounds that pack a *capacity*-byte buffer, splitting tensors if needed.
        Returns a list of rounds, each containing ``(key, byte_offset, chunk_bytes)`` tuples.
        """
        rounds: list[list[tuple[str, int, int]]] = []
        current: list[tuple[str, int, int]] = []
        used = 0

        for k in keys:
            remaining = self.key_sizes[k]
            offset = 0

            while remaining > 0:
                space = capacity - used
                if space == 0:
                    rounds.append(current)
                    current = []
                    used = 0
                    space = capacity

                chunk = min(remaining, space)
                current.append((k, offset, chunk))
                used += chunk
                offset += chunk
                remaining -= chunk

        if current:
            rounds.append(current)

        return rounds
