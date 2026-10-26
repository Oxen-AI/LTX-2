"""Distributed transformer weight tracker with LoRA hot-swap.
Shared infrastructure used by both TDP and SP builders.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable

import safetensors
import torch
import torch.distributed as dist

from ltx_core.loader.fuse_loras import fuse_lora_weights
from ltx_core.loader.primitives import LoraPathStrengthAndSDOps, LoraStateDictWithStrength, StateDict
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.model_protocol import ModelType
from ltx_core.multigpu.sharded_sd import ShardedSD


def _apply_loras_inplace(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    loras: tuple[LoraPathStrengthAndSDOps, ...],
    builder: Builder,  # type: ignore[type-arg]
    lora_keys: frozenset[str],
) -> list[str]:
    """Reset *target* to clean weights from *source*, then fuse all LoRAs in one pass.
    Returns the keys the fusion wrote.
    """
    for key, clean_weight in source.items():
        target[key].copy_(clean_weight)

    lora_sds = [
        LoraStateDictWithStrength(
            builder.load_sd(
                [lora.path],
                sd_ops=lora.sd_ops.with_additional_allowed_keys(lora_keys),
                registry=builder.registry,
                device=builder.lora_load_device,
            ),
            lora.strength,
        )
        for lora in loras
        if lora.strength != 0
    ]
    target_sd = StateDict(
        sd=target, device=next(iter(target.values())).device, size=0, dtype={next(iter(target.values())).dtype}
    )
    written: list[str] = []
    for key, fused in fuse_lora_weights(target_sd, lora_sds, fuse_rule=builder.fuse_rule):
        target[key].copy_(fused)
        written.append(key)
    return written


@functools.cache
def _lora_module_prefixes(path: str, sd_ops: SDOps) -> frozenset[str]:
    """The model modules a LoRA file targets, from its header's ``lora_A`` keys mapped through ``sd_ops``."""
    suffix = ".lora_A.weight"
    with safetensors.safe_open(path, framework="pt") as f:
        names = [sd_ops.apply_to_key(key) for key in f.keys()]  # noqa: SIM118 -- safe_open is not a mapping
    return frozenset(name[: -len(suffix)] for name in names if name is not None and name.endswith(suffix))


def lora_footprint(keys: Iterable[str], loras: Iterable[LoraPathStrengthAndSDOps]) -> frozenset[str]:
    """The keys of ``keys`` a LoRA set can write: every tensor of a module one of its non-zero LoRAs targets.
    That covers the fused ``.weight`` and any companion a fuse rule rewrites with it (``.weight_scale``, ...).
    Read from the LoRA files' headers, so every rank derives the same set without a collective.
    """
    prefixes: set[str] = set()
    for lora in loras:
        if lora.strength != 0:
            prefixes |= _lora_module_prefixes(lora.path, lora.sd_ops)
    return frozenset(key for key in keys if key.rpartition(".")[0] in prefixes)


class TransformerWeightTracker:
    """Tracks cached transformer weights with distributed LoRA hot-swap.
    Shared across stage builders that operate on the same checkpoint.
    Does **not** own the model weights — it references tensors stored in a
    :class:`Registry` and receives a builder at :meth:`build` time.
    Uses two :class:`ShardedSD` instances (created on first :meth:`build` call):
    - ``stored_sd`` — cloned backup of the original (pre-LoRA) weights.
      Used to restore registry tensors before applying a different LoRA set.
    - ``broadcast_sd`` — zero-copy view into the registry tensors.
      After in-place LoRA fusion on the owning rank, this broadcasts the
      fused results to all other ranks so every rank sees the same weights.
    Both are created together and are always either both ``None`` or both set.
    A swap touches only the LoRA footprints (:func:`lora_footprint`): the keys the outgoing set
    wrote (``dirty``) are restored and the incoming set's are re-fused, and only their union is
    broadcast. Weights neither set targets stay as they are, clean.
    With ``no_lora_swap``, the LoRA set is assumed fixed (none, or one set):
    the backup clone is skipped and any swap or reset raises.
    """

    def __init__(self, group: dist.ProcessGroup, bucket_mb: int = 256, no_lora_swap: bool = False) -> None:
        if bucket_mb <= 0:
            raise ValueError("bucket_mb must be > 0")
        self._group = group
        self._bucket_mb = bucket_mb
        self._no_lora_swap = no_lora_swap
        self._staging: torch.Tensor | None = None
        self.stored_sd: ShardedSD | None = None
        self.broadcast_sd: ShardedSD | None = None
        self.loras: list[tuple[str, float, str]] = []
        self.dirty: frozenset[str] = frozenset()  # keys holding LoRA-fused values, the same on every rank

    @property
    def staging(self) -> torch.Tensor:
        """The single broadcast scratch buffer, shared by both SDs, allocated on first use."""
        if self._staging is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            self._staging = torch.empty(self._bucket_mb * 1024 * 1024, dtype=torch.uint8, device=device)
        return self._staging

    def loras_match(self, lora_list: list[tuple[str, float, str]]) -> bool:
        if len(lora_list) != len(self.loras):
            return False
        return sorted(lora_list) == sorted(self.loras)

    def reset_loras(self, target_sd: dict[str, torch.Tensor]) -> None:
        """Restore *target_sd* to original (pre-LoRA) weights: the keys the applied LoRAs wrote.
        No-op if no LoRAs are currently applied. This is a cooperative
        operation — all ranks must call it simultaneously.
        """
        if not self.loras:
            return
        if self._no_lora_swap:
            raise RuntimeError("no_lora_swap tracker has no backup to reset from")
        if self.stored_sd is None:
            raise RuntimeError("stored_sd must be initialised before reset_loras")
        dirty, self.loras, self.dirty = self.dirty, [], frozenset()
        self.stored_sd.subset(dirty).broadcast_shards_into(target_sd, self.staging)

    def _local_lora_keys(self) -> frozenset[str]:
        """Derive LoRA key names from the locally owned model keys."""
        if self.stored_sd is None:
            return frozenset()

        keys: set[str] = set()
        for k in self.stored_sd.local_shard:
            if k.endswith(".weight"):
                prefix = k[: -len(".weight")]
                keys.add(f"{prefix}.lora_A.weight")
                keys.add(f"{prefix}.lora_B.weight")
        return frozenset(keys)

    def apply_loras_(
        self,
        target_sd: dict[str, torch.Tensor],
        loras: tuple[LoraPathStrengthAndSDOps, ...],
        builder: Builder,  # type: ignore[type-arg]
    ) -> None:
        """Fuse *loras* into *target_sd* in-place (trailing ``_`` denotes in-place).
        Skips work when the requested LoRAs already match. Restores the stored
        weights of the outgoing set's footprint, fuses the new set into its own,
        and broadcasts only the union. This is a cooperative operation — all
        ranks must call it simultaneously.
        """
        new_loras = [(lora.path, lora.strength, lora.sd_ops.name) for lora in loras]

        if self.loras_match(new_loras):
            return

        if self._no_lora_swap and self.loras:
            raise RuntimeError(f"no_lora_swap tracker cannot change LoRAs: have {self.loras}, requested {new_loras}")

        if all(lora.strength == 0 for lora in loras):
            self.reset_loras(target_sd)
            return

        if self.stored_sd is None or self.broadcast_sd is None:
            raise RuntimeError("ShardedSDs must be initialised before apply_loras_ (call build first)")

        footprint = lora_footprint(self.stored_sd.keys, loras)
        touched = footprint | self.dirty
        source = {k: v for k, v in self.stored_sd.local_shard.items() if k in touched}
        target = {k: target_sd[k] for k in source}
        written = _apply_loras_inplace(source, target, loras, builder, self._local_lora_keys()) if target else []
        stray = [key for key in written if key not in footprint]
        if stray:
            # Other ranks would never receive these: refuse rather than leave the ranks' weights apart.
            raise RuntimeError(f"LoRA fusion wrote keys outside the LoRA set's footprint: {stray[:5]}")

        self.broadcast_sd.subset(touched).broadcast_shards_into(target_sd, self.staging)
        self.loras = new_loras
        self.dirty = footprint

    def build(
        self,
        builder: Builder[ModelType],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: object,
    ) -> ModelType:
        """Build the transformer model with distributed LoRA hot-swap.
        Builds (or reuses a registry-cached shell with rebound clean weights) via
        ``clean_builder.build``. LoRAs are then fused in-place into the registry
        tensors and broadcast. Assumes a non-dummy :class:`Registry`.
        """
        loras = builder.loras
        clean_builder = builder.with_loras(())

        model_paths = list(builder.model_path) if isinstance(builder.model_path, tuple) else [builder.model_path]

        model = clean_builder.build(device=device, dtype=dtype, **kwargs)

        cached_sd = clean_builder.registry.get(model_paths, clean_builder.model_sd_ops)
        if cached_sd is None:
            raise RuntimeError("Expected model state dict in registry but found None")

        if self.stored_sd is None:
            self.stored_sd = ShardedSD.from_state_dict(cached_sd.sd, self._group, clone=not self._no_lora_swap)
            self.broadcast_sd = ShardedSD.from_state_dict(cached_sd.sd, self._group, clone=False)

        if loras:
            self.apply_loras_(cached_sd.sd, loras, builder)
        else:
            self.reset_loras(cached_sd.sd)

        return model
