import errno
import json
import logging
from pathlib import Path

import safetensors
import safetensors.torch
import torch

from ltx_core.loader.primitives import StateDict, StateDictLoader
from ltx_core.loader.sd_ops import SDOps

logger: logging.Logger = logging.getLogger(__name__)


class _InMemoryShard:
    """Serves keys() and get_tensor() like a safe_open handle, from a shard read fully into memory."""

    def __init__(self, path: str) -> None:
        self._tensors = safetensors.torch.load(Path(path).read_bytes())

    def __enter__(self) -> "_InMemoryShard":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._tensors.clear()

    def keys(self) -> list[str]:
        return list(self._tensors)

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]


def _mmap_unsupported(error: OSError) -> bool:
    # safetensors raises a failed mmap from Rust as OSError("No such device (os error 19)")
    # with errno unset, which leaves the message as the only place the error number appears.
    return error.errno == errno.ENODEV or f"(os error {errno.ENODEV})" in str(error)


def _open_shard(shard_path: str, device: torch.device) -> safetensors.safe_open | _InMemoryShard:
    # Some network filesystems refuse mmap with ENODEV. Reading the shard instead costs its
    # whole size in RAM (tens of GB for a full checkpoint), so it is only the fallback.
    try:
        return safetensors.safe_open(shard_path, framework="pt", device=str(device))
    except OSError as error:
        if not _mmap_unsupported(error):
            raise
        logger.warning("Cannot memory map %s (%s), reading it into memory instead", shard_path, error)
    return _InMemoryShard(shard_path)


class SafetensorsStateDictLoader(StateDictLoader):
    """
    Loads weights from safetensors files without metadata support.
    Use this for loading raw weight files. For model files that include
    configuration metadata, use SafetensorsModelStateDictLoader instead.
    """

    def metadata(self, path: str) -> dict:
        raise NotImplementedError("Not implemented")

    def load(self, path: str | list[str], sd_ops: SDOps, device: torch.device | None = None) -> StateDict:
        """
        Load state dict from path or paths (for sharded model storage) and apply sd_ops
        """
        sd = {}
        size = 0
        dtype = set()
        device = device or torch.device("cpu")
        model_paths = path if isinstance(path, list) else [path]
        for shard_path in model_paths:
            with _open_shard(shard_path, device) as f:
                safetensor_keys = f.keys()
                for name in safetensor_keys:
                    expected_name = name if sd_ops is None else sd_ops.apply_to_key(name)
                    if expected_name is None:
                        continue
                    value = f.get_tensor(name).to(device=device, non_blocking=True, copy=False)
                    key_value_pairs = ((expected_name, value),)
                    if sd_ops is not None:
                        key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
                    for key, value in key_value_pairs:
                        size += value.nbytes
                        dtype.add(value.dtype)
                        sd[key] = value

        return StateDict(sd=sd, device=device, size=size, dtype=dtype)


class SafetensorsModelStateDictLoader(StateDictLoader):
    """
    Loads weights and configuration metadata from safetensors model files.
    Unlike SafetensorsStateDictLoader, this loader can read model configuration
    from the safetensors file metadata via the metadata() method.
    """

    def __init__(self, weight_loader: SafetensorsStateDictLoader | None = None):
        self.weight_loader = weight_loader if weight_loader is not None else SafetensorsStateDictLoader()

    def metadata(self, path: str) -> dict:
        """Read the full safetensors ``__metadata__`` dict.
        Values are JSON string-encoded by convention (e.g. ``config``, ``gemma_source_checkpoint``);
        each is parsed if it's valid JSON and left as a raw string otherwise (e.g. ``license``,
        ``model_version``).
        """
        with safetensors.safe_open(path, framework="pt") as f:
            meta = f.metadata()
            if meta is None:
                return {}
            parsed = {}
            for key, value in meta.items():
                try:
                    parsed[key] = json.loads(value)
                except json.JSONDecodeError:
                    parsed[key] = value
            return parsed

    def load(self, path: str | list[str], sd_ops: SDOps | None = None, device: torch.device | None = None) -> StateDict:
        return self.weight_loader.load(path, sd_ops, device)
