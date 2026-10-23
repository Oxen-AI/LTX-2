import logging
import os
from pathlib import Path

import pytest
import safetensors
import safetensors.torch
import torch

from ltx_core.loader.sft_loader import SafetensorsStateDictLoader


def _write_shard(path: Path) -> dict[str, torch.Tensor]:
    tensors = {"a.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3), "b.bias": torch.ones(4)}
    safetensors.torch.save_file(tensors, str(path))
    return tensors


def _refuse_mmap(monkeypatch: pytest.MonkeyPatch, error: OSError) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(safetensors, "safe_open", refuse)


def test_load_reads_shard_into_memory_when_mmap_is_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # /dev/null has no mmap on Linux or macOS, so this is the exact error safetensors raises
    # on a filesystem that refuses mmap.
    with pytest.raises(OSError, match="os error 19") as mmap_failure:
        safetensors.safe_open(os.devnull, framework="pt")
    shard = tmp_path / "model.safetensors"
    expected = _write_shard(shard)
    _refuse_mmap(monkeypatch, mmap_failure.value)

    with caplog.at_level(logging.WARNING):
        state_dict = SafetensorsStateDictLoader().load(str(shard), sd_ops=None)

    assert state_dict.sd.keys() == expected.keys()
    for key, tensor in expected.items():
        torch.testing.assert_close(state_dict.sd[key], tensor)
    assert "reading it into memory" in caplog.text


def test_load_raises_other_os_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shard = tmp_path / "model.safetensors"
    _write_shard(shard)
    _refuse_mmap(monkeypatch, OSError("Permission denied (os error 13)"))

    with pytest.raises(OSError, match="os error 13"):
        SafetensorsStateDictLoader().load(str(shard), sd_ops=None)
