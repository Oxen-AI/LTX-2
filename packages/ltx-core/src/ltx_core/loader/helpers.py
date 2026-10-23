"""Shared model-construction helpers used by both SingleGPUModelBuilder and StreamingModelBuilder."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from os import PathLike, fspath
from typing import TypeVar

import torch
from torch import nn
from torch.nn.parameter import is_lazy

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import StateDict, StateDictLoader
from ltx_core.loader.registry import Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.model_protocol import ModelConfigurator

_M = TypeVar("_M", bound=nn.Module)
_PathArg = str | PathLike[str]


def _customizes_loading(module: nn.Module) -> bool:
    cls = type(module)
    return (
        cls._load_from_state_dict is not nn.Module._load_from_state_dict
        or cls.set_extra_state is not nn.Module.set_extra_state
        or bool(module._load_state_dict_pre_hooks)
        or bool(module._load_state_dict_post_hooks)
    )


def assign_state_dict(module: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    """Same result as ``module.load_state_dict(state_dict, strict=False, assign=True)``, in one pass.
    Each parameter / persistent buffer looks its key up directly, where ``nn.Module.load_state_dict``
    re-filters the whole state dict for every child module. Keys without a matching tensor are left
    untouched and unmatched entries are ignored (``strict=False``); a shape mismatch raises.
    Falls back to ``load_state_dict`` when any submodule customizes loading (an overridden
    ``_load_from_state_dict`` or ``set_extra_state``, load hooks) or swap-tensors loading is on,
    because those need the stock per-module protocol.
    """
    modules = list(module.named_modules(remove_duplicate=False))
    if torch.__future__.get_swap_module_params_on_conversion() or any(_customizes_loading(m) for _, m in modules):
        module.load_state_dict(state_dict, strict=False, assign=True)
        return

    error_msgs: list[str] = []
    for prefix, submodule in modules:
        key_prefix = f"{prefix}." if prefix else ""
        for name, param in submodule._parameters.items():
            tensor = None if param is None else state_dict.get(key_prefix + name)
            if tensor is None:
                continue
            tensor = _matching_shape(tensor, param, key_prefix + name, error_msgs)
            if tensor is None:
                continue
            if isinstance(tensor, nn.Parameter):
                tensor.requires_grad_(param.requires_grad)
            else:
                tensor = nn.Parameter(tensor, requires_grad=param.requires_grad)
            submodule._parameters[name] = tensor
        for name, buffer in submodule._buffers.items():
            if buffer is None or name in submodule._non_persistent_buffers_set:
                continue
            tensor = state_dict.get(key_prefix + name)
            if tensor is None:
                continue
            tensor = _matching_shape(tensor, buffer, key_prefix + name, error_msgs)
            if tensor is not None:
                submodule._buffers[name] = tensor
    if error_msgs:
        raise RuntimeError(f"Error(s) in loading state_dict for {type(module).__name__}:\n\t" + "\n\t".join(error_msgs))


def _matching_shape(
    tensor: torch.Tensor, current: torch.Tensor, key: str, error_msgs: list[str]
) -> torch.Tensor | None:
    """``tensor`` reshaped the way ``load_state_dict`` accepts it for ``current``, or ``None`` on a mismatch.
    A lazy ``current`` (``UninitializedParameter`` / ``UninitializedBuffer``) has no shape yet and takes
    ``tensor`` as is, as in ``load_state_dict``.
    """
    if is_lazy(current):
        return tensor
    if current.ndim == 0 and tensor.shape == (1,):
        tensor = tensor[0]
    if tensor.shape != current.shape:
        error_msgs.append(
            f"size mismatch for {key}: copying a param with shape {tensor.shape} from checkpoint, "
            f"the shape in current model is {current.shape}."
        )
        return None
    return tensor


def module_tensors(model: nn.Module) -> Iterator[torch.Tensor]:
    """Every parameter and buffer of ``model``, read straight off each module.
    Unlike ``named_parameters`` / ``named_buffers`` there is no de-duplication, so a tensor
    shared between modules is yielded once per owner.
    """
    for module in model.modules():
        yield from (t for t in module._parameters.values() if t is not None)
        yield from (t for t in module._buffers.values() if t is not None)


def to_device(model: _M, device: torch.device) -> _M:
    """``model.to(device)``, skipped when every parameter and buffer already lives on ``device``.
    ``nn.Module.to`` rebuilds every parameter even when nothing moves, which costs a full walk of
    a large model per call.
    """
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if all(t.device == device for t in module_tensors(model)):
        return model
    return model.to(device)


def as_path_list(paths: _PathArg | tuple[_PathArg, ...] | list[_PathArg]) -> list[str]:
    """Normalize checkpoint path args to the ``list[str]`` shape registries key on."""
    if isinstance(paths, (str, PathLike)):
        return [fspath(paths)]
    return [fspath(p) for p in paths]


def load_state_dict(
    paths: _PathArg | tuple[_PathArg, ...] | list[_PathArg],
    loader: StateDictLoader,
    registry: Registry,
    device: torch.device | None,
    sd_ops: SDOps | None = None,
) -> StateDict:
    """Load a state dict from disk, using registry caching."""
    path_list = as_path_list(paths)
    cached = registry.get(path_list, sd_ops)
    if cached is not None:
        return cached
    result = loader.load(path_list, sd_ops=sd_ops, device=device)
    # ``add`` returns the retained copy when the registry rewrites storage (e.g. pin to CPU).
    return registry.add(path_list, sd_ops=sd_ops, state_dict=result)


def read_model_metadata(
    model_path: str | tuple[str, ...],
    loader: StateDictLoader,
) -> dict:
    """Read the full metadata dict from the first shard of a checkpoint."""
    first = model_path[0] if isinstance(model_path, tuple) else model_path
    return loader.metadata(first)


class LazyModelMetadata:
    """A checkpoint's parsed metadata, read on first access and shared by every builder copy holding it."""

    def __init__(self, model_path: str | tuple[str, ...], loader: StateDictLoader) -> None:
        self._model_path = model_path
        self._loader = loader
        self._metadata: dict | None = None

    def get(self) -> dict:
        if self._metadata is None:
            self._metadata = read_model_metadata(self._model_path, self._loader)
        return copy.deepcopy(self._metadata)


def read_model_config(
    model_path: str | tuple[str, ...],
    loader: StateDictLoader,
) -> dict:
    """Read the ``config`` sub-dict (transformer/vae/scheduler/...) from a checkpoint's metadata."""
    return read_model_metadata(model_path, loader).get("config", {})


def parse_model_version(version: str | None) -> tuple[int, ...]:
    """Parse a checkpoint's ``model_version`` into comparable numeric components.
    Parsing stops at the first dot-separated component that isn't a plain integer, so
    pre-release tags are dropped (the pre-2.4 era used strings like ``"2.3.rc1"``). A
    shorter numeric prefix already compares as "less than" a longer one sharing its
    leading components (``(2, 3) < (2, 4, 0)``), so dropping the tail is safe. An unset
    or non-numeric version parses to ``()``, which compares below every real version --
    callers get their oldest fallback.
    Tags are not always dot-separated (``"2.4-rc2"`` parses to ``(2,)``). Callers that
    want such a build to compare equal to its own generation should normalize the
    separator first -- see ``detect_params`` in ltx-pipelines.
    """
    if not version:
        return ()
    numeric_parts = []
    for part in version.split("."):
        if not part.isdigit():
            break
        numeric_parts.append(int(part))
    return tuple(numeric_parts)


def create_meta_model(
    configurator: type[ModelConfigurator[_M]],
    metadata: dict,
    module_ops: tuple[ModuleOps, ...] = (),
) -> _M:
    """Create a model on the meta device and apply module operations."""
    with torch.device("meta"):
        model = configurator.from_metadata(metadata)
    for op in module_ops:
        if op.matcher(model):
            model = op.mutator(model)
    return model
