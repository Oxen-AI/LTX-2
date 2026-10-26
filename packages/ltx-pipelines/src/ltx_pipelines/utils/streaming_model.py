"""Context manager for block-streaming model builds (paired with :func:`~ltx_pipelines.utils.gpu_model.gpu_model`)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.block_streaming import StreamingModelBuilder
from ltx_pipelines.utils.helpers import cleanup_memory


@contextmanager
def streaming_model(
    builder: StreamingModelBuilder,
    target_device: torch.device,
    dtype: torch.dtype,
    alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
) -> Iterator[object]:
    """Build a streaming wrapper, yield it, then tear down and free memory.
    The builder's own ``cpu_slots_count`` selects RAM vs disk streaming.
    ``teardown()`` always runs -- it releases non-memory resources (forward
    hooks, the disk I/O worker thread, open file handles) that GC would not
    reclaim promptly. ``dispose`` always metas parameter storage; ``TRIM``
    also runs ``cleanup_memory()``, while ``DEFER`` leaves the CUDA cache warm.
    """
    wrapped = builder.build(device=target_device, dtype=dtype)
    try:
        yield wrapped
    finally:
        wrapped.teardown()
        wrapped.dispose()
        if alloc_trim_strategy == AllocatorTrimStrategy.TRIM:
            cleanup_memory()
