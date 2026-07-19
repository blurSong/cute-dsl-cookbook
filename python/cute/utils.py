import torch
import math
import cutlass
import cutlass.cute as cute
from typing import Callable, List
from cutlass.base_dsl.typing import __STR_TO_DTYPE__

VERBOSE = False
LOG = "[CuTe Info]"


def check_cuda():
    assert torch.cuda.is_available(), "NO CUDA device detected."


def get_cutlass_dtype(type: str):
    for k, v in __STR_TO_DTYPE__.items():
        if type == k.lower():
            return v
    raise ValueError(f"Unknown type: {type}")


def make_rasterized_grid(M: int, N: int, m: int, n: int):
    # rasterize gemm grid dimension
    dim_x = math.ceil(N / n)
    dim_y = math.ceil(M / m)
    # tile_size_X = m * K * gemm_dtype.dtype.width // 8
    # tile_size_Y = n * K * gemm_dtype.dtype.width // 8
    raster = 8 if dim_y > 4 else (4 if dim_y > 2 else (2 if dim_y > 1 else 1))
    grid_rasterized = [dim_x * raster, math.ceil(dim_y / raster), 1]
    return raster, grid_rasterized


def derasterize(x, y, f):
    new_x = x // f
    new_y = (x % f) + (y * f)
    return (new_x, new_y)


def benchmark_torch(
    fn: Callable,
    workspace_generator: Callable,
    workspace_count: int = 1,
    warmup_iterations: int = 10,
    iterations: int = 100,
):
    assert fn is not None
    assert workspace_generator is not None
    assert warmup_iterations >= 0
    assert iterations > 0

    workspaces = [workspace_generator() for _ in range(workspace_count)]

    workspace_index = 0
    torch.cuda.empty_cache()
    for _ in range(warmup_iterations):
        workspace = workspaces[workspace_index]
        fn(*workspace)
        workspace_index = (workspace_index + 1) % workspace_count
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iterations):
        workspace = workspaces[workspace_index]
        fn(*workspace)
        workspace_index = (workspace_index + 1) % workspace_count
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / iterations
    return avg_time_ms * 1e3  # return in microseconds

