import torch
import math
import cutlass
import cutlass_torch
from functools import partial
import cutlass.cute as cute
from typing import Callable, Tuple, Type
from cutlass.base_dsl.typing import __STR_TO_DTYPE__


LOG = "[CuTe Info]"


def check_cuda():
    assert torch.cuda.is_available(), "NO CUDA device detected."


def str_to_cutlass_dtype(type_str: str):
    type_str = type_str.lower()
    for k, v in __STR_TO_DTYPE__.items():
        if type_str == k.lower():
            return v
    raise ValueError(f"Unknown type: {type_str}")


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


def parse_comma_separated_ints(value: str) -> Tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("Invalid format. Expected comma-separated integers.") from exc


def create_gpu_tensors(
    torch_tensor_cpu,
    cutlass_dtype,
    is_dynamic_layout=True,
    assumed_align=16,
):
    cute_tensor, torch_tensor = cutlass_torch.cute_tensor_like(
        torch_tensor_cpu,
        cutlass_dtype,
        is_dynamic_layout,
        assumed_align,
    )

    if cutlass_dtype.is_float and cutlass_dtype.width == 8:
        f32_torch_tensor_cpu = torch_tensor_cpu.to(dtype=torch.float32)
        cute_tensor = cutlass_torch.convert_cute_tensor(
            f32_torch_tensor_cpu,
            cute_tensor,
            cutlass_dtype,
            is_dynamic_layout=is_dynamic_layout,
        )
    return cute_tensor, torch_tensor


def generate_gemm_tensors(
    M: int,
    N: int,
    K: int,
    L: int,
    a_major: str,
    b_major: str,
    c_major: str,
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    is_dynamic_layout=True,
    bytes_alignment=16,
    return_type: str = "both"
):
    assert a_major in ["m", "n"], f"Invalid a_major: {a_major}"
    assert b_major in ["m", "n"], f"Invalid b_major: {b_major}"
    assert c_major in ["m", "n"], f"Invalid c_major: {c_major}"
    assert return_type in ["both", "cute", "torch"], f"Invalid return_type: {return_type}"

    a_torch_cpu = cutlass_torch.matrix(L, M, K, a_major == "m", a_dtype)
    b_torch_cpu = cutlass_torch.matrix(L, N, K, b_major == "n", b_dtype)
    c_torch_cpu = cutlass_torch.matrix(
        L,
        M,
        N,
        c_major == "m",
        c_dtype,
        init_type=cutlass_torch.TensorInitType.SKIP,
    )

    a_cute, a_torch = create_gpu_tensors(a_torch_cpu, a_dtype, is_dynamic_layout, bytes_alignment)
    b_cute, b_torch = create_gpu_tensors(b_torch_cpu, b_dtype, is_dynamic_layout, bytes_alignment)
    c_cute, c_torch = create_gpu_tensors(c_torch_cpu, c_dtype, is_dynamic_layout, bytes_alignment)

    if return_type == "cute":
        return (a_cute, b_cute, c_cute)
    elif return_type == "torch":
        return (a_torch, b_torch, c_torch)
    return a_cute, b_cute, c_cute, a_torch, b_torch, c_torch
