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


def flush_torch_l2_cache() -> None:
    from cutlass.utils import HardwareInfo

    l2_cache_bytes = HardwareInfo().get_l2_cache_size_in_bytes()
    l2_flush_buffer = torch.empty(
        l2_cache_bytes * 2 // 4,
        dtype=torch.int32,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    l2_flush_buffer.zero_()


def benchmark_torch(
    fn: Callable,
    workspace_generator: Callable,
    workspace_count: int = 1,
    warmup_iterations: int = 10,
    iterations: int = 100,
) -> float:
    if not callable(fn):
        raise TypeError("fn must be callable")
    if not callable(workspace_generator):
        raise TypeError("workspace_generator must be callable")
    if workspace_count < 1:
        raise ValueError("workspace_count must be at least 1")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    workspaces = [workspace_generator() for _ in range(workspace_count)]

    def loop_and_call(iteration_count: int, workspace_index: int = 0) -> int:
        for _ in range(iteration_count):
            workspace = workspaces[workspace_index]
            fn(*workspace)
            workspace_index = (workspace_index + 1) % workspace_count
        return workspace_index

    stream = torch.cuda.current_stream()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream):
        if workspace_count > 1:
            flush_torch_l2_cache()

        workspace_index = loop_and_call(warmup_iterations)
        start_event.record(stream)
        loop_and_call(iterations, workspace_index)
        end_event.record(stream)

    end_event.synchronize()
    return start_event.elapsed_time(end_event) / iterations * 1e3


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
