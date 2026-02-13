"""
Matrix Transpose Example using CUTE DSL.

Key optimizations:

- Coalesced global memory accesses.
- Use shared memory and swizzle to avoid bank conflicts.

References:

- An Efficient Matrix Transpose in CUDA C/C++
  https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

- Tutorial: Matrix Transpose in CUTLASS
  https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/

- CuTe Matrix Transpose
  https://leimao.github.io/article/CuTe-Matrix-Transpose/

"""

import argparse
import copy
import math
import time
from functools import partial
from typing import Union

import cuda.bindings.driver as cuda
import torch
from utils import *

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu as nvgpu
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

VERBOSE = False
LOG = "[CuTe Info]"

# Here, in Lei's blog, they use (32, 64) for cta tiler and
# (8, 32)/(32, 8) for thread tiler.
# We follow the same setting here for better comparison
cta_tiler = (64, 64)
thr_tiler = (8, 32)
thr_tiler_t = (32, 8)
assert all(cta_tiler[i] % thr_tiler[i] == 0 for i in range(2))
assert all(cta_tiler[i] % thr_tiler_t[i] == 0 for i in range(2))


def constexpr_log2(x: int) -> int:
    return 0 if x < 2 else 1 + constexpr_log2(x // 2)


@cutlass.dsl_user_op
def get_swizzle_bms(dtype: cute.Numeric, *, loc=None, ip=None):
    m = constexpr_log2(1)
    b = constexpr_log2(32 * 32 // dtype.width) - m
    s = constexpr_log2(cta_tiler[1]) - m
    return b, m, s


@cute.kernel
def transpose_naive_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gCrd: cute.Tensor,
    thr_layout: cute.Layout,
    global_shape: cutlass.Shape,
):
    blk_x, blk_y, _ = cute.arch.block_idx()
    thr_x, _, _ = cute.arch.thread_idx()

    # Note that here y denotes the grid row index of res_m
    # and x denotes the grid column index of res_n
    cta_coord = ((None, None), blk_y, blk_x)
    gA = gA[cta_coord]
    gB = gB[cta_coord]
    tAgA = cute.local_partition(gA, thr_layout, thr_x)
    tBgB = cute.local_partition(gB, thr_layout, thr_x)

    tArA = cute.make_rmem_tensor_like(tAgA)

    if VERBOSE:
        print(f"gA: {gA}")
        print(f"gB: {gB}")
        print(f"tAgA: {tAgA}")
        print(f"tBgB: {tBgB}")

    # Make coordinate prediction
    gCrd = gCrd[cta_coord]
    tAgCrd = cute.local_partition(gCrd, thr_layout, thr_x)
    tArPrd = cute.make_rmem_tensor_like(tAgCrd, cutlass.Boolean)

    for i in range(cute.size(tArPrd)):
        tArPrd[i] = cute.elem_less(tAgCrd[i], global_shape)

    copy_atom = cute.make_copy_atom(nvgpu.CopyUniversalOp(), gA.element_type)

    cute.copy(copy_atom, tAgA, tArA, pred=tArPrd)
    cute.copy(copy_atom, tArA, tBgB, pred=tArPrd)


@cute.kernel
def transpose_smem_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gCrd: cute.Tensor,
    thr_layout_a: cute.Layout,
    thr_layout_b: cute.Layout,
    smem_layout: cute.Layout | cute.ComposedLayout,
    copy_atom_a: cute.CopyAtom,
    copy_atom_b: cute.CopyAtom,
    global_shape: cutlass.Shape,
):
    blk_x, blk_y, _ = cute.arch.block_idx()
    thr_x, _, _ = cute.arch.thread_idx()

    sA = cutlass.utils.SmemAllocator().allocate_tensor(
        gA.element_type,
        smem_layout,
    )
    sB = cute.make_tensor(sA.iterator, smem_layout)  # Aliasing smem for B

    cta_coord = ((None, None), blk_y, blk_x)
    gA = gA[cta_coord]
    gB = gB[cta_coord]

    tAgA = cute.local_partition(gA, thr_layout_a, thr_x)
    tBgB = cute.local_partition(gB, thr_layout_b, thr_x)
    tAsA = cute.local_partition(sA, thr_layout_a, thr_x)
    tBsB = cute.local_partition(sB, thr_layout_b, thr_x)

    if VERBOSE:
        print(f"gA: {gA}")
        print(f"gB: {gB}")
        print(f"tAgA: {tAgA}")
        print(f"tBgB: {tBgB}")
        print(f"tAsA: {tAsA}")
        print(f"tBsB: {tBsB}")

    # Make coordinate prediction
    gCrd = gCrd[cta_coord]
    tAgCrd = cute.local_partition(gCrd, thr_layout_a, thr_x)
    tBgCrd = cute.local_partition(gCrd, thr_layout_b, thr_x)
    tArPrd = cute.make_rmem_tensor_like(tAgCrd, cutlass.Boolean)
    tBrPrd = cute.make_rmem_tensor_like(tBgCrd, cutlass.Boolean)

    for i in range(cute.size(tArPrd)):
        tArPrd[i] = cute.elem_less(tAgCrd[i], global_shape)
    for i in range(cute.size(tBrPrd)):
        tBrPrd[i] = cute.elem_less(tBgCrd[i], global_shape)

    # Practically we need to create tiled-copy for A and B. But we were being lazy
    cute.basic_copy_if(tArPrd, tAgA, tAsA)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()
    # if blk_x == 0 and blk_y == 0 and thr_x == 0:
    #    cute.print_tensor(tAgA)
    #    cute.print_tensor(tAsA)
    cute.basic_copy_if(tBrPrd, tBsB, tBgB)


@cute.kernel
def transpose_tma_kernel():
    pass


@cute.jit
def transpose_naive(
    A: cute.Tensor,
    B: cute.Tensor,
    coalesce_read: cutlass.Constexpr = True,
):
    # For output matrix B, we define a column-major layout (M,N):(1,M)
    B = cute.make_tensor(B.iterator, cute.make_layout(B.shape, stride=(1, B.shape[0])))

    a_major_mode = utils.LayoutEnum.from_tensor(A)
    b_major_mode = utils.LayoutEnum.from_tensor(B)
    assert a_major_mode == utils.LayoutEnum.ROW_MAJOR
    assert b_major_mode == utils.LayoutEnum.COL_MAJOR

    # * gA ((tile_m, tile_n), res_m, res_n)
    # * gB ((tile_m, tile_n), res_m, res_n)
    gA = cute.tiled_divide(A, cta_tiler)
    gB = cute.tiled_divide(B, cta_tiler)

    Crd = cute.make_identity_tensor(A.shape)
    gCrd = cute.tiled_divide(Crd, cta_tiler)

    # - Here we define 2 thread-level layouts for coalesced access
    #   Either use (8,32) row-major for A reading coalesced or (32,8) col-major
    #   for B writing coalesced
    if cutlass.const_expr(coalesce_read):
        thr_layout = cute.make_ordered_layout(thr_tiler, order=(1, 0))
    else:
        thr_layout = cute.make_ordered_layout(thr_tiler_t, order=(0, 1))

    grid_dim = [cute.size(gA, mode=[2]), cute.size(gA, mode=[1]), 1]
    block_dim = [cute.size(thr_tiler), 1, 1]

    if VERBOSE:
        print(f"gA: {gA}")
        print(f"gB: {gB}")
        print(f"gCrd: {gCrd}")
        print(f"grid_dim: {grid_dim}")
        print(f"block_dim: {block_dim}")
        print(f"thr_layout: {thr_layout}")

    transpose_naive_kernel(
        gA,
        gB,
        gCrd,
        thr_layout,
        A.shape,
    ).launch(
        grid=grid_dim,
        block=block_dim,
    )


@cute.jit
def transpose_smem(
    A: cute.Tensor,
    B: cute.Tensor,
    padding: cutlass.Constexpr = False,
    swizzle: cutlass.Constexpr = False,
    coalesce_read: cutlass.Constexpr = False,
):
    dtype = A.element_type

    # For output matrix B, we define a column-major layout (M,N):(1,M)
    B = cute.make_tensor(B.iterator, cute.make_layout(B.shape, stride=(1, B.shape[0])))

    a_major_mode = utils.LayoutEnum.from_tensor(A)
    b_major_mode = utils.LayoutEnum.from_tensor(B)
    assert a_major_mode == utils.LayoutEnum.ROW_MAJOR
    assert b_major_mode == utils.LayoutEnum.COL_MAJOR

    # * gA ((tile_m, tile_n), res_m, res_n)
    # * gB ((tile_m, tile_n), res_m, res_n)
    gA = cute.tiled_divide(A, cta_tiler)
    gB = cute.tiled_divide(B, cta_tiler)

    Crd = cute.make_identity_tensor(A.shape)
    gCrd = cute.tiled_divide(Crd, cta_tiler)

    # Define the thread layout and smem layout
    # 1. To avoid uncloalesced GMEM access for both A and B (which is the key reason to use SMEM).
    #    We define two thread layouts for A(row-major) and B(col-major).
    # 2. Two options here when using smem for transpose:
    #    1) Load A -> transpose -> Store SMEM -> Load SMEM -> Store B
    #       The STS will be uncoalesced (i.e., bank conflict when writing SMEM)
    #    2) Load A -> Store SMEM -> Load SMEM -> transpose -> Store B
    #       The LDS will be uncoalesced (i.e., bank conflict when loading SMEM)
    #    The uncoalesced SMEM access is less harmful than uncoalesced GMEM access.
    # 3. To further optimize, we can apply swizzle or padding to avoid bank conflict.

    thr_layout_a = cute.make_ordered_layout(thr_tiler, order=(1, 0))
    thr_layout_b = cute.make_ordered_layout(thr_tiler_t, order=(0, 1))

    if cutlass.const_expr(swizzle):
        swizzle = cute.make_swizzle(*get_swizzle_bms(dtype))
        smem_layout_swizzled = cute.make_composed_layout(
            swizzle, 0, cute.make_layout(cta_tiler, stride=(cta_tiler[1], 1))
        )
    elif cutlass.const_expr(padding):
        smem_layout_padded = cute.make_layout(cta_tiler, stride=(cta_tiler[1] + 1, 1))
    elif cutlass.const_expr(coalesce_read):
        smem_layout_read_coalesced = cute.make_layout(cta_tiler, stride=(cta_tiler[1], 1))
    else:
        smem_layout_write_coalesced = cute.make_layout(
            cta_tiler, stride=(1, cta_tiler[0])
        )
    # Here the smem layout is essentially option 1) or 2) mentioned above

    if cutlass.const_expr(swizzle):
        smem_layout = smem_layout_swizzled
    elif cutlass.const_expr(padding):
        smem_layout = smem_layout_padded
    elif cutlass.const_expr(coalesce_read):
        smem_layout = smem_layout_read_coalesced
    else:
        smem_layout = smem_layout_write_coalesced

    copy_atom_a = cute.make_copy_atom(
        nvgpu.cpasync.CopyG2SOp(cache_mode=nvgpu.cpasync.LoadCacheMode.NONE),
        dtype,
        num_bits_per_copy=dtype.width,
    )
    copy_atom_b = cute.make_copy_atom(
        nvgpu.CopyUniversalOp(),
        dtype,
        num_bits_per_copy=dtype.width,
    )

    grid_dim = [cute.size(gA, mode=[2]), cute.size(gA, mode=[1]), 1]
    block_dim = [cute.size(thr_tiler), 1, 1]
    smem_size = cute.cosize(smem_layout) * dtype.width // 8

    if VERBOSE:
        print(f"A: {A}")
        print(f"B: {B}")
        print(f"gA: {gA}")
        print(f"gB: {gB}")
        print(f"grid_dim: {grid_dim}")
        print(f"block_dim: {block_dim}")
        print(f"smem_layout: {smem_layout}")
        print(f"thr_layout_a: {thr_layout_a}")
        print(f"thr_layout_b: {thr_layout_b}")
        print(f"copy_atom_a: {copy_atom_a}")
        print(f"copy_atom_b: {copy_atom_b}")

    transpose_smem_kernel(
        gA,
        gB,
        gCrd,
        thr_layout_a,
        thr_layout_b,
        smem_layout,
        copy_atom_a,
        copy_atom_b,
        A.shape,
    ).launch(
        grid=grid_dim,
        block=block_dim,
        smem=smem_size,
    )


@cute.jit
def transpose_tma():
    pass


def run_transpose(
    M: int,
    N: int,
    dtype: cutlass.Numeric = cutlass.Float32,
    warmup_iterations: int = 5,
    iterations: int = 100,
    dynamic_layout: bool = False,
    skip_verify: bool = False,
):
    print(f"Running transpose with M={M}, N={N}, dtype={dtype}")

    torch_stream = torch.cuda.Stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    torch_dtype = cutlass_torch.dtype(dtype)

    def _tensor_generator(return_torch=False):
        a_torch = torch.arange(M * N).view(M, N).to(dtype=torch_dtype).to("cuda")
        b_torch = torch.empty((M, N)).to(dtype=torch_dtype).to("cuda")

        a_cute = (
            from_dlpack(a_torch)
            if not dynamic_layout
            else from_dlpack(a_torch).mark_layout_dynamic()
        )
        b_cute = (
            from_dlpack(b_torch)
            if not dynamic_layout
            else from_dlpack(b_torch).mark_layout_dynamic()
        )

        if return_torch:
            return a_cute, b_cute, a_torch, b_torch

        return a_cute, b_cute

    kernels = {
        "naive-coalesce-read": partial(transpose_naive, coalesce_read=True),
        "naive-coalesce-write": partial(transpose_naive, coalesce_read=False),
        "smem-coalesce-read": partial(transpose_smem, coalesce_read=True),
        "smem-coalesce-write": partial(transpose_smem, coalesce_read=False),
        "smem-padding": partial(transpose_smem, padding=True),
        "smem-swizzle": partial(transpose_smem, swizzle=True),
    }

    if not skip_verify:
        for kernel_name, kernel in kernels.items():
            try:
                _a_cute, _b_cute, _a_torch, _b_torch = _tensor_generator(True)
                kernel(_a_cute, _b_cute)
                torch.testing.assert_close(
                    _a_torch.transpose(0, 1), _b_torch.reshape(N, M)
                )
                print(f"Kernel {kernel_name} verification passed.")
            except Exception as e:
                print(f"Kernel {kernel_name} verification failed: {e}")
                print(_a_torch, _b_torch)
    else:
        print("Transpose correctness varification skipped.")

    workspace_bytes = M * N * dtype.width // 8
    workspace_count = testing.get_workspace_count(
        workspace_bytes, warmup_iterations, iterations
    )
    workspace_generator = lambda: testing.JitArguments(*_tensor_generator())

    for kernel_name, kernel in kernels.items():
        compile_tic = time.perf_counter()
        transpose_func = cute.compile(kernel, *_tensor_generator())
        print(
            f"Kernel {kernel_name} compiled in {time.perf_counter() - compile_tic:.4f} seconds"
        )

        # Benchmarking
        torch.cuda.empty_cache()
        average_kernel_time_us = testing.benchmark(
            transpose_func,
            workspace_generator=workspace_generator,
            workspace_count=workspace_count,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
            stream=current_stream,
        )

        average_kernel_time_ms = average_kernel_time_us / 1e3
        dram_throughput_gb_s = (
            (2 * M * N * dtype.width // 8)
            / (average_kernel_time_ms / 1e3)
            / 1024
            / 1024
            / 1024
        )
        print(
            f"Kernel {kernel.__name__} average execution time: {average_kernel_time_ms:.3f} ms"
        )
        print(f"Achieved memory throughput: {dram_throughput_gb_s:.2f} GB/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of elementwise ops to demonstrate the numpy/pytorch as input for kernels"
    )

    parser.add_argument("--M", "-M", default=128, type=int)
    parser.add_argument("--N", "-N", default=256, type=int)
    parser.add_argument("--dtype", default="int32", type=str)
    parser.add_argument("--warmup-iterations", default=5, type=int)
    parser.add_argument("--iterations", default=30, type=int)
    parser.add_argument("--dynamic-layout", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    check_cuda()

    VERBOSE = args.verbose

    run_transpose(
        args.M,
        args.N,
        dtype=get_cutlass_dtype(args.dtype),
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        dynamic_layout=args.dynamic_layout,
        skip_verify=args.skip_verify,
    )
    print("PASS!")
