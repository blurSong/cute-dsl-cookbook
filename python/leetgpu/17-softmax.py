"""
1D Array Softmax using CuTe.

    Note that atomicMax for float32 is not natively supported on CUDA,
    so here we use atomicMax on int64 after bit-casting float32 to int64.

Reference:

    Quack Softmax impl:
    https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md
    https://github.com/Dao-AILab/quack/blob/main/quack/softmax.py

"""

import math
import torch
import operator
from typing import Callable

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm, llvm

import cuda.bindings as cuda

# NOTE in cutlass 4.3 only import cutlass.dsl_user_op
from cutlass.base_dsl._mlir_helpers.op import dsl_user_op
from cutlass.cutlass_dsl import T


VERBOSE = False
LOG = "[CuTe Info][LeetGPU]"

threads = 256
cta_tiler = (4096,)
vl = 128 // cutlass.Float32.width


@dsl_user_op
def warp_reduce(
    val: cutlass.Float32,
    op: Callable,
    warp_reduce_size: cutlass.Int32,
    *,
    loc=None,
    ip=None,
):
    upper = cutlass.Int32(math.log2(warp_reduce_size))
    for i in range(upper - 1, -1, -1):
        offset = 1 << i
        val = op(val, cute.arch.shuffle_sync_down(val, offset))
    return val


# cute.core._Pointer()
# depracated.
@dsl_user_op
def atomic_fmax_f32_llvm(a: cutlass.Float32, gmem_ptr: cutlass.Pointer, *, loc=None, ip=None):
    addr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    a_value = a.ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [addr_i64, a_value],
        "red.global.max.i64 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atomic_add_f32(a: cutlass.Float32, gmem_ptr: cutlass.Pointer, *, loc=None, ip=None):
    nvvm.atomicrmw(res=T.f32(), op=nvvm.AtomicOpKind.FADD, ptr=gmem_ptr.llvm_ptr, a=a.ir_value())


@dsl_user_op
def atomic_max_i64(a: cutlass.Int64, gmem_ptr: cutlass.Pointer, *, loc=None, ip=None):
    nvvm.atomicrmw(res=T.i64(), op=nvvm.AtomicOpKind.MAX, ptr=gmem_ptr.llvm_ptr, a=a.ir_value())


@cute.kernel
def max_reduction_kernel(
    input: cute.Tensor,
    max_value: cute.Tensor,
    crd: cute.Tensor,
    tiled_copy: cute.TiledCopy,
    N: cutlass.Int32,
):
    thr_idx = cute.arch.thread_idx()[0]
    blk_idx = cute.arch.block_idx()[0]
    wrp_idx = cute.arch.warp_idx()
    lne_idx = cute.arch.lane_idx()

    # Allocate reduce buffer
    warp_size = cute.arch.WARP_SIZE
    num_warps = cute.ceil_div(threads, warp_size)
    reduce_buffer = cutlass.utils.SmemAllocator().allocate_tensor(
        input.element_type, cute.make_layout((num_warps,))
    )

    input_tile = input[(None, blk_idx)]
    crd_tile = crd[(None, blk_idx)]
    thr_copy = tiled_copy.get_slice(thr_idx)
    input_thr = thr_copy.partition_S(input_tile)
    crd_thr = thr_copy.partition_S(crd_tile)

    input_frag_thr = cute.make_fragment_like(input_thr)
    pred_frag_thr = cute.make_fragment_like(crd_thr, cute.Boolean)
    input_frag_thr.fill(-cutlass.Float32.inf)

    for i in cutlass.range_constexpr(cute.cosize(pred_frag_thr)):
        pred_frag_thr[i] = cute.elem_less(crd_thr[i], (N,))

    cute.copy(tiled_copy, input_thr, input_frag_thr, pred=pred_frag_thr)

    # local reduction
    thr_ssa = input_frag_thr.load()
    max_thr = thr_ssa.reduce(
        cute.ReductionOp.MAX, init_val=-cutlass.Float32.inf, reduction_profile=0
    )

    # warp reduction
    cute.arch.sync_warp()
    max_thr = warp_reduce(max_thr, cute.arch.fmax, warp_size)
    if lne_idx == 0:
        reduce_buffer[wrp_idx] = max_thr

    # block reduction
    # note here we only have 8 warps by default,
    # so just let the first thread do the final reduction
    # instead of launching another warp_reduce
    cute.arch.sync_threads()
    if thr_idx == 0:
        max_blk = -cutlass.Float32.inf
        for i in cutlass.range_constexpr(num_warps):
            max_blk = cute.arch.fmax(max_blk, reduce_buffer[i])

        # final reduction
        atomic_max_i64(max_blk.to(cutlass.Int64), max_value.iterator)


@cute.kernel
def exp_sum_kernel(
    input: cute.Tensor,
    output: cute.Tensor,
    max_value: cute.Tensor,
    sum_value: cute.Tensor,
    crd: cute.Tensor,
    tiled_copy: cute.TiledCopy,
    N: cutlass.Int32,
):
    thr_idx = cute.arch.thread_idx()[0]
    blk_idx = cute.arch.block_idx()[0]
    wrp_idx = cute.arch.warp_idx()
    lne_idx = cute.arch.lane_idx()

    # Allocate reduce buffer
    warp_size = cute.arch.WARP_SIZE
    num_warps = cute.ceil_div(threads, warp_size)
    reduce_buffer = cutlass.utils.SmemAllocator().allocate_tensor(
        input.element_type, cute.make_layout((num_warps,))
    )

    input_tile = input[(None, blk_idx)]
    output_tile = output[(None, blk_idx)]
    crd_tile = crd[(None, blk_idx)]

    thr_copy = tiled_copy.get_slice(thr_idx)
    input_thr = thr_copy.partition_S(input_tile)
    output_thr = thr_copy.partition_S(output_tile)
    crd_thr = thr_copy.partition_S(crd_tile)

    input_frag_thr = cute.make_fragment_like(input_thr)
    exp_frag_thr = cute.make_fragment_like(input_thr)
    max_value_frag_thr = cute.make_fragment(1, dtype=max_value.element_type)
    pred_frag_thr = cute.make_fragment_like(crd_thr, cute.Boolean)

    input_frag_thr.fill(0)
    exp_frag_thr.fill(0)

    for i in cutlass.range_constexpr(cute.cosize(pred_frag_thr)):
        pred_frag_thr[i] = cute.elem_less(crd_thr[i], (N,))

    # local sub_max_exp
    cute.copy(tiled_copy, input_thr, input_frag_thr, pred=pred_frag_thr)
    copy_atom_max = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), max_value.element_type)
    cute.copy(copy_atom_max, max_value, max_value_frag_thr)

    input_frag_thr.store(
        cute.math.exp(input_frag_thr.load() - max_value_frag_thr.load()[0].to(input.element_type))
    )

    for i in cutlass.range_constexpr(cute.cosize(pred_frag_thr)):
        if pred_frag_thr[i]:
            exp_frag_thr[i] = input_frag_thr[i]

    # local reduction
    thr_ssa = exp_frag_thr.load()
    sum_thr = thr_ssa.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)

    # write back now
    cute.copy(tiled_copy, input_frag_thr, output_thr, pred=pred_frag_thr)

    # warp reduction
    cute.arch.sync_warp()
    sum_thr = warp_reduce(sum_thr, operator.add, warp_size)
    if lne_idx == 0:
        reduce_buffer[wrp_idx] = sum_thr

    # block reduction
    # note here we only have 8 warps by default,
    # so just let the first thread do the final reduction
    # instead of launching another warp_reduce
    cute.arch.sync_threads()
    if thr_idx == 0:
        sum_blk = cutlass.Float32(0)
        for i in cutlass.range_constexpr(num_warps):
            sum_blk += reduce_buffer[i]

        # final reduction
        atomic_add_f32(sum_blk, sum_value.iterator)


@cute.kernel
def elemwise_kernel(
    input: cute.Tensor,
    sum_value: cute.Tensor,
    crd: cute.Tensor,
    tiled_copy: cute.TiledCopy,
    N: cute.Int32,
):
    tid_x = cute.arch.thread_idx()[0]
    bid_x = cute.arch.block_idx()[0]

    input_tile = input[(None, bid_x)]
    crd_tile = crd[(None, bid_x)]

    thr_copy = tiled_copy.get_slice(tid_x)
    input_thr = thr_copy.partition_S(input_tile)
    crd_thr = thr_copy.partition_S(crd_tile)

    input_frag_thr = cute.make_fragment_like(input_thr)
    pred_frag_thr = cute.make_fragment_like(crd_thr, cute.Boolean)
    sum_value_frag_thr = cute.make_fragment((1,), dtype=sum_value.element_type)

    for i in cutlass.range_constexpr(cute.cosize(pred_frag_thr)):
        pred_frag_thr[i] = cute.elem_less(crd_thr[i], (N,))

    copy_atom_sum = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), sum_value.element_type)
    cute.copy(tiled_copy, input_thr, input_frag_thr, pred=pred_frag_thr)
    cute.copy(copy_atom_sum, sum_value, sum_value_frag_thr)

    input_frag_thr.store(input_frag_thr.load() / sum_value_frag_thr.load()[0])

    cute.copy(tiled_copy, input_frag_thr, input_thr, pred=pred_frag_thr)


@cute.jit
def softmax(
    input: cute.Tensor,
    output: cute.Tensor,
    max_value: cute.Tensor,
    sum_value: cute.Tensor,
    N: cute.Int32,
):
    crd = cute.make_identity_tensor(input.shape)

    thr_layout = cute.make_layout((threads,))
    val_layout = cute.make_layout((vl,))

    input_tiled = cute.flat_divide(input, cta_tiler)
    output_tiled = cute.flat_divide(output, cta_tiler)
    crd_tiled = cute.flat_divide(crd, cta_tiler)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), input.element_type)
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    grid_size = [input_tiled.shape[1], 1, 1]
    block_size = [threads, 1, 1]

    max_reduction_kernel(
        input_tiled,
        max_value,
        crd_tiled,
        tiled_copy,
        N,
    ).launch(grid=grid_size, block=block_size)

    exp_sum_kernel(
        input_tiled,
        output_tiled,
        max_value,
        sum_value,
        crd_tiled,
        tiled_copy,
        N,
    ).launch(grid=grid_size, block=block_size)

    elemwise_kernel(
        output_tiled,
        sum_value,
        crd_tiled,
        tiled_copy,
        N,
    ).launch(grid=grid_size, block=block_size)


def solve(input: cute.Tensor, output: cute.Tensor, N: cute.Int32):

    int64_min = torch.iinfo(torch.int64).min
    max_value_torch = torch.tensor([int64_min], dtype=torch.int64, device='cuda')
    sum_value_torch = torch.tensor([0], dtype=torch.float32, device='cuda')
    max_value = cute.runtime.from_dlpack(max_value_torch)
    sum_value = cute.runtime.from_dlpack(sum_value_torch)

    softmax(input, output, max_value, sum_value, N)


def test():
    n = 10000
    input_torch = torch.empty(n, dtype=torch.float32, device='cuda').uniform_(-10, 10)
    # input_torch = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device='cuda')
    output_torch = torch.empty(n, dtype=torch.float32, device='cuda')

    input = cute.runtime.from_dlpack(input_torch)
    output = cute.runtime.from_dlpack(output_torch)

    solve(input, output, n)

    output_ref = torch.softmax(input_torch, dim=0)

    print(output_torch)
    print(output_ref)


if __name__ == "__main__":
    test()
