import torch
import cutlass
from cutlass import cute
from cutlass.cute.nvgpu import cpasync

import cuda.bindings.driver as cuda


@cute.kernel
def ldgsts_kernel(
    A: cute.Tensor,
    tiled_copy_4B: cute.TiledCopy,
    tiled_copy_16B: cute.TiledCopy,
):
    tid = cute.arch.thread_idx()[0]
    smem = cutlass.utils.SmemAllocator()

    sA = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((128, 128)),
        byte_alignment=16,
    )

    thr_cppy_4B = tiled_copy_4B.get_slice(tid)
    thr_cppy_16B = tiled_copy_16B.get_slice(tid)

    tAgA_4B = thr_cppy_4B.partition_S(A)
    tAgA_16B = thr_cppy_16B.partition_S(A)

    tAsA_4B = thr_cppy_4B.partition_D(sA)
    tAsA_16B = thr_cppy_16B.partition_D(sA)

    # The legacy barrier.
    barrier = cutlass.pipeline.NamedBarrier(barrier_id=1, num_threads=256)

    # The fasionable mbarrier, see cute.arch.mbar.py
    mbarrier = smem.allocate(cutlass.Int64, byte_alignment=8)
    if cute.arch.warp_idx() == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(mbarrier, cnt=256)
    cute.arch.mbarrier_init_fence()
    # Fence to switch to async proxy. Sometime you see pipeline_init_arrive(), which
    # calls mbarrier_init_fence and then arrive on clusters.

    cute.copy(tiled_copy_4B, tAgA_4B, tAsA_4B)
    cute.arch.cp_async_commit_group()
    """ Other irrelevant works """
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()
    # The asyn_group completion.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group
    barrier.arrive_and_wait()
    # In CuTe DSL, the barrier.arrive_and_wait() is equal to `cute.arch.sync_threads()`
    # or `pipeline.sync(barrier_id)` which will be lowered to nvvm.barrier.

    cute.copy(tiled_copy_16B, tAgA_16B, tAsA_16B)
    cute.arch.mbarrier_arrive(mbarrier, arrive_count=1)
    """ Other irrelevant works """
    cute.arch.mbarrier_wait(mbarrier, phase=0)
    # Phase? https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html#explicit-phase-tracking


@cute.jit
def ldgsts(A: cute.Tensor):
    op_normal = cpasync.CopyG2SOp(cpasync.LoadCacheMode.ALWAYS)
    op_bypass_l2 = cpasync.CopyG2SOp(cpasync.LoadCacheMode.GLOBAL)
    LDGSTS_4B = cute.make_copy_atom(op_normal, cutlass.Float32, num_bits_per_copy=32)
    LDGSTS_16B = cute.make_copy_atom(op_bypass_l2, cutlass.Float32, num_bits_per_copy=128)

    val_layout_4B = cute.make_layout((1, 1))
    val_layout_16B = cute.make_layout((4, 1))
    thr_layout = cute.make_layout((16, 16))

    tiled_copy_4B = cute.make_tiled_copy_tv(LDGSTS_4B, thr_layout, val_layout_4B)
    tiled_copy_16B = cute.make_tiled_copy_tv(LDGSTS_16B, thr_layout, val_layout_16B)

    ldgsts_kernel(A, tiled_copy_4B, tiled_copy_16B).launch(grid=[1, 1, 1], block=[256, 1, 1])


def run():
    A = (
        torch.arange(128 * 128, dtype=torch.float32, device="cuda")
        .reshape(128, 128)
        .transpose(0, 1)
    )
    A_cute = cute.runtime.from_dlpack(A, assumed_align=16)

    cute.compile(ldgsts, A_cute, options="--keep-ptx --keep-cubin")


if __name__ == "__main__":
    run()
