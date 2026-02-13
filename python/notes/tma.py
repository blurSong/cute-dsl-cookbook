import cuda.bindings.driver as cuda
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkTensorTileG2SOp,
    CopyBulkTensorTileS2GOp,
    make_tiled_tma_atom,
    tma_partition,
)

import cutlass
from cutlass import cute


@cute.kernel
def tma_load_store_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    tma_atom_g2s: cute.CopyAtom,
    tma_atom_s2g: cute.CopyAtom,
):
    tid = cute.arch.thread_idx()[0]
    smem = cutlass.utils.SmemAllocator()

    sA = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((128, 128)),
        byte_alignment=16,
    )

    mbarrier = smem.allocate(cutlass.Int64, byte_alignment=8)
    with cute.arch.elect_one():
        cute.arch.mbarrier_init(mbarrier, cnt=32)
    cute.arch.mbarrier_init_fence()

    with cute.arch.elect_one():
        cute.copy(tma_atom_g2s, A, sA, mbar_ptr=mbarrier)
        cute.arch.mbarrier_expect_tx(
            mbarrier,
            bytes=cute.size_in_bytes(cutlass.Float32, sA.layout),
        )
    cute.arch.mbarrier_arrive(mbarrier)
    cute.arch.mbarrier_wait(mbarrier, phase=0)

    cute.arch.fence_proxy(
        kind=cute.arch.ProxyKind.async_shared,
        space=cute.arch.SharedSpace.shared_cta,
    )
    cute.arch.sync_threads()
    with cute.arch.elect_one():
        cute.copy(tma_atom_s2g, sA, B)
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0)


@cute.jit
def tma_bulk(A: cute.Tensor, B: cute.Tensor):
    tma_g2s = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), cutlass.Float32)
    tma_s2g = cute.make_copy_atom(cpasync.CopyBulkS2GOp(), cutlass.Float32)

    tma_load_store_kernel(A, B, tma_g2s, tma_s2g).launch(grid=[1, 1, 1], block=[32, 1, 1])


def run_tests():

    A = (
        torch.arange(128 * 128, dtype=torch.float32, device="cuda")
        .reshape(128, 128)
        .transpose(0, 1)
    )
    B = torch.zeros((128, 128), dtype=torch.float32, device="cuda").transpose(0, 1)
    A_cute = cute.runtime.from_dlpack(A, assumed_align=16)
    B_cute = cute.runtime.from_dlpack(B, assumed_align=16)

    tma_bulk_compiled = cute.compile(tma_bulk, A_cute, B_cute, options="--keep-ptx")
    tma_bulk_compiled(A_cute, B_cute)

    print("Input A (first 8x8):")
    print(A[:8, :8].cpu())
    print("\nOutput B (first 8x8):")
    print(B[:8, :8].cpu())

    # 验证结果
    if torch.allclose(A.cpu(), B.cpu()):
        print("\n✓ Bulk copy test PASSED!")
    else:
        print("\n✗ Bulk copy test FAILED!")


def main():
    cuda_device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(cuda_device)
    print(f"CUDA Device: {torch.cuda.get_device_name(cuda_device)}")
    print(f"Compute Capability: {capability[0]}.{capability[1]}")

    if capability[0] < 9:
        print("\nWARNING: TMA requires SM90+ (Hopper architecture)")
    else:
        run_tests()


if __name__ == "__main__":
    main()
