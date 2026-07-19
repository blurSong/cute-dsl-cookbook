import cuda.bindings.driver as cuda
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkTensorTileG2SOp,
    CopyBulkTensorTileS2GOp,
    make_tiled_tma_atom,
    tma_partition,
)
from cutlass.cute.nvgpu.tcgen05 import (
    SmemLayoutAtomKind,
    make_smem_layout_atom,
    tile_to_mma_shape,
)
from cutlass.utils import print_latex

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
def tma_load_store_test(
    A: cute.Tensor,
    B: cute.Tensor,
):
    tma_g2s = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), cutlass.Float32)
    tma_s2g = cute.make_copy_atom(cpasync.CopyBulkS2GOp(), cutlass.Float32)

    tma_load_store_kernel(A, B, tma_g2s, tma_s2g).launch(
        grid=[1, 1, 1], block=[32, 1, 1]
    )


@cute.jit
def tma_box_test(
    A: cute.Tensor,
    B: cute.Tensor,
    cta_tiler: cutlass.Constexpr[cute.Shape],
    smem_layout_atom_kind: cutlass.Constexpr[SmemLayoutAtomKind],
):
    smem_layout_atom = make_smem_layout_atom(
        smem_layout_atom_kind,
        A.element_type,
    )  # The `atom` is essentially a layout or composed layout.
    tiled_smem_layout = cute.tile_to_shape(
        smem_layout_atom, cta_tiler, order=(0, 1)
    )  # With order=(1, 2), basically we say the swizzle atom is first stacked along M, then K
    tma_atom, tma_tensor = make_tiled_tma_atom(
        CopyBulkTensorTileG2SOp(),
        A,
        tiled_smem_layout,  # Destination SMEM layout for 1 DMA_Stage, ((Mma_M, Mma_K), NumMma_M, NumMma_K)
        cta_tiler,  # cta_tiler. I.e.m, the TMA box shape, it's cosize must match the cosize of the destination smem layout
    )

    # print(f"A Tensor: {A}")
    # print(f"Smem Layout Atom: {smem_layout_atom}")
    # print(f"Tiled SMEM Layout: {tiled_smem_layout}")
    # print(f"TMA Atom: {tma_atom}")
    # print(f"TMA Atom SMEM Layout: {tma_atom.smem_layout}")
    # print(f"TMA Tensor: {tma_tensor}")

    print_latex(tiled_smem_layout)


def run_tests():
    cta_tiler = (64, 64)

    A = torch.arange(128 * 256, dtype=torch.float16, device="cuda").reshape(128, 256)
    B = torch.zeros_like(A)
    A_cute = cute.runtime.from_dlpack(A, assumed_align=16)
    B_cute = cute.runtime.from_dlpack(B, assumed_align=16)

    smem_layout_atom_kind = SmemLayoutAtomKind.K_SW128

    tma_box_test(A_cute, B_cute, cta_tiler, smem_layout_atom_kind)
    # tma_load_store_test(A_cute, B_cute, cta_tiler)

    # tma_load_store_compiled = cute.compile(tma_load_store_test, A_cute, B_cute, options="--keep-ptx")

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
        print("\nWARNING: TMA requires SM90+ (Hopper architecture)\n")

    run_tests()


if __name__ == "__main__":
    main()
