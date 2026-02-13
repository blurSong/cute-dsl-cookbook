import cutlass
import cutlass.cute as cute

warp_tiler = (4, 2, 1)
thr_tiler_warp = (4, 8, 1)
mma_permute = 4
warptile_permute_m = 2
warptile_permute_n = 2

gemm_dtype = cutlass.Float32


@cute.jit
def test():
    warp_layout = cute.make_layout((4, 2, 1), stride=(2, 1, 0))
    thr_layout_warp = cute.make_layout((4, 8, 1), stride=(8, 1, 0))
    mma_atom_layout_by_prod = cute.raked_product(thr_layout_warp, warp_layout)
    mma_atom_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))
    # TODO. Any better way?
    permutation_m = cute.make_ordered_layout(
        (warp_tiler[0], warptile_permute_m, thr_tiler_warp[0], mma_permute),
        order=(3, 2, 1, 0),
    )
    permutation_n = cute.make_ordered_layout(
        (warp_tiler[1], warptile_permute_n, thr_tiler_warp[1], mma_permute),
        order=(3, 2, 1, 0),
    )
    tiled_mma = cute.make_tiled_mma(
        cute.nvgpu.MmaUniversalOp(gemm_dtype),
        atom_layout_mnk=mma_atom_layout_by_prod,
        # permutation_mnk=(permutation_m, permutation_n, None),
    )
    print(warp_layout)
    print(thr_layout_warp)
    print(mma_atom_layout_by_prod)
    print(mma_atom_layout)
    print(permutation_m)
    print(permutation_n)
    print(tiled_mma)


if __name__ == "__main__":

    test()
