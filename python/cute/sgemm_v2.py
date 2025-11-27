"""
A dense FP32 SIMT GEMM using CUTE DSL.

The SGEMM implementation handles both row-major and col-major layouts.
To bridge the gap of GEMM order between BLAS and CUTE, we can use the following definitions:
------------------------------------------
Blas      T                   N
------------------------------------------
A         (M, K):(K, 1)      (M, K):(1, M)
B         (N, K):(1, N)      (N, K):(K, 1)
------------------------------------------
See also:
    https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0x_gemm_tutorial.html#aside-m-major-n-major-k-major


This GEMM kernel supports the following features:
    - Utilizes FPU for matrix multiply-accumulate (MMA) operations
    - Use multistage pipeline to overlap computation and memory access
      * Shared memory pipeline: hides gmem-to-smem latency.
      * Register pipeline: overlaps shared memory-to-register transfers with
        computations and eliminates false data dependencies for
        better parallelism.
    - Use vectorized copies
    - Add padding to reduce bank conflicts in global -> shared memory copies
    - Use predication to avoid unnecessary copies or copies of stale data
Our new optimizations:
    - Improved Row-major data loading, see `make_tiled_copy_AB`
    - Warp tiling
    - Coalesced Epilogue RMEM->SMEM->GMEM

Basicaly, this CuTe kernel is a faithful translation of this awsome blog:
    -  https://salykova.github.io/sgemm-gpu

Reference:
    - CUTLASS blog  https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda
    - SIBOEHM's blog  https://siboehm.com/articles/22/CUDA-MMM
    - Lei Mao's blog  https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/


To run this example:

.. code-block:: bash

    python sgemm_v2.py -M 8192 -N 8192 -K 8192

To collect performance with NCU profiler:

.. code-block:: bash

    ncu -o ../../logs/sgemm_v2 -f --set full \
        python sgemm_v2.py \
        --skip-verify --warmup-iterations 1  --iterations 1


Constraints:
    1. Supported input, output, and accumulator data types: fp32
    2. Default tile shape is set to be 128x128x8
    3. The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned
"""

import time
import math
import torch
import argparse
import operator
from typing import Type, List, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import LayoutEnum

from utils import check_cuda, benchmark_torch

VERBOSE = True
LOG = "[CuTe Info]"

# GLOBAL ================================================
# DATA TYPES --------------------------------------------
gemm_dtype = cutlass.Float32
# SGEMM CONFIGURATIONS ----------------------------------
num_stages = 3
threads = 256
copy_bits = 128
mma_permute = 4
cta_tiler = (128, 128, 8)
warp_tiler = (32, 64, 8)
wmma_tiler = (4, 8)
# mma_tiler = (16, 16)
# warp_tiler = (4, 2)
# thr_tiler_warp = (4, 8)
# PARAMETERS DERIVED ------------------------------------
tile_m, tile_n, tile_k = cta_tiler
warp_m, warp_n, warp_k = warp_tiler
wmma_m, wmma_n = wmma_tiler
num_warps_m = tile_m // warp_m
num_warps_n = tile_n // warp_n
num_warps = num_warps_m * num_warps_n
vl = copy_bits // gemm_dtype.width
bytes_alignment = copy_bits // 8
# ASSERTIONS --------------------------------------------
assert mma_permute == vl
assert wmma_m * wmma_n == 32
assert wmma_m * wmma_n * num_warps == threads
assert tile_m % (warp_m) == 0
assert tile_n % (warp_n) == 0
assert warp_m % (mma_permute * wmma_m) == 0
assert warp_n % (mma_permute * wmma_n) == 0
assert tile_m % bytes_alignment == 0
assert tile_n % bytes_alignment == 0
assert bytes_alignment % (copy_bits // 8) == 0
# =======================================================


def make_tiled_copy_AB(
    major_mode: LayoutEnum,
    cta_tiler: Tuple[int, int],
):
    """Make G2S tiled copy A and B

    - The thread layout follows the major_mode of the GMEM.
    - If A/B is ROW_MAJOR `A@T (M,K):(K,1), B@N (N,K):(K,1)`, don't need vl since
      multiple threads's LDGs can coalesce. We apply the magic number mma_permute=vl for
      the val_layout outer demension. Each tile_k threads will cooprate to read
      mma_permute * tile_k values.
    - If A/B is COL_MAJOR `A@N (M,K):(1,M), B@T (N,K):(1,N)`, try to use 128bits vector load.

    Params:
        - cta_tiler: A `(tile_m, tile_k)`, B `(tile_n, tile_k)`
    """
    if major_mode == LayoutEnum.ROW_MAJOR:
        order = (1, 0)
        thr_tiler_0 = threads // cta_tiler[1]
        thr_tiler_1 = cta_tiler[1]
        val_tiler = (mma_permute, 1)
        num_bits_per_copy = gemm_dtype.width
    else:  # LayoutEnum.COL_MAJOR
        order = (0, 1)
        thr_tiler_0 = cta_tiler[0] // vl
        thr_tiler_1 = threads * vl // cta_tiler[0]
        val_tiler = (vl, 1)
        num_bits_per_copy = copy_bits

    thr_layout = cute.make_ordered_layout((thr_tiler_0, thr_tiler_1), order=order)
    val_layout = cute.make_ordered_layout(val_tiler, order=order)
    copy_atom_AB = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        # cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL,
        # LoadCacheMode.GLOBAL will bypass L1
        gemm_dtype,
        num_bits_per_copy=num_bits_per_copy,
    )
    tiled_copy = cute.make_tiled_copy_tv(copy_atom_AB, thr_layout, val_layout)
    return tiled_copy


def make_tiled_copy_C(
    major_mode: LayoutEnum,
):
    """Make G2S tiled copy A and B
    """
    if major_mode == LayoutEnum.ROW_MAJOR:
        order = (1, 0)
        thr_tiler_1 = tile_n // vl
        thr_tiler_0 = threads * vl // tile_n
        val_tiler = (1, vl)
    else:  # LayoutEnum.COL_MAJOR
        order = (0, 1)
        thr_tiler_0 = tile_m // vl
        thr_tiler_1 = threads * vl // tile_m
        val_tiler = (vl, 1)

    thr_layout = cute.make_ordered_layout((thr_tiler_0, thr_tiler_1), order=order)
    val_layout = cute.make_ordered_layout(val_tiler, order=order)
    copy_atom_C = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        gemm_dtype,
        num_bits_per_copy=copy_bits,
    )
    tiled_copy = cute.make_tiled_copy_tv(copy_atom_C, thr_layout, val_layout)
    return tiled_copy


@cute.kernel
def sgemm_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    smem_layout_A: cute.Layout,
    smem_layout_B: cute.Layout,
    smem_layout_C: cute.Layout,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    tiled_copy_C: cute.TiledCopy,
    tiled_warp_mma: cute.TiledMma,
    epilogue_op: cutlass.Constexpr = lambda x: x,
):
    thr_idx = cute.arch.thread_idx()[0]
    blk_idx, blk_idy = cute.arch.block_idx()[:2]
    wrp_idx = cute.arch.warp_idx()
    lne_idx = cute.arch.lane_idx()

    cta_coord = (blk_idx, blk_idy, None)
    # Here None means 'all' for cute slicing

    # Use local tile to tile the gA/B/C.
    # - each cta handles one (tile_m, K)@A and one (tile_n, K)@B
    # - Another choice is to use zipped_devide and apply the coord for indexing.
    #   https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0x_gemm_tutorial.html#cta-partitioning

    # * gA (tile_m, tile_k, num_k_tiles)
    # * gB (tile_n, tile_k, num_k_tiles)
    # * gC (tile_m, tile_n)
    gA = cute.local_tile(A, cta_tiler, cta_coord, proj=(1, None, 1))
    gB = cute.local_tile(B, cta_tiler, cta_coord, proj=(None, 1, 1))
    gC = cute.local_tile(C, cta_tiler, cta_coord, proj=(1, 1, None))
    # here, None means 'not select' as X for Cute C++ API.

    # - Optimization: Move the pointer of gA/gB in the -k direction, making the
    #   first tile (instead of the last one) irregular in shape when k is irregular.
    # - We first handle the irregular tile in the prologue
    #   to avoid checking for this condition within the mainloop.
    # - Note that residual_k is 0 or negative
    residue_k = A.shape[1] - tile_k * gA.shape[2]
    gA = cute.domain_offset((0, residue_k, 0), gA)
    gB = cute.domain_offset((0, residue_k, 0), gB)

    # Allocate SMEM
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(gemm_dtype, smem_layout_A, 16)
    sB = smem.allocate_tensor(gemm_dtype, smem_layout_B, 16)
    sC = smem.allocate_tensor(gemm_dtype, smem_layout_C, 16)

    # Partition tiled copies
    # - Note that the tCgC is for RMEM->GMEM epilogue.
    #   The tCgC shape should be the same as MMA's tCrC.
    # * tAgA ((atom_v, rest_v), copy_m, copy_k, num_k_tiles)
    # * tAsA ((atom_v, rest_v), copy_m, copy_k, num_stages)
    # * tBgB ((atom_v, rest_v), copy_n, copy_k, num_k_tiles)
    # * tBsB ((atom_v, rest_v), copy_n, copy_k, num_stages)
    # * tCgC ((atom_v, rest_v), copy_m, copy_n)
    thr_copy_A = tiled_copy_A.get_slice(thr_idx)
    thr_copy_B = tiled_copy_B.get_slice(thr_idx)
    thr_copy_C = tiled_copy_C.get_slice(thr_idx)
    tAgA = thr_copy_A.partition_S(gA)
    tAsA = thr_copy_A.partition_D(sA)
    tBgB = thr_copy_B.partition_S(gB)
    tBsB = thr_copy_B.partition_D(sB)
    tCsC_copy = thr_copy_C.partition_S(sC)
    tCgC_copy = thr_copy_C.partition_D(gC)

    # Make predicator
    # - Mark indices that need to copy when the problem shape
    #   isn't a multiple of the tile shape.
    # - If tApredA/B[i] is 0, then do not do the copy
    #   atom associated with index i.
    # - Refer to https://shorturl.at/8D9PK

    # Construct identity layout for sA and sB, used for predication
    crdA = cute.make_identity_tensor(A.shape)
    crdB = cute.make_identity_tensor(B.shape)
    crdC = cute.make_identity_tensor(C.shape)
    crdA = cute.local_tile(crdA, cta_tiler, cta_coord, proj=(1, None, 1))
    crdB = cute.local_tile(crdB, cta_tiler, cta_coord, proj=(None, 1, 1))
    crdC = cute.local_tile(crdC, cta_tiler, cta_coord, proj=(1, 1, None))
    crdA = cute.domain_offset((0, residue_k, 0), crdA)
    crdB = cute.domain_offset((0, residue_k, 0), crdB)
    tAcrdA = thr_copy_A.partition_S(crdA)
    tBcrdB = thr_copy_B.partition_S(crdB)
    tCcrdC = thr_copy_C.partition_D(crdC)

    # Making predicators of A, B and C
    # - To avoid creating multiple predicators for normal/residual blocks
    #   We reuse the tAprdA and tBprdB for both irregular and regular tiles.
    # - Note that tXpreX's 0th dim is rest_v in (atom_v, rest_v). This is due to
    #   that the predicators is checked at the granularity of the Copy Atom.
    # - Here the mode3's stride of tAprdA/tBprdB is 0. We only pred the m and n bounds.
    #   k is broadcasted. And tCprdC's mode3 has stride=1.
    # * tAprdA (rest_v, copy_m, copy_k)
    # * tBprdB (rest_v, copy_n, copy_k)
    # * tCprdC (rest_v, copy_m, copy_n)
    tAprdA = cute.make_rmem_tensor(
        cute.make_ordered_layout(
            (tAgA.shape[0][1], tAgA.shape[1], tAgA.shape[2]),
            order=(2, 1, 0),
        ),
        dtype=cutlass.Boolean,
    )
    tBprdB = cute.make_rmem_tensor(
        cute.make_ordered_layout(
            (tBgB.shape[0][1], tBgB.shape[1], tBgB.shape[2]),
            order=(2, 1, 0),
        ),
        dtype=cutlass.Boolean,
    )
    tCprdC = cute.make_rmem_tensor(
        cute.make_ordered_layout(
            (tCgC_copy.shape[0][1], tCgC_copy.shape[1], tCgC_copy.shape[2]),
            order=(2, 1, 0),
        ),
        dtype=cutlass.Boolean,
    )

    # First tAprdA and tBprdB will be used to prefethc the
    # first irregular tile.
    for i in range(tAprdA.shape[0]):
        for j in range(tAprdA.shape[1]):
            for k in range(tAprdA.shape[2]):
                coord_tmp = ((0, i), j, k, 0)
                # tAcrdA[0] should be < A.shape[0] for M
                # tAcrdA[1] should be > -1 for K
                tAprdA[i, j, k] = cute.elem_less(
                    (tAcrdA[coord_tmp][0], -1), (A.shape[0], tAcrdA[coord_tmp][1])
                )
    for i in range(tBprdB.shape[0]):
        for j in range(tBprdB.shape[1]):
            for k in range(tBprdB.shape[2]):
                coord_tmp = ((0, i), j, k, 0)
                tBprdB[i, j, k] = cute.elem_less(
                    (tBcrdB[coord_tmp][1], -1), (B.shape[0], tBcrdB[coord_tmp][1])
                )
    for i in range(tCprdC.shape[0]):
        for j in range(tCprdC.shape[1]):
            for k in range(tCprdC.shape[2]):
                coord_tmp = ((0, i), j, k)
                tCprdC[i, j, k] = cute.elem_less(tCcrdC[coord_tmp], C.shape)

    # =============================== Prefetch Prologue ===============================
    # ---------------------------------------------------------------------------------
    # Prefetch the first tile GMEM->SMEM
    # - Since the domain_offset is applied, now the first tile of A, B may be irregular.
    #   We reuse the predicators of A, B to handle the M/N boundary conditions.
    # - Before that, we handle the k boundary one-by-one using if-else statement.
    #   If one k_index is OOB, its corresponding coord tensore will be NEGATIVE.
    tAsA.fill(0)
    tBsB.fill(0)
    cute.arch.sync_threads()

    coord_g2s = (None, None, None, 0)
    cute.copy(tiled_copy_A, tAgA[coord_g2s], tAsA[coord_g2s], pred=tAprdA)
    cute.copy(tiled_copy_B, tBgB[coord_g2s], tBsB[coord_g2s], pred=tBprdB)
    cute.arch.cp_async_commit_group()

    # Then we should reshpe the predicators and update the pred values
    # with only M/N boundary taken into considerd.
    tAprdA = cute.make_tensor(
        tAprdA.iterator,
        layout=cute.make_layout(
            tAprdA.shape,
            stride=(tAprdA.shape[1], 1, 0),
        ),
    )
    tBprdB = cute.make_tensor(
        tBprdB.iterator,
        layout=cute.make_layout(
            tBprdB.shape,
            stride=(tBprdB.shape[1], 1, 0),
        ),
    )
    for i in range(tAprdA.shape[0]):
        for j in range(tAprdA.shape[1]):
            coord_tmp = ((0, i), j, 0, 0)
            tAprdA[i, j, 0] = cute.elem_less(tAcrdA[coord_tmp][0], A.shape[0])
    for i in range(tBprdB.shape[0]):
        for j in range(tBprdB.shape[1]):
            coord_tmp = ((0, i), j, 0, 0)
            tBprdB[i, j, 0] = cute.elem_less(tBcrdB[coord_tmp][1], B.shape[0])

    # Then start async loads and fill the 1 to num_stages-2 pipes
    num_k_tiles = cute.size(tAgA, mode=[3])
    k_tile_index_gmem = cutlass.Int32(1)
    k_tile_index_smem = cutlass.Int32(1)
    for _ in range(1, num_stages - 1):
        if k_tile_index_smem < num_k_tiles:
            coord_gmem = (None, None, None, k_tile_index_gmem)
            coord_smem = (None, None, None, k_tile_index_smem)
            cute.copy(tiled_copy_A, tAgA[coord_gmem], tAsA[coord_smem], pred=tAprdA)
            cute.copy(tiled_copy_B, tBgB[coord_gmem], tBsB[coord_smem], pred=tBprdB)
            k_tile_index_gmem += 1
            k_tile_index_smem += 1
        cute.arch.cp_async_commit_group()
    # if reach num_k_tiles, clear the predictors to cancle all copies
    if k_tile_index_smem == num_k_tiles:
        tAprdA.fill(0)
        tBprdB.fill(0)

    # Key modification. Tiled MMA partitions
    # - To perform warp tiling in SIMD..., we tile the block of sA, sB and sC
    #   with warp id. Then each warp partition its tile with MMA atom.
    # - Prefetch SMEM2RMEM
    # * wAsA ((warp_m, warp_k), res_m, res_k, num_stages) -> (warp_m, warp_k, res_k, num_stages)
    # * wBsB ((warp_n, warp_k), res_n, res_k, num_stages) -> (warp_n, warp_k, res_k, num_stages)
    # * wCsC_mma ((warp_m, warp_n), res_m, res_n) -> (warp_m, warp_n)
    # * tCsA (mma_atom, (mma_permute, num_m_frags), num_k_frags, ?, num_stages)
    # * tCsB (mma_atom, (mma_permute, num_n_frags), num_k_frags, ?, num_stages)
    # * tCgC (mma_atom, (mma_permute, num_m_frags), (mma_permute, num_n_frags))
    wrp_idm, wrp_idn = wrp_idx // num_warps_n, wrp_idx % num_warps_n
    warp_coord_A = ((None, None), wrp_idm, None, None)
    warp_coord_B = ((None, None), wrp_idn, None, None)
    warp_coord_C = ((None, None), wrp_idm, wrp_idn)
    wCsA = cute.tiled_divide(sA, (warp_m, warp_k))[warp_coord_A]
    wCsB = cute.tiled_divide(sB, (warp_n, warp_k))[warp_coord_B]
    wCsC_mma = cute.tiled_divide(sC, (warp_m, warp_n))[warp_coord_C]
    wCgC_mma = cute.tiled_divide(gC, (warp_m, warp_n))[warp_coord_C]

    thr_mma = tiled_warp_mma.get_slice(lne_idx)
    tCsA = thr_mma.partition_A(wCsA)
    tCsB = thr_mma.partition_B(wCsB)
    tCsC_mma = thr_mma.partition_C(wCsC_mma)
    tCgC_mma = thr_mma.partition_C(wCgC_mma)

    tCrA = tiled_warp_mma.make_fragment_A(tCsA[None, None, None, None, 0])
    tCrB = tiled_warp_mma.make_fragment_B(tCsB[None, None, None, None, 0])
    tCrC = tiled_warp_mma.make_fragment_C(tCgC_mma)
    tCrC.fill(0.0)

    # Prefetch SMEM->RMEM for the first MMA tile
    smem_pipe_read = cutlass.Int32(0)
    smem_pipe_write = cutlass.Int32(num_stages - 1)
    gmem_pipe_read = k_tile_index_gmem

    tCsA_p = tCsA[None, None, None, None, smem_pipe_read]
    tCsB_p = tCsB[None, None, None, None, smem_pipe_read]

    num_k_frags = cute.size(tCrA, mode=[2])
    cta_sync_barrier = cutlass.pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=threads,
    )
    if num_k_frags > 1:
        # Wait until the first prefetched tile is loaded in
        cute.arch.cp_async_wait_group(num_stages - 2)
        cta_sync_barrier.arrive_and_wait()
        # Prefetch the first rmem from the first k-tile
        coord_frag = (None, None, 0, None)
        cute.autovec_copy(tCsA_p[coord_frag], tCrA[coord_frag])
        cute.autovec_copy(tCsB_p[coord_frag], tCrB[coord_frag])

    if cutlass.const_expr(VERBOSE):
        print(f"{LOG} gA {gA}")
        print(f"{LOG} gB {gB}")
        print(f"{LOG} gC {gC}")
        print(f"{LOG} sA {sA}")
        print(f"{LOG} sB {sB}")
        print(f"{LOG} tAgA {tAgA}")
        print(f"{LOG} tAsA {tAsA}")
        print(f"{LOG} tBgB {tBgB}")
        print(f"{LOG} tBsB {tBsB}")
        print(f"{LOG} tCsC_copy {tCsC_copy}")
        print(f"{LOG} tCgC_copy {tCgC_copy}")
        print(f"{LOG} wCsA {wCsA}")
        print(f"{LOG} wCsB {wCsB}")
        print(f"{LOG} wCsC_mma {wCsC_mma}")
        print(f"{LOG} wCgC_mma {wCgC_mma}")
        print(f"{LOG} tCsA {tCsA}")
        print(f"{LOG} tCsB {tCsB}")
        print(f"{LOG} tCsC_mma {tCsC_mma}")
        print(f"{LOG} tCgC_mma {tCgC_mma}")
        print(f"{LOG} tCrA {tCrA}")
        print(f"{LOG} tCrB {tCrB}")
        print(f"{LOG} tCrC {tCrC}")

    # if blk_idx == 0 and blk_idy == 0 and thr_idx == 0:
    #     cute.print_tensor(sA[None, None, 0])
    #     cute.print_tensor(sB[None, None, 0])

    # =============================== Mainloop ===============================
    # 1. Shared memory pipeline (gmem -> smem):
    #    The default smem pipeline depth is 3, meaning that for shared
    # memory buffers, we allocate three times the size described by the
    # CTA tiler. We prefetch 2 of these buffers before entering the main
    # loop. Considering only the transfer from global memory to shared
    # memory, the general structure of the mainloop is:
    #   (1) copy k-tile from gmem to smem;
    #   (2) perform gemm computation on k-tile;
    #   (3) wait for the next copy to finish.
    #    The `cute.arch.cp_async_wait_group(num_smem_stages - 2)` command
    # waits for the number of unfinished 'copy' to be <= 1. The advantage
    # of this approach is that it allows for simultaneous production
    # (i.e., step (1)) and consumption (i.e., step (2)) of smem.
    #    A common misconception is to prefetch N buffers and rewrite
    # the pipeline logic to wait on N-1 pending copies. The disadvantage
    # of this approach is that it requires fully consuming a buffer in
    # order to open an empty buffer for the next copy.
    # 2. Register pipeline (smem -> register):
    #    Similarly, the register pipeline produces i+1, consumes i, and
    # produces i+2... Notably, i and i+1 do not use the same register,
    # eliminating dependencies on the same register for better parallelism.
    # 3. Combining the smem and register pipelines results in the mainloop.
    # ========================================================================

    for k_tile_index in range(num_k_tiles):
        # Fetching next k-tile GMEM->SMEM
        # - Note that current SMEM state is
        #   [F][O][O][E][E(if)] F(FETCHED), O(ON-FLY), E(EMPTY)
        # - The initialized smem_pipe_r/w index are 0 and num_stages-1.
        #   Firstly the [E] smem tile will be fetched.
        if gmem_pipe_read < num_k_tiles:
            coord_g = (None, None, None, gmem_pipe_read)
            coord_s = (None, None, None, smem_pipe_write)
            cute.copy(tiled_copy_A, tAgA[coord_g], tAsA[coord_s], pred=tAprdA)
            cute.copy(tiled_copy_B, tBgB[coord_g], tBsB[coord_s], pred=tBprdB)
            cute.arch.cp_async_commit_group()
            # Update meta pointers of gmem/smem pipes
            gmem_pipe_read += 1
            smem_pipe_write = (smem_pipe_write + 1) % num_stages
        # Always move smem_pipe_read ptr to fethch SMEM->RMEM
        smem_pipe_read = (smem_pipe_read + 1) % num_stages
        for k_frag_index in cutlass.range(num_k_frags, unroll_full=True):
            # - If the inner loop reaches the last fragment, we need to
            #   update tCsA_p/tCsB_p to prefetch the first fragment in the next smem.
            # - Note that the smem_pipe_read is already updated in the outer loop.
            if k_frag_index == num_k_frags - 1:
                coord_smem_next = (None, None, None, None, smem_pipe_read)
                tCsA_p = tCsA[coord_smem_next]
                tCsB_p = tCsB[coord_smem_next]
                cute.arch.cp_async_wait_group(num_stages - 2)
                cta_sync_barrier.arrive_and_wait()
            # - Then fetch next frag SMEM->RMEM
            coord_frag_next = (None, None, (k_frag_index + 1) % num_k_frags, None)
            cute.autovec_copy(tCsA_p[coord_frag_next], tCrA[coord_frag_next])
            cute.autovec_copy(tCsB_p[coord_frag_next], tCrB[coord_frag_next])
            # - Finally perform mma on current frag
            coord_mma = (None, None, k_frag_index, None)
            cute.gemm(tiled_warp_mma, tCrC, tCrA[coord_mma], tCrB[coord_mma], tCrC)

    cute.arch.cp_async_wait_group(0)
    cta_sync_barrier.arrive_and_wait()

    # =============================== Epilogue with fusion ===============================
    # ------------------------------------------------------------------------------------
    tCrC.store(epilogue_op(tCrC.load()))
    cute.autovec_copy(tCrC, tCsC_mma)
    cute.arch.sync_threads()
    tCrC_copy = cute.make_rmem_tensor_like(tCsC_copy)
    cute.autovec_copy(tCsC_copy, tCrC_copy)
    cute.copy(tiled_copy_C, tCrC_copy, tCgC_copy, pred=tCprdC)


@cute.jit
def sgemm(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    epilogue_op: cutlass.Constexpr = lambda x: x,
):
    # num_stages for overlapping loading and computation
    # Each cta handles a tile of size (tile_m, K)@A and (tile_n, K)@B produce (tile_m, tile_n)@C
    # Every time loads (tile_m, tile_k)@A and (tile_n, tile_k)@B, loop K with tile_k
    # Default cta threads 256，So the mma shape is (16, 16), each thread loads 4 elems A and 4 B.

    # A is ROW_MAJOR@T and COL_MAJOR@N
    # B is COL_MAJOR@T and ROW_MAJOR@N
    major_mode_A = LayoutEnum.from_tensor(A)
    major_mode_B = LayoutEnum.from_tensor(B)
    major_mode_C = LayoutEnum.from_tensor(C)

    # Create layouts for shared memory for A and B
    # use padding to avoid bank conflict when ldg then sts.
    padding_A = mma_permute if major_mode_A == LayoutEnum.ROW_MAJOR else 0
    padding_B = mma_permute if major_mode_B == LayoutEnum.ROW_MAJOR else 0
    smem_layout_A = cute.make_layout(
        (tile_m, tile_k, num_stages), stride=(1, tile_m + padding_A, tile_k * (tile_m + padding_A))
    )
    smem_layout_B = cute.make_layout(
        (tile_n, tile_k, num_stages), stride=(1, tile_n + padding_B, tile_k * (tile_n + padding_B))
    )
    smem_layout_C = cute.make_layout((tile_m, tile_n), stride=(tile_n, 1))
    smem_size = sum(
        [cute.size_in_bytes(gemm_dtype, lo) for lo in [smem_layout_A, smem_layout_B, smem_layout_C]]
    )

    # Create copy layout for A, B and C
    tiled_copy_A = make_tiled_copy_AB(major_mode_A, (tile_m, tile_k))
    tiled_copy_B = make_tiled_copy_AB(major_mode_B, (tile_n, tile_k))
    tiled_copy_C = make_tiled_copy_C(major_mode_C)

    # Create MMA layout for GEMM
    # - The MmaUniversalOp has a trivial 1x1x1 MMA trait.
    # - The permutation = 4 means each thread loads 4 contiguous A/B
    #   values from smem to regs thus enable ld.shared.v4

    # One important observation:
    # - Seems that CuTe atom_layout_mnk only supports rank-3 layout w/o nesting.
    #   However, if we want to apply the warp tiling optimzation, we may need
    #   an atom_layout_mnk like `((4,4),(2,8),1):((64,8),(32,1),0)` which will
    #   cause an abortion. We may conclude temporarily that
    #   THE WARP TILING IS UNACHIEVABLE.

    # - About how permutation_mnk works:
    #   https://github.com/NVIDIA/cutlass/discussions/1345
    # - About warp tiling for sgemm:
    #   https://salykova.github.io/sgemm-gpu

    warp_mma_atom_layout = cute.make_layout((wmma_m, wmma_n, 1), stride=(wmma_n, 1, 0))
    permutation_m = cute.make_ordered_layout((wmma_m, mma_permute), order=(1, 0))
    permutation_n = cute.make_ordered_layout((wmma_n, mma_permute), order=(1, 0))
    tiled_warp_mma = cute.make_tiled_mma(
        cute.nvgpu.MmaUniversalOp(gemm_dtype),
        atom_layout_mnk=warp_mma_atom_layout,
        permutation_mnk=(permutation_m, permutation_n, None),
    )

    grid_dim = [*cute.ceil_div(C.shape, (tile_m, tile_n)), 1]
    block_dim = [threads, 1, 1]

    if cutlass.const_expr(VERBOSE):
        print(f"{LOG} Tensor A {A}")
        print(f"{LOG} Tensor B {B}")
        print(f"{LOG} Tensor C {C}")
        print(f"{LOG} Major mode A {major_mode_A}")
        print(f"{LOG} Major mode B {major_mode_B}")
        print(f"{LOG} Smem layout A {smem_layout_A}")
        print(f"{LOG} Smem layout B {smem_layout_B}")
        print(f"{LOG} Copy layout A {tiled_copy_A}")
        print(f"{LOG} Copy layout B {tiled_copy_B}")
        print(f"{LOG} Copy layout C {tiled_copy_C}")
        print(f"{LOG} Mma layout (Warp level) {tiled_warp_mma}")
        print(f"{LOG} Gemm tile size {cta_tiler}")
        print(f"{LOG} Sgemm grid {grid_dim}")
        print(f"{LOG} Sgemm block {block_dim}")

    sgemm_kernel(
        A,
        B,
        C,
        smem_layout_A,
        smem_layout_B,
        smem_layout_C,
        tiled_copy_A,
        tiled_copy_B,
        tiled_copy_C,
        tiled_warp_mma,
        epilogue_op,
    ).launch(grid=grid_dim, block=block_dim, smem=smem_size)


def run_sgemm(
    M: int,
    N: int,
    K: int,
    blas: str = "tn",
    skip_verify: bool = False,
    dynamic_layout: bool = False,
    warmup_iterations: int = 10,
    iterations: int = 100,
):
    print("Running sgemm with M={}, N={}, K={}, BLAS {}.".format(M, N, K, blas.upper()))

    a_transpose, b_transpose = [t == "t" for t in blas]

    def tensor_generator(return_type: str = "all"):
        assert return_type in ["all", "cute_only", "torch_only"]

        if a_transpose:
            a = torch.empty((M, K))  # (M, K):(K, 1)
        else:
            a = torch.empty(K, M).transpose()  # (M, K):(1, M)

        if b_transpose:
            b = torch.empty((K, N)).transpose()  # (N, K):(1, N)
        else:
            b = torch.empty((N, K))  # (N, K):(K, 1)

        c = torch.zeros((M, N))  # (M, N):(N, 1)

        a = a.random_(-10, 10).to(device=torch.device("cuda"), dtype=torch.float32)
        b = b.random_(-10, 10).to(device=torch.device("cuda"), dtype=torch.float32)
        # a = torch.arange(0, M * K, device=torch.device("cuda"), dtype=torch.float32).reshape(M, K)
        # b = torch.arange(0, N * K, device=torch.device("cuda"), dtype=torch.float32).reshape(N, K)
        c = c.to(device=torch.device("cuda"), dtype=torch.float32)

        if return_type == "torch_only":
            return a, b, c

        a_tensor = from_dlpack(a, assumed_align=bytes_alignment)
        b_tensor = from_dlpack(b, assumed_align=bytes_alignment)
        c_tensor = from_dlpack(c, assumed_align=bytes_alignment)

        if dynamic_layout:
            a_tensor = a_tensor.mark_layout_dynamic(leading_dim=1 if a_transpose else 0)
            b_tensor = b_tensor.mark_layout_dynamic(leading_dim=0 if a_transpose else 0)
            c_tensor = c_tensor.mark_layout_dynamic(leading_dim=1)

        if return_type == "cute_only":
            return a_tensor, b_tensor, c_tensor

        return a, b, c, a_tensor, b_tensor, c_tensor

    a_torch, b_torch, c_torch, a_tensor, b_tensor, c_tensor = tensor_generator()

    # Verify
    if not skip_verify:
        sgemm(a_tensor, b_tensor, c_tensor)
        torch.cuda.synchronize()
        c_torch_ref = torch.einsum("mk,nk->mn", a_torch, b_torch)
        torch.testing.assert_close(c_torch.cpu(), c_torch_ref.cpu(), atol=1e-03, rtol=1e-05)
        print("Verification passed!")
    else:
        print("Verification skipped.")

    # Compile
    compile_tic = time.perf_counter()
    sgemm_compiled = cute.compile(sgemm, a_tensor, b_tensor, c_tensor)
    print(f"Kernel compiled in {time.perf_counter() - compile_tic:.4f} seconds")

    # Benchmarking
    workspace_bytes = (M * K + N * K + M * N) * 2
    workspace_count = testing.get_workspace_count(workspace_bytes, warmup_iterations, iterations)

    def torch_workspace_generator():
        return ["mk,nk->mn", *tensor_generator("torch_only")[:2]]

    def cute_workspace_generator():
        return testing.JitArguments(*tensor_generator("cute_only"))

    torch_avg_time_us = benchmark_torch(
        torch.einsum,
        torch_workspace_generator,
        workspace_count,
        warmup_iterations,
        iterations,
    )

    cute_avg_time_us = testing.benchmark(
        sgemm_compiled,
        workspace_generator=cute_workspace_generator,
        workspace_count=workspace_count,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        use_cuda_graphs=False,
    )

    GEMM_TOPS = 2 * M * N * K / 1e12
    print(f"Torch kernel execution time: {torch_avg_time_us:.2f} us")
    print(f"Torch achieved TOPS: {GEMM_TOPS / torch_avg_time_us * 1e6:.2f}")
    print(f"Cute kernel execution time: {cute_avg_time_us:.2f} us")
    print(f"Cute achieved TOPS: {GEMM_TOPS / cute_avg_time_us * 1e6:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example of elementwise ops to demonstrate the numpy/pytorch as input for kernels"
    )
    parser.add_argument("--M", "-M", default=1024, type=int)
    parser.add_argument("--N", "-N", default=1024, type=int)
    parser.add_argument("--K", "-K", default=1024, type=int)
    parser.add_argument("--blas", type=str, default="tn", choices=["tn", "tt", "nn", "nt"])
    parser.add_argument("--warmup-iterations", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--dynamic-layout", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    check_cuda()

    VERBOSE = args.verbose

    run_sgemm(
        args.M,
        args.N,
        args.K,
        blas=args.blas,
        skip_verify=args.skip_verify,
        dynamic_layout=args.dynamic_layout,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
    )
    print("PASS!")
