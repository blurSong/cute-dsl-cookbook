"""
A dense HGEMM using the sm120 cuda cores in CUTE DSL.

The HGEMM implementation handles only row-major and col-major layouts.
To bridge the gap of GEMM order between BLAS and CUTE, we can use the following definitions:
------------------------------------------
Blas      T                   N
------------------------------------------
A         (M, K):(K, 1)      (M, K):(1, M)
B         (N, K):(1, N)      (N, K):(K, 1)
------------------------------------------
See also:
    https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0x_gemm_tutorial.html#aside-m-major-n-major-k-major

Key optimizations:
    TODO.

References:
    TODO.

To run:
    TODO.

Constraints:
    TODO.
"""

import argparse
from functools import partial
import math
import time
from typing import List, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import cutlass.testing as testing
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.nvgpu import cpasync, warp  # sm120 is not tcgen05
from cutlass.utils import LayoutEnum
from ..utils import (
    benchmark_torch,
    check_cuda,
    parse_comma_separated_ints,
    str_to_cutlass_dtype,
    generate_gemm_tensors,
)


import cutlass

VERBOSE = False
LOG = "[CuTe Info]"


class GemmConfig:
    # DTYPES
    a_dtype = cutlass.Float16
    b_dtype = cutlass.Float16
    c_dtype = cutlass.Float16
    acc_dtype = cutlass.Float32
    # CONFIGURATIONS
    cluster_shape = (1, 1, 1)
    cta_tiler_mnk = (128, 128, 64)
    mma_inst_shape = (16, 8, 16)
    mma_atom_shape = (2, 2, 1)
    copy_bits = 128
    bytes_alignment = 16
    vla = copy_bits // a_dtype.width
    vlb = copy_bits // b_dtype.width
    max_active_clusters = 0

    def check_sanity(self):
        # PARAMETERS DERIVED
        tile_m, tile_n, tile_k = self.cta_tiler_mnk
        mma_inst_m, mma_inst_n, mma_inst_k = self.mma_inst_shape
        mma_atom_m, mma_atom_n, mma_atom_k = self.mma_atom_shape
        assert tile_m % (mma_atom_m * mma_inst_m) == 0
        assert tile_n % (mma_atom_n * mma_inst_n) == 0
        assert mma_atom_k == 1
        assert tile_k % mma_inst_k == 0
        assert self.bytes_alignment % self.vla == 0
        assert self.bytes_alignment % self.vlb == 0


class Sm120GemmKernel:
    def __init__(self, config: GemmConfig):
        # kernel
        self.sm_version = "sm_120"
        self.gemm_dtype = config.gemm_dtype
        self.acc_dtype = config.acc_dtype
        # cluster
        self.occupancy = 1
        self.cluster_shape = config.cluster_shape
        self.num_mcast_ctas_A = self.cluster_shape[1]
        self.num_mcast_ctas_B = self.cluster_shape[0]
        self.max_active_clusters = config.max_active_clusters
        # mma
        self.tiled_mma = None
        self.mma_atom_shape = config.mma_atom_shape
        self.mma_inst_shape = config.mma_inst_shape
        # cta
        self.cta_tiler = config.cta_tiler
        self.epilogue_tiler = None
        self.num_mma_warps = math.prod(config.mma_atom_shape)
        self.num_dma_warps = 1
        self.warp_size = 32
        self.threads = (self.num_mma_warps + self.num_dma_warps) * self.warp_size
        # smem
        self.smem_capacity = cutlass.utils.get_smem_capacity_in_bytes(self.sm_version)
        self.num_stages_AB = None
        self.num_stages_C = None
        self.smem_layout_A = None
        self.smem_layout_B = None
        self.smem_layout_C = None
        self.shared_storage = None
        # other
        self.tma_copy_bytes = None
        self.epilogue_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.warp_size,
        )
        self.buffer_align_bytes = 1024
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

    def calculate_stages(self):
        """Computes the number of stages for A/B/C operands based on heuristics."""
        assert self.epilogue_tiler
        assert not self.num_stages_AB
        assert not self.num_stages_C

        num_stages_C = 8
        bytes_mbar_helpers = 1024
        bytes_per_value = self.gemm_dtype.width // 8
        bytes_per_stage_C = cute.size(self.epilogue_tiler) * bytes_per_value
        bytes_per_stage_AB = (
            (self.cta_tiler[0] + self.cta_tiler[1]) * self.cta_tiler[2] * bytes_per_value
        )
        num_stages_AB = (
            (self.smem_capacity - self.occupancy * 1024) // self.occupancy
            - bytes_mbar_helpers
            - bytes_per_stage_C * num_stages_C
        ) // bytes_per_stage_AB

        assert num_stages_AB > 0
        self.num_stages_AB = num_stages_AB
        self.num_stages_C = num_stages_C

    def make_tiled_mma(self):
        assert not self.tiled_mma
        mma_op = warp.MmaF16BF16Op(
            self.gemm_dtype,
            self.acc_dtype,
            self.mma_inst_shape,
        )
        mma_atom_layout = cute.make_layout(self.mma_atom_shape)
        permutation = (
            self.mma_atom_shape[0] * self.mma_inst_shape[0],
            self.mma_atom_shape[1] * self.mma_inst_shape[1] * 2,
            self.mma_atom_shape[2] * self.mma_inst_shape[2],
        )
        self.tiled_mma = cute.make_tiled_mma(mma_op, mma_atom_layout, permutation)

    def make_tma_atoms_and_tensors(
        self,
        tensor: cute.Tensor,
        smem_layout: cute.ComposedLayout | cute.Layout,
        smem_tiler: cutlass.Shape,
        direction: cutlass.Constexpr = "G2S",
        mcast_dim: cutlass.Constexpr = 1,
    ):
        assert direction in ["G2S", "S2G"]

        if cutlass.const_expr(direction == "G2S"):
            if cutlass.const_expr(mcast_dim == 1):
                tma_op = cpasync.CopyBulkTensorTileG2SOp()
            else:
                tma_op = cpasync.CopyBulkTensorTileG2SMulticastOp()
        elif cutlass.const_expr(direction == "S2G"):
            tma_op = cpasync.CopyBulkTensorTileS2GOp()
        smem_layout = cute.slice_(smem_layout, (None, None, 0))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            tma_op, tensor, smem_layout, smem_tiler, num_multicast=mcast_dim
        )
        return tma_atom, tma_tensor

    def calculate_epilogue_tiler(self):
        self.epilogue_tiler = sm90_utils.compute_tile_shape_or_override(
            self.cta_tiler, self.gemm_dtype, is_cooperative=False
        )

    def _calculate_grid(
        self,
        output_tensor: cute.Tensor,
    ):
        num_ctas_shape = cute.ceil_div(output_tensor.shape, self.cta_tiler[:2])
        tile_sched_params = cutlass.utils.PersistentTileSchedulerParams(
            num_ctas_shape,
            self.cluster_shape,
            # Note here the default swizzle is 1. I.e., no grid rasterization
        )
        grid = tile_sched_params.get_grid_shape(self.max_active_clusters)
        return grid, tile_sched_params

    @cute.jit
    def __call__(
        self,
        A: cute.Tensor,
        B: cute.Tensor,
        C: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        # Make input-irrelevant initializations
        self.calculate_epilogue_tiler()
        self.calculate_stages()
        self.make_tiled_mma()

        # A can be ROW_MAJOR@T or COL_MAJOR@N
        # B can be COL_MAJOR@T or ROW_MAJOR@N
        major_mode_A = LayoutEnum.from_tensor(A)
        major_mode_B = LayoutEnum.from_tensor(B)
        major_mode_C = LayoutEnum.from_tensor(C)

        # See nvgpu.warpgroup.helpers.make_smem_layout_atom for swizzle.
        self.smem_layout_A = sm90_utils.make_smem_layout_a(
            major_mode_A,
            self.cta_tiler,
            self.gemm_dtype,
            self.num_stages_AB,
        )
        self.smem_layout_B = sm90_utils.make_smem_layout_b(
            major_mode_B,
            self.cta_tiler,
            self.gemm_dtype,
            self.num_stages_AB,
        )
        self.smem_layout_C = sm90_utils.make_smem_layout_epi(
            self.gemm_dtype,
            major_mode_C,
            self.epilogue_tiler,
            self.num_stages_C,
        )

        tma_atom_A, tma_tensor_A = self.make_tma_atoms_and_tensors(
            A, self.smem_layout_A, (self.cta_tiler[0], self.cta_tiler[2]), "G2S"
        )
        tma_atom_B, tma_tensor_B = self.make_tma_atoms_and_tensors(
            B, self.smem_layout_B, (self.cta_tiler[1], self.cta_tiler[2]), "G2S"
        )
        tma_atom_C, tma_tensor_C = self.make_tma_atoms_and_tensors(
            C, self.smem_layout_C, self.epilogue_tiler, "S2G"
        )

        self.tma_copy_bytes = (
            self.gemm_dtype.width // 8 * self.cta_tiler[2] * (self.cta_tiler[0] + self.cta_tiler[1])
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64,
                self.num_stages_AB * 2,
            ]
            sA = cute.struct.Align[
                cute.struct.MemRange[self.gemm_dtype, cute.cosize(self.smem_layout_A)],
                self.buffer_align_bytes,
            ]
            sB = cute.struct.Align[
                cute.struct.MemRange[self.gemm_dtype, cute.cosize(self.smem_layout_B)],
                self.buffer_align_bytes,
            ]
            sC = cute.struct.Align[
                cute.struct.MemRange[self.gemm_dtype, cute.cosize(self.smem_layout_C)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        grid, tile_sched_params = self._calculate_grid(C)
        block = [self.threads, 1, 1]

        if VERBOSE:
            print(f"{LOG} A {A}")
            print(f"{LOG} B {B}")
            print(f"{LOG} C {C}")
            print(f"{LOG} grid {grid}")
            print(f"{LOG} block {block}")
            print(f"{LOG} cluster {self.cluster_shape}")
            print(f"{LOG} tile_sched_params {tile_sched_params}")
            print(f"{LOG} smem_layout_A {self.smem_layout_A}")
            print(f"{LOG} smem_layout_B {self.smem_layout_B}")
            print(f"{LOG} smem_layout_C {self.smem_layout_C}")
            print(f"{LOG} tma_atom_A {tma_atom_A}")
            print(f"{LOG} tma_atom_B {tma_atom_B}")
            print(f"{LOG} tma_atom_C {tma_atom_C}")
            print(f"{LOG} tma_tensor_A {tma_tensor_A}")
            print(f"{LOG} tma_tensor_B {tma_tensor_B}")
            print(f"{LOG} tma_tensor_C {tma_tensor_C}")
            print(f"{LOG} tiled_mma {self.tiled_mma}")

        self.kernel(
            tma_tensor_A,
            tma_tensor_B,
            tma_tensor_C,
            tma_atom_A,
            tma_atom_B,
            tma_atom_C,
            tile_sched_params,
        ).launch(grid=grid, block=block, cluster=self.cluster_shape)

    @cute.kernel
    def kernel(
        self,
        A: cute.Tensor,
        B: cute.Tensor,
        C: cute.Tensor,
        tma_atom_A: cute.CopyAtom,
        tma_atom_B: cute.CopyAtom,
        tma_atom_C: cute.CopyAtom,
        tile_sched_params: cutlass.utils.PersistentTileSchedulerParams,
    ):
        thr_idx = cute.arch.thread_idx()[0]
        blk_idz = cute.arch.block_idx()[2]
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch Tma descriptor
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_A)
            cpasync.prefetch_descriptor(tma_atom_B)
            cpasync.prefetch_descriptor(tma_atom_C)

        cta_layout_in_cluster = cute.make_layout(self.cluster_shape)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        coord_cluster = cta_layout_in_cluster.get_flat_coord(cta_rank_in_cluster)

        # Get mcast mask
        a_mcast_mask = cute.make_layout_image_mask(cta_layout_in_cluster, coord_cluster, mode=1)
        b_mcast_mask = cute.make_layout_image_mask(cta_layout_in_cluster, coord_cluster, mode=0)
        a_mcast_mask = a_mcast_mask if self.num_mcast_ctas_A > 1 else 0
        b_mcast_mask = b_mcast_mask if self.num_mcast_ctas_B > 1 else 0

        mcast_size = self.num_mcast_ctas_A + self.num_mcast_ctas_B - 1
        consumer_arrive_cnt = mcast_size * self.num_mma_warps

        # Alloc and init AB full/empty + ACC full mbar (pipeline)
        shared_storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
        mainloop_pipeline_array_ptr = shared_storage.mainloop_pipeline_array_ptr.data_ptr()
        mainloop_pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread
        )
        mainloop_pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            size=consumer_arrive_cnt,
        )
        cta_layout_vmnk = cute.make_layout((1, *cta_layout_in_cluster.shape))
        mainloop_pipeline = cutlass.pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages_AB,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=self.tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        #  Cluster arrive after barrier init
        if cute.size(self.cluster_shape) > 1:
            cute.arch.cluster_arrive_relaxed()

        # Partition the gmem of A, B and C
        # * gA (tile_m, tile_k, num_m_tiles, num_k_tiles, batch)
        # * gB (tile_n, tile_k, num_n_tiles, num_k_tiles, batch)
        # * gC (tile_m, tile_n, num_m_tiles, num_n_tiles, batch)
        coord_cta = (None, None, None)
        gA = cute.local_tile(A, (self.cta_tiler[0], self.cta_tiler[2]), coord=coord_cta)
        gB = cute.local_tile(B, (self.cta_tiler[1], self.cta_tiler[2]), coord=coord_cta)
        gC = cute.local_tile(C, (self.cta_tiler[0], self.cta_tiler[1]), coord=coord_cta)

        # Manage smem and partition shared tensor for TMA load A/B
        sA = shared_storage.sA.get_tensor(
            self.smem_layout_A.outer, swizzle=self.smem_layout_A.inner
        )
        sB = shared_storage.sB.get_tensor(
            self.smem_layout_B.outer, swizzle=self.smem_layout_B.inner
        )
        sC = shared_storage.sC.get_tensor(
            self.smem_layout_C.outer, swizzle=self.smem_layout_C.inner
        )

        cta_layout_A = cute.make_layout((self.cluster_shape[1]))
        cta_layout_B = cute.make_layout((self.cluster_shape[0]))
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_A,
            cta_coord=coord_cluster[1],
            cta_layout=cta_layout_A,
            smem_tensor=cute.group_modes(sA, 0, 2),
            gmem_tensor=cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_B,
            cta_coord=coord_cluster[0],
            cta_layout=cta_layout_B,
            smem_tensor=cute.group_modes(sB, 0, 2),
            gmem_tensor=cute.group_modes(gB, 0, 2),
        )

        # Get thread MMA and fragments
        thr_mma = self.tiled_mma.get_slice(thr_idx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = thr_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = thr_mma.make_fragment_B(tCsB[None, None, None, 0])

        tCgC = thr_mma.partition_C(gC)
        tCrC_mma = cute.make_rmem_tensor_like(tCgC[None, None, None, 0], dtype=self.acc_dtype)

        # Cluster wait for barrier init
        if cute.size(self.cluster_shape) > 1:
            cute.arch.cluster_wait()
        else:
            # this will call cute.arch.barrier -> nvvm.barrier
            # same as sync_threads()
            cutlass.pipeline.sync(barrier_id=1)

        num_k_tiles = cute.ceil_div(A.shape[1], self.cta_tiler[2])

        # Create the tile scheduler
        tile_sched = cutlass.utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        # Create the pipeline states for P and C
        mainloop_producer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.num_stages_AB
        )
        mainloop_consumer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.num_stages_AB
        )


def run_gemm(
    mnkl: Tuple[int, int, int, int],
    tile_shape_mnk: Tuple[int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    c_major: str,
    warmup_iterations: int = 10,
    iterations: int = 100,
    skip_verify: bool = False,
    dynamic_layout: bool = False,
):

    print("Running SM120 Dense GEMM with:")
    print(f"GEMM mnkl: {mnkl}")
    print(f"Tile mnk: {tile_shape_mnk}")
    print(f"Dtype - A: {a_dtype}, B: {b_dtype}, C: {c_dtype}, Acc: {acc_dtype}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Iterations: {warmup_iterations} warmup, {iterations} run")
    print(f"Skip verification: {skip_verify}")

    M, N, K, L = mnkl

    config = GemmConfig()
    config.a_dtype = a_dtype
    config.b_dtype = b_dtype
    config.c_dtype = c_dtype
    config.acc_dtype = acc_dtype
    config.cta_tiler_mnk = tile_shape_mnk
    config.max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
        config.cluster_shape[0] * config.cluster_shape[1]
    )

    if VERBOSE:
        print(f"{LOG} GemmConfig: {config.__dict__}")

    # Verification and compilation

    gemm = Sm120GemmKernel(config)

    general_tensor_generator = partial(
        generate_gemm_tensors,
        M,
        N,
        K,
        L,
        a_major,
        b_major,
        c_major,
        a_dtype,
        b_dtype,
        c_dtype,
        dynamic_layout,
        config.bytes_alignment,
    )

    (
        a_cute,
        b_cute,
        c_cute,
        a_torch,
        b_torch,
        c_torch,
    ) = general_tensor_generator(return_type="both")

    if not skip_verify:
        gemm(a_cute, b_cute, c_cute)
        torch.cuda.synchronize()
        c_ref_torch = torch.einsum(
            "mkl,nkl->mnl",
            a_torch.to(dtype=torch.float32),
            b_torch.to(dtype=torch.float32),
        ).to(dtype=c_torch.dtype)
        torch.testing.assert_close(c_torch.cpu(), c_ref_torch.cpu(), atol=1e-01, rtol=1e-03)
        print("Verification passed.")
    else:
        print("Verification skipped.")

    tic = time.perf_counter()
    gemm_compiled = cute.compile(gemm, a_cute, b_cute, c_cute)
    print(f"Kernel compiled time {time.perf_counter() - tic:.4f} seconds")

    # benchmarking
    workspace_bytes = (M * K * a_dtype.bytes + N * K * b_dtype.bytes + M * N * c_dtype.bytes) * L
    workspace_count = testing.get_workspace_count(workspace_bytes, warmup_iterations, iterations)

    def torch_workspace_generator():
        return [
            "mkl,nkl->mnl",
            *general_tensor_generator(return_type="torch")[:2],
        ]

    def cute_workspace_generator():
        return cutlass.testing.JitArguments(
            *general_tensor_generator(return_type="cute"),
        )

    torch_avg_time_us = benchmark_torch(
        torch.einsum,
        torch_workspace_generator,
        workspace_count,
        warmup_iterations,
        iterations,
    )

    cute_avg_time_us = testing.benchmark(
        gemm_compiled,
        workspace_generator=cute_workspace_generator,
        workspace_count=workspace_count,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        use_cuda_graphs=False,
    )

    GEMM_TOPS = 2 * M * N * K * L / 1e12
    print(f"Torch kernel execution time: {torch_avg_time_us:.2f} us")
    print(f"Torch achieved TOPS: {GEMM_TOPS / torch_avg_time_us * 1e6:.2f}")
    print(f"Cute kernel execution time: {cute_avg_time_us:.2f} us")
    print(f"Cute achieved TOPS: {GEMM_TOPS / cute_avg_time_us * 1e6:.2f}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Example of elementwise ops to demonstrate the numpy/pytorch as input for kernels"
    )
    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(1024, 2048, 4096, 1),
        help="MNKL dimensions (comma-separated)",
    )
    parser.add_argument(
        "--tile_shape_mnk",
        type=parse_comma_separated_ints,
        choices=[
            (64, 64, 64),
            (64, 128, 64),
            (128, 64, 64),
            (128, 128, 64),
            (128, 256, 64),
            (128, 128, 128),
        ],
        default=(128, 128, 64),
        help="CTA tile shape (comma-separated)",
    )
    parser.add_argument("--a-dtype", type=str, default="float16")
    parser.add_argument("--b-dtype", type=str, default="float16")
    parser.add_argument("--c-dtype", type=str, default="float16")
    parser.add_argument("--acc-dtype", type=str, default="float32")
    parser.add_argument("--a-major", type=str, choices=["k", "m"], default="k")
    parser.add_argument("--b-major", type=str, choices=["k", "n"], default="k")
    parser.add_argument("--c-major", type=str, choices=["n", "m"], default="n")
    parser.add_argument("--warmup-iterations", default=10, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--dynamic-layout", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    if len(args.mnkl) != 4:
        parser.error("--mnkl must contain exactly 4 values")
    return args


if __name__ == "__main__":
    args = parse_arguments()

    check_cuda()

    VERBOSE = args.verbose

    run_gemm(
        args.mnkl,
        args.tile_shape_mnk,
        str_to_cutlass_dtype(args.a_dtype),
        str_to_cutlass_dtype(args.b_dtype),
        str_to_cutlass_dtype(args.c_dtype),
        str_to_cutlass_dtype(args.acc_dtype),
        args.a_major,
        args.b_major,
        args.c_major,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        skip_verify=args.skip_verify,
        dynamic_layout=args.dynamic_layout,
    )
    print("PASS!")
