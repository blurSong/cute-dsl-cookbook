# This is a cute tma copy example copied from Yifan Yang.
# Some comments and variable names are changed for better readability.
# Reference Blog: https://yang-yifan.github.io/blogs/cute_tma/cute_tma.html

import cutlass.cute as cute
import cutlass.cute.runtime as cute_rt
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import Agent, agent_sync

import cutlass

CTA_M = 64
CTA_K = 256


@cute.kernel
def cute_tma_load_kernel(
    tma_load: cute.CopyAtom,
    tma_tensor: cute.Tensor,
    gmem_tensor: cute.Tensor,
    smem_layout: cute.Layout | cute.ComposedLayout,
    prefetch: cutlass.Constexpr = False,
):
    M, K = tma_tensor.shape
    tidx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()
    bytes = cute.size_in_bytes(gmem_tensor.element_type, smem_layout)

    # Create shared memory buffer
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(gmem_tensor.element_type, smem_layout, 16)

    # Initialize a single mbarrier (64bit)
    tma_load_mbar = smem.allocate_array(cutlass.Int64)

    # Initialize the barrier and set arrival count to 1. the initial phase is 0
    with cute.arch.elect_one():
        cute.arch.mbarrier_init(tma_load_mbar, 1)

    # barrier init fence to ensure barrier is visible to all threads
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()

    with cute.arch.elect_one():
        if block_idx == 0:
            cute.print_tensor(tma_tensor)

    """
    if cutlass.const_expr(prefetch):
        gmem_tensor_coord_cta_0 = cute.local_tile(
            tma_tensor, smem_layout.shape, (block_idx, 0)
        )
        _, tAgA = cpasync.tma_partition(
            tma_load,
            cta_coord=(0),
            cta_layout=cute.make_layout((1)),
            smem_tensor=cute.group_modes(sA, 0, 2),
            gmem_tensor=cute.group_modes(gmem_tensor_coord_cta_0, 0, 2),
        )
        cute.prefetch(tma_load, tAgA)
    """

    cute.arch.barrier()

    current_phase = 0
    # Load tiles of size (CTA_M, CTA_K) for K // CTA_K times
    for k in range(K // CTA_K):

        # Get the tile of gmem (coordinate) tensor of this k block, tiled from the whole gmem coordinate tensor
        # NOTE here coord of the local_tile is (bidx, k) to fetch the k-th tile of the gmem tensor [CTA_M, K]
        gmem_tensor_coord_cta = cute.local_tile(
            tma_tensor, smem_layout.shape, (block_idx, k)
        )

        # Set the expect_tx bytes to be loaded by tma
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(tma_load_mbar, bytes)

        # - cta_layout and cta_coord represents the cta coord in a cga, setting cta_layout to 1 means we use 1x1 cga
        # - tma partition only partition the first mode of smem/gmem tensor into various utmaldg instructions
        # - this means we need to manually group/pack the smem/gmem modes (that we want the tma to load) into mode 0
        #   then tAsA and tAgA will have shape [TMA_atom, rest]
        tAsA, tAgA = cpasync.tma_partition(
            tma_load,
            cta_coord=(0),
            cta_layout=cute.make_layout((1)),
            smem_tensor=cute.group_modes(sA, 0, 2),
            gmem_tensor=cute.group_modes(gmem_tensor_coord_cta, 0, 2),
        )

        cute.copy(tma_load, tAgA, tAsA, tma_bar_ptr=tma_load_mbar, mcast_mask=None)

        cute.arch.mbarrier_wait(tma_load_mbar, current_phase)

        # phase is flipped between 0 and 1
        current_phase = 1 - current_phase

        with cute.arch.elect_one():
            if block_idx == 0:
                if k == 0:
                    print(
                        f"[CuTeDSL][Kernel] gmem_tensor_coord_cta: {gmem_tensor_coord_cta}"
                    )
                    print(f"[CuTeDSL][Kernel] tAgA: {tAgA}")
                    print(f"[CuTeDSL][Kernel] tAsA: {tAsA}")
                # cute.printf("k: %d", k)
                # cute.print_tensor(sA)


@cute.kernel
def cute_tma_load_multicast_kernel(
    tma_load: cute.CopyAtom,
    tma_tensor: cute.Tensor,
    gmem_tensor: cute.Tensor,
    smem_layout: cute.Layout | cute.ComposedLayout,
):
    M, K = tma_tensor.shape
    tidx, _, _ = cute.arch.thread_idx()
    # the cta linearized rank within a cluster
    block_rank_in_cluster = cute.arch.block_idx_in_cluster()
    # the cluster identifier within a grid
    cluster_idx, cluster_idy, _ = cute.arch.cluster_idx()

    bytes = cute.size_in_bytes(gmem_tensor.element_type, smem_layout)
    # - When using multicast, we don't need to reduce the except_tx to 1/multicast.
    #   This is because although we break the smem_tensor load into 2 CopyAtom and let each CTA works on its own share,
    #   when the data arrives, TMA updates the transaction_bytes of both barriers in the cluster.

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(gmem_tensor.element_type, smem_layout, 16)

    # Initialize a single mbarrier (64bit)
    tma_load_mbar = smem.allocate_array(cutlass.Int64)

    # Initialize the barrier and set arrival count to 1. the initial phase is 0
    with cute.arch.elect_one():
        cute.arch.mbarrier_init(tma_load_mbar, 1)

    # barrier init fence to ensure barrier is visible to all threads
    cute.arch.mbarrier_init_fence()
    agent_sync(Agent.ThreadBlockCluster, is_relaxed=False)

    # Create tma multicast mask
    # mask = cpasync.create_tma_multicast_mask(
    #     cta_layout_vmnk=None,
    #     cta_coord_vmnk=None,
    #     mcast_mode=2,
    # )

    mask = cutlass.Int16(1) << 2 - 1  # multicast to 2 ctas, so the mask should be 0b11

    with cute.arch.elect_one():
        if cluster_idx == 0 and cluster_idy == 0:
            cute.print_tensor(tma_tensor)

    current_phase = 0
    # 2 CTAs in a cluster load tiles of size (CTA_M, CTA_K) for K // CTA_K times
    for k in range(K // CTA_K):

        # Get the tile of gmem (coordinate) tensor of this k block, tiled from the whole gmem coordinate tensor
        # NOTE here coord of the local_tile is (cidx, k) to fetch the k-th tile of the gmem tensor [CTA_M, K]
        gmem_tensor_coord_cluster = cute.local_tile(
            tma_tensor, smem_layout.shape, (cluster_idx, k)
        )

        # Set the expect_tx bytes to be loaded by tma
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(tma_load_mbar, bytes)

        # - cta_layout and cta_coord represents the cta coord in a cga, setting cta_layout to block_rank_in_cluster means we use 1x2 cga
        # - tma partition only partition the first mode of smem/gmem tensor into various utmaldg instructions
        # - this means we need to manually group/pack the smem/gmem modes (that we want the tma to load) into mode 0
        #   then tAsA and tAgA will have shape [TMA_atom, rest]
        tAsA, tAgA = cpasync.tma_partition(
            tma_load,
            cta_coord=(block_rank_in_cluster,),
            cta_layout=cute.make_layout((2,)),
            smem_tensor=cute.group_modes(sA, 0, 2),
            gmem_tensor=cute.group_modes(gmem_tensor_coord_cluster, 0, 2),
        )

        cute.copy(tma_load, tAgA, tAsA, tma_bar_ptr=tma_load_mbar, mcast_mask=mask)

        cute.arch.mbarrier_wait(tma_load_mbar, current_phase)

        current_phase = 1 - current_phase

        with cute.arch.elect_one():
            if cluster_idx == 0:
                if k == 0:
                    print(
                        f"[CuTeDSL][Kernel] gmem_tensor_coord_cluster: {gmem_tensor_coord_cluster}"
                    )
                    print(f"[CuTeDSL][Kernel] tAgA: {tAgA}")
                    print(f"[CuTeDSL][Kernel] tAsA: {tAsA}")
                # cute.printf("k: %d", k)
                # cute.print_tensor(sA)


@cute.jit
def cute_host_load(
    a: cute.Tensor,
    prefetch: cutlass.Constexpr = False,
):
    # Initialize CUDA context for launching a kernel with error checking
    # We make context initialization explicit to allow users to control the context creation
    # and avoid potential issues with multiple contexts
    cutlass.cuda.initialize_cuda_context()

    M, K = a.shape

    gmem_layout = cute.make_layout((M, K), stride=(K, 1))
    gmem_tensor = cute.make_tensor(a.iterator, gmem_layout)
    smem_layout = cute.make_layout((CTA_M, CTA_K), stride=(CTA_K, 1))
    # smem_layout = cute.make_composed_layout(cute.make_swizzle(0, 4, 3), 0, smem_layout)
    # tma_tensor is the arithmetic tuple tracking the coordinate of the gmem tensor to load from
    tma_load, tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        gmem_tensor,
        smem_layout,
        cta_tiler=(CTA_M, CTA_K),
    )

    # need to explicitly calculate smem usage to pass in to kernel launch
    bytes = cute.size_in_bytes(gmem_tensor.element_type, smem_layout) + 8

    print(f"[CuTeDSL][Jit] gmem_layout: {gmem_layout}")
    print(f"[CuTeDSL][Jit] smem_layout: {smem_layout}")
    print(f"[CuTeDSL][Jit] tma_load: {tma_load}")
    print(f"[CuTeDSL][Jit] tma_tensor: {tma_tensor}")

    # Launch kernel
    # each cta will load tiles of size [CTA_M, CTA_K] for K // CTA_K times using tma.
    cute_tma_load_kernel(
        tma_load,
        tma_tensor,
        gmem_tensor,
        smem_layout,
        prefetch,
    ).launch(
        grid=(M // CTA_M, 1, 1),  # Single thread block
        block=(32, 1, 1),  # One warp (32 threads) per thread block
        smem=bytes,
    )


@cute.jit
def cute_host_load_multicast(
    a: cute.Tensor,
    prefetch: cutlass.Constexpr = False,
):
    cutlass.cuda.initialize_cuda_context()

    M, K = a.shape
    gmem_layout = cute.make_layout((M, K), stride=(K, 1))
    gmem_tensor = cute.make_tensor(a.iterator, gmem_layout)
    smem_layout = cute.make_layout((CTA_M, CTA_K), stride=(CTA_K, 1))

    # Here the tma_op is `cp.async.bulk.tensor.multicast`.
    tma_load, tma_tensor = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SMulticastOp(),
        gmem_tensor,
        smem_layout,
        cta_tiler=(CTA_M, CTA_K),
        num_multicast=2,  # multicast to 2 ctas in a cluster
    )

    print(f"[CuTeDSL][Jit] gmem_layout: {gmem_layout}")
    print(f"[CuTeDSL][Jit] smem_layout: {smem_layout}")
    print(f"[CuTeDSL][Jit] tma_load: {tma_load}")
    print(f"[CuTeDSL][Jit] tma_tensor: {tma_tensor}")

    # Launch kernel
    cute_tma_load_multicast_kernel(
        tma_load,
        tma_tensor,
        gmem_tensor,
        smem_layout,
    ).launch(
        grid=(M // CTA_M, 2, 1),
        block=(32, 1, 1),  # One warp (32 threads) per thread block
        cluster=(
            1,
            2,
            1,
        ),  # 2 ctas in one cluster along dim_y load one (CTA_M, CTA_K) tile with multicasting
    )


def run_tma(M, K):
    a = torch.arange(M * K, device="cuda", dtype=torch.int32).view(M, K)
    print(f"{'-'*10} TMA Load {'-'*10}")
    cute_host_load(from_dlpack(a))
    torch.cuda.synchronize()

    print(f"{'-'*10} TMA Load with Multicast{'-'*10}")
    cute_host_load_multicast(from_dlpack(a))
    torch.cuda.synchronize()

if __name__ == "__main__":
    run_tma(2112, 1792)
