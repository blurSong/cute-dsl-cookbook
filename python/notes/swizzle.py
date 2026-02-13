import time
import math
import torch
import argparse
from typing import Type, List, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch

# https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/

def swizzle(offset, b_bits, m_base, s_shift=None):
    """
    CuTe like swizzle function.
    https://github.com/NVIDIA/cutlass/include/cute/swizzle.hpp#L55
    Args:
        offset (int): The offset to swizzle.
        b_bits (int): The number of bits in the base address.
        m_base (int): The base address for the swizzle.
        s_shift (int): The shift value for the swizzle.
    """
    if s_shift is None:
        s_shift = b_bits
    assert b_bits >= 0 and m_base >= 0 and abs(s_shift) >= b_bits

    bit_mask = (1 << b_bits) - 1
    yyy_mask = bit_mask << (m_base + max(0, s_shift))
    zzz_mask = bit_mask << (m_base - min(0, s_shift))
    mask_shift = s_shift

    swizzle_code = yyy_mask | zzz_mask


def make_smem_layout_AB(
    major_mode: LayoutEnum,
    smem_tiler: Tuple[int, int, int],
):
    """Make shared memory layout for A and B tiles with swizzling.

    The PTX `ldmatrix` section describes the ldmatrix instruction in detail.
    The PTX `Shared Memory Layout and Swizzling` section provides all useful `swizzle layout atom`s.

    Since we are doing fp16 gemm, we manually apply a `<3,3,3>` swizzle onto a 8x128B tile.
    I.e., the `128B Swizzling with 16B atomicity` follows the PTX naming convention.

    - https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-mma

    Refer to this awesome blog for the ldmatrix/swizzle mechanism:
        - https://yang-yifan.github.io/blogs/mma_swizzle/mma_swizzle.html

    Params:
        - major_mode: `LayoutEnum.ROW_MAJOR` or `LayoutEnum.COL_MAJOR`
        - smem_tiler: smem sizes of `row`, `col` and `depth`
    """
    contiguous_size = smem_tiler[1] if major_mode == LayoutEnum.ROW_MAJOR else smem_tiler[0]
    contiguous_size = min(64, contiguous_size)
    # swizzle layout atom is upto `128B Swizzling with 16B atomicity`

    m_base = 3
    # one row of 2^3=8 values loaded by 1 thread quad
    s_shift = 3
    # 8 rows per ldmatrix.m8n8.b16
    b_bits = min(int(math.log2(contiguous_size / vl)), 3)
    # - b_bits is limited upto 3 because smem bank size is 32x4B.
    #   For ldmatrix.m8n8.b16 the up-most swizzle BBits is log2(128B/16B)=3.
    # - By default vl is 8 values (16B), making the BBits exactly 3.
    #   If vl is larger, to avoid smem bank conflicts,
    #   we will need less BBits to cover the contiguous_size.
    swizzle = cute.make_swizzle(b_bits, m_base, s_shift)

    if major_mode == LayoutEnum.ROW_MAJOR:
        # K-Major swizzle
        layout_atom_outer = cute.make_ordered_layout((8, contiguous_size), order=(1, 0))
    else:
        # MN-Major swizzle
        layout_atom_outer = cute.make_ordered_layout((contiguous_size, 8), order=(0, 1))

    layout_atom = cute.make_composed_layout(swizzle, 0, layout_atom_outer)
    layout = cute.tile_to_shape(layout_atom, smem_tiler, order=(0, 1, 2))
    return layout


def make_smem_layout_C(
    smem_tiler: Tuple[int, int],
):
    """C is always row-major layout (K-Major SMEM atom).

    Params:
        - smem_tiler: smem sizes of `row` and `col`
    """
    contiguous_size = smem_tiler[1]
    m_base = 3
    s_shift = 4
    b_bits = min(int(math.log2(contiguous_size / vl)), 3)
    swizzle = cute.make_swizzle(b_bits, m_base, s_shift)

    layout_atom_outer = cute.make_ordered_layout((8, contiguous_size), order=(1, 0))
    layout_atom = cute.make_composed_layout(swizzle, 0, layout_atom_outer)
    layout = cute.tile_to_shape(layout_atom, smem_tiler, order=(0, 1))
    return layout
