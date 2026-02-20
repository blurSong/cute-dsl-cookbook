"""
To understand the calculation of swizzle,
refer to https://veitner.bearblog.dev/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns/
"""
from typing import Type

import cutlass
import cutlass.cute as cute

from cutlass.cute.nvgpu.tcgen05 import (
    SmemLayoutAtomKind,
    make_smem_layout_atom,
)
from cutlass.utils import print_latex

import cuda.bindings.driver as cuda


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


@cute.jit
def simple_swizzle(
    S: cute.Shape,
    D: cute.Stride,
    bms: cute.IntTuple,
    coord: cute.IntTuple,
):
    """
    To understand how to compute the  swizzle. We can write the index in binary `bin(index)`.
    Note that Swizzle<2,4,3> will act on a number of the form `0bxxxxxxxUVxXYxxxx` such that X will get flipped
    if U=1 and Y will get flipped if V=1.
    """
    L = cute.make_layout(S, stride=D)
    b, m, s = bms[0], bms[1], bms[2]
    sw = cute.make_swizzle(b, m, s)
    L_swizzled = cute.make_composed_layout(sw, 0, L)

    index = cute.crd2idx(coord, L)
    index_swizzled = cute.crd2idx(coord, L_swizzled)
    print(f"coord: {coord}")
    print(f"index: {index}")
    print(f"index binary: {bin(index)}")
    print(f"index_swizzled: {index_swizzled}")
    print(f"index_swizzled binary: {bin(index_swizzled)}")


@cute.jit
def print_smem_layout_atom(
    element_type: Type[cutlass.Numeric],
    smem_layout_atom_kind: cutlass.Constexpr[SmemLayoutAtomKind],
):
    smem_layout_atom = make_smem_layout_atom(
        smem_layout_atom_kind,
        element_type,
    )
    # The `atom` is essentially a layout or composed layout.
    print_latex(smem_layout_atom)


@cute.jit
def print_smem_layout(
    element_type: Type[cutlass.Numeric],
    smem_layout_atom_kind: cutlass.Constexpr[SmemLayoutAtomKind],
    smem_shape: cute.Shape,
    order: cute.Shape,
):
    smem_layout_atom = make_smem_layout_atom(
        smem_layout_atom_kind,
        element_type,
    )
    tiled_smem_layout = cute.tile_to_shape(smem_layout_atom, smem_shape, order=order)
    # With order=(1, 2), basically we say the swizzle atom is first stacked along M, then K
    print_latex(tiled_smem_layout)


def test_simple_swizzle():
    S = (8, 32)
    D = (32, 1)
    bms = (2, 4, 3)
    coord = (7, 25)
    simple_swizzle(S, D, bms, coord)


def test_print_swizzle():
    elem_type = cutlass.Float8E4M3
    smem_layout_atom_kind = SmemLayoutAtomKind.K_SW128
    print_smem_layout_atom(elem_type, smem_layout_atom_kind)


def test_print_smem_layout():
    smem_shape = (128, 128)
    elem_type = cutlass.Float16
    smem_layout_atom_kind = SmemLayoutAtomKind.K_SW32
    order = (1, 2)
    print_smem_layout(elem_type, smem_layout_atom_kind, smem_shape, order)


if __name__ == "__main__":
    # test_simple_swizzle()
    test_print_swizzle()
    # test_print_smem_layout()
