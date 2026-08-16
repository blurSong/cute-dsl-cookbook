from dataclasses import dataclass, field
from typing import Tuple, Type

import cutlass


@dataclass
class GemmConfig:
    a_dtype: Type[cutlass.Numeric]
    b_dtype: Type[cutlass.Numeric]
    c_dtype: Type[cutlass.Numeric]
    acc_dtype: Type[cutlass.Numeric]
    cta_tiler_mnk: Tuple[int, int, int]
    mma_inst_shape: Tuple[int, int, int] = (16, 8, 16)
    mma_atom_shape: Tuple[int, int, int] = (2, 2, 1)
    copy_bits: int = 128
    bytes_alignment: int = 16
    vla: int = field(init=False)
    vlb: int = field(init=False)

    def __post_init__(self) -> None:
        self.cta_tiler_mnk = tuple(self.cta_tiler_mnk)
        self.mma_inst_shape = tuple(self.mma_inst_shape)
        self.mma_atom_shape = tuple(self.mma_atom_shape)
        self.vla = self.copy_bits // self.a_dtype.width
        self.vlb = self.copy_bits // self.b_dtype.width
        self.check_sanity()

    def check_sanity(self) -> None:
        if len(self.cta_tiler_mnk) != 3:
            raise ValueError("cta_tiler_mnk must contain exactly 3 values")
        if any(dim <= 0 for dim in self.cta_tiler_mnk):
            raise ValueError("cta_tiler_mnk values must be positive")
        if self.copy_bits % self.a_dtype.width != 0:
            raise ValueError("copy_bits must be divisible by a_dtype width")
        if self.copy_bits % self.b_dtype.width != 0:
            raise ValueError("copy_bits must be divisible by b_dtype width")

        tile_m, tile_n, tile_k = self.cta_tiler_mnk
        mma_inst_m, mma_inst_n, mma_inst_k = self.mma_inst_shape
        mma_atom_m, mma_atom_n, mma_atom_k = self.mma_atom_shape
        if tile_m % (mma_atom_m * mma_inst_m) != 0:
            raise ValueError("CTA tile M must be divisible by MMA atom M")
        if tile_n % (mma_atom_n * mma_inst_n) != 0:
            raise ValueError("CTA tile N must be divisible by MMA atom N")
        if mma_atom_k != 1:
            raise ValueError("MMA atom K must be 1")
        if tile_k % mma_inst_k != 0:
            raise ValueError("CTA tile K must be divisible by MMA instruction K")
        if self.bytes_alignment % self.vla != 0:
            raise ValueError("bytes_alignment must be divisible by vla")
        if self.bytes_alignment % self.vlb != 0:
            raise ValueError("bytes_alignment must be divisible by vlb")


@dataclass
class GemmConfigSm120(GemmConfig):
    cluster_shape: Tuple[int, int, int] = (1, 1, 1)
    max_active_clusters: int = field(init=False)

    def __post_init__(self) -> None:
        self.cluster_shape = tuple(self.cluster_shape)
        super().__post_init__()
        self.max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
            self.cluster_shape[0] * self.cluster_shape[1]
        )

    def check_sanity(self) -> None:
        super().check_sanity()
        if len(self.cluster_shape) != 3:
            raise ValueError("cluster_shape must contain exactly 3 values")
        if any(dim <= 0 for dim in self.cluster_shape):
            raise ValueError("cluster_shape values must be positive")
