import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.utils import print_latex


@cute.jit
def test_cta_v_map(
    gmem_tensor: cute.Tensor,
):
    """Test the computation of CTA V Map in cpasync.make_tiled_tma_atom
    """
    cta_tiler = (64, 128)
    cta_v_map = cute.composition(cute.make_identity_layout(gmem_tensor.shape), cta_tiler)
    print("CTA V Map:", cta_v_map)
    # print_latex(cta_v_map)


def run_test_cta_v_map():
    torch_tensor = cutlass_torch.matrix(
        1, 1024, 2048, is_mode0_major=False, cutlass_dtype=cutlass.Float32
    )
    cute_tensor, _ = cutlass_torch.cute_tensor_like(
        torch_tensor, cutlass.Float32, is_dynamic_layout=False
    )
    test_cta_v_map(cute_tensor)


if __name__ == "__main__":
    run_test_cta_v_map()
