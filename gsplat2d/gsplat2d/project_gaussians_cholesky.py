from typing import Tuple

from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

from gsplat2d import cuda as _C


def project_gaussians_cholesky(
    cholesky: Float[Tensor, "*batch 3"],
    means2d: Float[Tensor, "*batch 2"],
    img_height: int,
    img_width: int,
    block_width: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Project gaussians directly from cholesky decomposition.

    Args:
        cholesky: [l11, l21, l22] elements of lower triangular cholesky factor
        means2d: 2D positions
        img_height, img_width: image dimensions
        block_width: tile block width (2-16)

    Returns:
        xys, radii, conics, num_tiles_hit
    """
    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"
    return _ProjectGaussiansCholesky.apply(
        cholesky.contiguous(),
        means2d.contiguous(),
        img_height,
        img_width,
        block_width,
    )


class _ProjectGaussiansCholesky(Function):

    @staticmethod
    def forward(
        ctx,
        cholesky: Float[Tensor, "*batch 3"],
        means2d: Float[Tensor, "*batch 2"],
        img_height: int,
        img_width: int,
        block_width: int,
    ):
        num_points = cholesky.shape[-2]
        if num_points < 1 or cholesky.shape[-1] != 3:
            raise ValueError(f"Invalid shape for cholesky: {cholesky.shape}")

        (
            xys,
            radii,
            conics,
            num_tiles_hit,
        ) = _C.project_gaussians_forward_cholesky(
            num_points,
            cholesky,
            means2d,
            img_height,
            img_width,
            block_width,
        )

        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.num_points = num_points

        ctx.save_for_backward(
            radii,
            cholesky,
            conics,
        )

        return (xys, radii, conics, num_tiles_hit)

    @staticmethod
    def backward(
        ctx,
        v_xys,
        v_radii,
        v_conics,
        v_num_tiles_hit
    ):
        (
            radii,
            cholesky,
            conics,
        ) = ctx.saved_tensors

        v_cholesky, v_mean2d = _C.project_gaussians_backward_cholesky(
            ctx.num_points,
            radii,
            cholesky,
            conics,
            v_xys,
            v_conics,
        )
        return (
            v_cholesky,
            v_mean2d,
            None,
            None,
            None,
        )
