"""Python bindings for 2D gaussian projection"""

from typing import Optional, Tuple
import torch

from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import gsplat2d.cuda as _C


def project_gaussians(
    cov2d: Float[Tensor, "*batch 3"],
    means2d: Float[Tensor, "*batch 2"],
    img_height: int,
    img_width: int,
    block_width: int,
    opacities: Optional[Float[Tensor, "*batch"]] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"
    return _ProjectGaussians.apply(
        cov2d.contiguous(),
        means2d.contiguous(),
        opacities.contiguous() if opacities is not None else None,
        img_height,
        img_width,
        block_width,
    )


class _ProjectGaussians(Function):

    @staticmethod
    def forward(
        ctx,
        cov2d: Float[Tensor, "*batch 3"],
        means2d: Float[Tensor, "*batch 2"],
        opacities: Optional[Float[Tensor, "*batch"]],
        img_height: int,
        img_width: int,
        block_width: int,
    ):
        num_points = cov2d.shape[-2]
        if num_points < 1 or cov2d.shape[-1] != 3:
            raise ValueError(f"Invalid shape for cov2d: {cov2d.shape}")

        (
            xys,
            extents,
            conics,
            num_tiles_hit,
        ) = _C.project_gaussians_forward(
            num_points,
            cov2d,
            means2d,
            opacities,
            img_height,
            img_width,
            block_width,
        )

        # Save non-tensors.
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.num_points = num_points

        # Save tensors.
        ctx.save_for_backward(
            extents,
            conics,
        )

        return (xys, extents, conics, num_tiles_hit)

    @staticmethod
    def backward(
        ctx,
        v_xys,
        v_extents,
        v_conics,
        v_num_tiles_hit
    ):
        (   
            extents,
            conics,
        ) = ctx.saved_tensors


        v_cov2d, v_mean2d = _C.project_gaussians_backward(
            ctx.num_points,
            extents,
            conics,
            v_xys,
            v_conics,
        )
        return (
            v_cov2d,
            v_mean2d,
            None,
            None,
            None,
            None,
        )
