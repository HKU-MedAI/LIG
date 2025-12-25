"""Python bindings for custom Cuda functions"""

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

import gsplat2d.cuda as _C

from .utils import bin_and_sort_gaussians, compute_cumulative_intersects

def rasterize_gaussians(
    xys: Float[Tensor, "*batch 2"],
    radii: Float[Tensor, "*batch 1"],
    conics: Float[Tensor, "*batch 3"],
    num_tiles_hit: Int[Tensor, "*batch 1"],
    colors: Float[Tensor, "*batch channels"],
    img_height: int,
    img_width: int,
    block_width: int,
) -> tuple[Tensor, Tensor]:
    
    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"
    if colors.dtype == torch.uint8:
        colors = colors.float() / 255

    if xys.ndimension() != 2 or xys.size(1) != 2:
        raise ValueError("xys must have dimensions (N, 2)")

    if colors.ndimension() != 2:
        raise ValueError("colors must have dimensions (N, D)")

    return _RasterizeGaussians.apply(
        xys.contiguous(),
        radii.contiguous(),
        conics.contiguous(),
        num_tiles_hit.contiguous(),
        colors.contiguous(),
        img_height,
        img_width,
        block_width,
    )


class _RasterizeGaussians(Function):
    """Rasterizes 2D gaussians"""

    @staticmethod
    def forward(
        ctx,
        xys: Float[Tensor, "*batch 2"],
        radii: Float[Tensor, "*batch 1"],
        conics: Float[Tensor, "*batch 3"],
        num_tiles_hit: Int[Tensor, "*batch 1"],
        colors: Float[Tensor, "*batch channels"],
        img_height: int,
        img_width: int,
        block_width: int,
    ) -> tuple[Tensor, Tensor]:
        num_points = xys.size(0)
        tile_bounds = (
            (img_width + block_width - 1) // block_width,
            (img_height + block_width - 1) // block_width,
            1,
        )
        block = (block_width, block_width, 1)
        img_size = (img_width, img_height, 1)

        depths = torch.zeros_like(xys[..., 0], device=xys.device)

        num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)

        if num_intersects < 1:
            out_img = (
                torch.ones(img_height, img_width, colors.shape[-1], device=xys.device)
            )
            out_wsum = torch.zeros(img_height, img_width, device=xys.device)
            gaussian_ids_sorted = torch.zeros(0, 1, device=xys.device)
            tile_bins = torch.zeros(0, 2, device=xys.device)
            final_idx = torch.zeros(img_height, img_width, device=xys.device)
        else:
            (
                isect_ids_unsorted,
                gaussian_ids_unsorted,
                isect_ids_sorted,
                gaussian_ids_sorted,
                tile_bins,
            ) = bin_and_sort_gaussians(
                num_points,
                num_intersects,
                xys,
                depths,
                radii,
                cum_tiles_hit,
                tile_bounds,
                block_width,
            )
            rasterize_fn = _C.rasterize_forward
            
            out_img, out_wsum, final_idx = rasterize_fn(
                tile_bounds,
                block,
                img_size,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
            )

        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.num_intersects = num_intersects
        ctx.block_width = block_width
        ctx.save_for_backward(
            gaussian_ids_sorted,
            tile_bins,
            xys,
            conics,
            colors,
            final_idx,
        )

        return out_img, out_wsum

    @staticmethod
    def backward(ctx, v_out_img, v_out_wsum):
        img_height = ctx.img_height
        img_width = ctx.img_width
        num_intersects = ctx.num_intersects

        (
            gaussian_ids_sorted,
            tile_bins,
            xys,
            conics,
            colors,
            final_idx,
        ) = ctx.saved_tensors

        if num_intersects < 1:
            v_xy = torch.zeros_like(xys)
            v_xy_abs = torch.zeros_like(xys)
            v_conic = torch.zeros_like(conics)
            v_colors = torch.zeros_like(colors)

        else:
            rasterize_fn = _C.rasterize_backward

            v_xy, v_xy_abs, v_conic, v_colors = rasterize_fn(
                img_height,
                img_width,
                ctx.block_width,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
                final_idx,
                v_out_img,
                v_out_wsum,
            )

        xys.absgrad = v_xy_abs

        return (
            v_xy,  # xys
            None,  # radii
            v_conic,  # conics
            None,  # num_tiles_hit
            v_colors,  # colors
            None,  # img_height
            None,  # img_width
            None,  # block_width
        )
