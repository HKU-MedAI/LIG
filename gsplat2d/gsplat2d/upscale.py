"""Bicubic spline upscaling using analytical gradients from rasterization"""

from typing import Optional, Tuple
from torch import Tensor
from torch.autograd import Function

import gsplat2d.cuda as _C


class _GradientAwareSplineUpscale(Function):
    @staticmethod
    def forward(
        ctx,
        render: Tensor,  # [H, W, 3]
        dx: Tensor,
        dy: Tensor,
        dxy: Tensor,
        dst_h: int,
        dst_w: int,
        roi: tuple[float, float, float, float],
    ) -> Tensor:
        ctx.save_for_backward(render, dx, dy, dxy)
        ctx.dst_h = dst_h
        ctx.dst_w = dst_w
        ctx.roi = roi
        
        output = _C.gradient_aware_upscale_forward(
            render, dx, dy, dxy, dst_h, dst_w, roi
        )
        return output
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        render, dx, dy, dxy = ctx.saved_tensors
        
        grad_render, grad_dx, grad_dy, grad_dxy = _C.gradient_aware_upscale_backward(
            grad_output.contiguous(),
            render,
            dx,
            dy,
            dxy,
            ctx.dst_h,
            ctx.dst_w,
            ctx.roi,
        )
        
        return grad_render, grad_dx, grad_dy, grad_dxy, None, None, None


def gradient_aware_upscale(
    render: Tensor,  # [H, W, 3]
    dx: Tensor,      # [H, W, 3]
    dy: Tensor,      # [H, W, 3]
    dxy: Tensor,     # [H, W, 3]
    dst_h: int,
    dst_w: int,
    roi: Optional[Tuple[float, float, float, float]] = None,  # (x1, y1, x2, y2)
) -> Tensor:
    """
    Bicubic spline interpolation using analytical gradients.
    
    Args:
        render: Rendered image [H, W, 3]
        dx: Gradient w.r.t. x [H, W, 3]
        dy: Gradient w.r.t. y [H, W, 3]
        dxy: Mixed partial derivative [H, W, 3]
        dst_h: Output height
        dst_w: Output width
        roi: Region of interest (x1, y1, x2, y2), defaults to full image
    
    Returns:
        Upscaled image [dst_h, dst_w, 3]
    """
    h, w, c = render.shape
    
    if roi is None:
        roi = (0.0, 0.0, float(w), float(h))
    
    return _GradientAwareSplineUpscale.apply(
        render.contiguous(), dx.contiguous(), dy.contiguous(), dxy.contiguous(),
        dst_h, dst_w, roi
    )
