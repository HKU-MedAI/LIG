import torch


"""
Gradient-Aware Bicubic Spline Upscaling for 3D Gaussian Splatting

Reference:
  Niedermayr, S., Neuhauser, C., & Westermann, R. (2025).
  "Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images"
  arXiv:2503.14171v2 [cs.CV]
  https://arxiv.org/abs/2503.14171

OVERVIEW

Bicubic spline interpolation using analytical image gradients from 3DGS rendering,
rather than finite-difference approximations. The key insight is that 3DGS provides
exact gradients ∂I/∂x, ∂I/∂y, ∂²I/∂x∂y at each pixel, enabling more accurate spline fitting.

MATHEMATICAL FORMULATION

BICUBIC SPLINE PARAMETERIZATION (Section G, Eq. 21):

  p(x,y) = Σᵢ₌₀³ Σⱼ₌₀³ aᵢⱼ xⁱ yʲ

The spline coefficients A ∈ ℝ⁴ˣ⁴ are computed by solving (Eq. 25-26):

  F = C · A · Cᵀ
  A = C⁻¹ · F · (Cᵀ)⁻¹

where F is the constraint matrix containing function values and derivatives
at the four corner points of the interpolation cell.

F MATRIX LAYOUT (Eq. 6, 27):

  F = [ f(0,0)    f(0,1)    fᵧ(0,0)   fᵧ(0,1)  ]
      [ f(1,0)    f(1,1)    fᵧ(1,0)   fᵧ(1,1)  ]
      [ fₓ(0,0)   fₓ(0,1)   fₓᵧ(0,0)  fₓᵧ(0,1) ]
      [ fₓ(1,0)   fₓ(1,1)   fₓᵧ(1,0)  fₓᵧ(1,1) ]

where:
  f(i,j)    = pixel value at corner (i,j)
  fₓ(i,j)   = ∂f/∂x at corner (i,j)  -- analytical gradient from 3DGS
  fᵧ(i,j)   = ∂f/∂y at corner (i,j)  -- analytical gradient from 3DGS
  fₓᵧ(i,j)  = ∂²f/∂x∂y at corner (i,j) -- mixed partial from 3DGS

C MATRIX (Eq. 28):

Derived from cubic polynomial constraints at x=0 and x=1:
  f(x)  = a₀ + a₁x + a₂x² + a₃x³
  f'(x) = a₁ + 2a₂x + 3a₃x²

  C = [ 1  0  0  0 ]    (f(0)  = a₀)
      [ 1  1  1  1 ]    (f(1)  = a₀ + a₁ + a₂ + a₃)
      [ 0  1  0  0 ]    (f'(0) = a₁)
      [ 0  1  2  3 ]    (f'(1) = a₁ + 2a₂ + 3a₃)

  C⁻¹ = [  1   0   0   0 ]
        [  0   0   1   0 ]
        [ -3   3  -2  -1 ]
        [  2  -2   1   1 ]

POLYNOMIAL EVALUATION (Eq. 7):

  p(x,y) = [1  x  x²  x³] · A · [1  y  y²  y³]ᵀ

3DGS ANALYTICAL GRADIENTS (Section 5)

The image I(x,y) from 3DGS is computed via alpha blending (Eq. 3):

  I(x,y) = Σᵢ₌₁ᴺ Tᵢ(x,y) · αᵢ(x,y) · cᵢ

Analytical gradients are computed during rendering (Eq. 8-10):

  ∂I/∂x = Σᵢ cᵢ · (∂Tᵢ/∂x · αᵢ + Tᵢ · ∂αᵢ/∂x)

  ∂I/∂y = Σᵢ cᵢ · (∂Tᵢ/∂y · αᵢ + Tᵢ · ∂αᵢ/∂y)

  ∂²I/∂x∂y = Σᵢ cᵢ · (∂²Tᵢ/∂x∂y · αᵢ + ∂Tᵢ/∂x · ∂αᵢ/∂y
                      + ∂Tᵢ/∂y · ∂αᵢ/∂x + Tᵢ · ∂²αᵢ/∂x∂y)

These gradients are computed iteratively during the blending loop with
minimal overhead, then passed to this upscaling kernel.

IMPLEMENTATION NOTES

- Forward: computes A = C⁻¹ · F · (C⁻¹)ᵀ, then evaluates p(tx,ty)
- Memory layout: HWC format (height × width × channels)
"""


@torch.jit.script
def torch_gradient_aware_upscale_single_channel(render: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, dxy: torch.Tensor, new_h: int, new_w: int):
    """
    Gradient aware spline interpolation for a single channel.
    render, dx, dy, dxy: tensors of shape [H, W]
    """
    C = torch.tensor([
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 2, 3]
    ], dtype=torch.float32, device="cuda:0")
    C_inv = torch.inverse(C)

    h, w = render.shape
    device = render.device

    y_coords = torch.linspace(0, h - 1, new_h, device=device)
    x_coords = torch.linspace(0, w - 1, new_w, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

    y0 = torch.floor(grid_y).long()
    x0 = torch.floor(grid_x).long()
    y1 = torch.clamp(y0 + 1, 0, h - 1)
    x1 = torch.clamp(x0 + 1, 0, w - 1)

    ty = (grid_y - y0.float()).unsqueeze(-1)  # [new_h, new_w, 1]
    tx = (grid_x - x0.float()).unsqueeze(-1)  # [new_h, new_w, 1]

    y0 = torch.clamp(y0, 0, h - 1)
    x0 = torch.clamp(x0, 0, w - 1)

    f00 = render[y0, x0]  # [new_h, new_w]
    f01 = render[y0, x1]
    f10 = render[y1, x0]
    f11 = render[y1, x1]

    fx00 = dx[y0, x0]
    fx01 = dx[y0, x1]
    fx10 = dx[y1, x0]
    fx11 = dx[y1, x1]

    fy00 = dy[y0, x0]
    fy01 = dy[y0, x1]
    fy10 = dy[y1, x0]
    fy11 = dy[y1, x1]

    fxy00 = dxy[y0, x0]
    fxy01 = dxy[y0, x1]
    fxy10 = dxy[y1, x0]
    fxy11 = dxy[y1, x1]

    # F matrix: [new_h, new_w, 4, 4]
    F = torch.stack([
        torch.stack([f00, f10, fy00, fy10], dim=-1),
        torch.stack([f01, f11, fy01, fy11], dim=-1),
        torch.stack([fx00, fx10, fxy00, fxy10], dim=-1),
        torch.stack([fx01, fx11, fxy01, fxy11], dim=-1)
    ], dim=-2)

    # A = C_inv @ F @ C_inv.T
    F_reshaped = F.reshape(-1, 4, 4)
    A = torch.matmul(torch.matmul(C_inv, F_reshaped), C_inv.t())
    A = A.reshape(new_h, new_w, 4, 4)

    # p(x,y) = px^T @ A @ py
    ones = torch.ones_like(tx)
    px_squeezed = torch.cat([ones, tx, tx ** 2, tx ** 3], dim=-1)  # [new_h, new_w, 4]
    py_squeezed = torch.cat([ones, ty, ty ** 2, ty ** 3], dim=-1)  # [new_h, new_w, 4]

    # Perform batched matrix multiplication
    temp = torch.einsum('hwi,hwij->hwj', px_squeezed, A)  # [new_h, new_w, 4]
    upscaled = torch.einsum('hwi,hwi->hw', temp, py_squeezed)  # [new_h, new_w]

    return upscaled


def torch_gradient_aware_upscale(render_hwc: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, dxy: torch.Tensor, new_h: int, new_w: int) -> torch.Tensor:
    """
    Bicubic spline interpolation for multi-channel image.
    render_hwc, dx, dy, dxy: tensors of shape [H, W, C]
    Returns: [new_h, new_w, C]
    """
    c = render_hwc.shape[2]
    upscaled_channels = []
    for ch in range(c):
        upscaled_ch = torch_gradient_aware_upscale_single_channel(
            render_hwc[..., ch], dx[..., ch], dy[..., ch], dxy[..., ch],
            new_h, new_w
        )
        upscaled_channels.append(upscaled_ch)
    
    return torch.stack(upscaled_channels, dim=-1)
