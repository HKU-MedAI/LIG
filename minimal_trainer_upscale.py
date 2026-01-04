"""Minimal trainer for 2D Gaussian splatting with optional bicubic upscale"""

# =============================================================================
# RESOLUTION LOGIC:
# 
# 1. Load image, optionally downscale to MAX_TRAIN_RESOLUTION -> this is gt_full
# 2. gt_full is ALWAYS the target for loss comparison (H_full x W_full)
#
# WITHOUT upscale (USE_UPSCALE=False):
#   - Model trains at gt_full resolution (H_train = H_full)
#   - Loss: render vs gt_full directly
#
# WITH upscale (USE_UPSCALE=True):
#   - Model trains at MODEL_RESOLUTION (max dimension)
#   - Render upscaled via bicubic_spline_upscale before comparison
#   - Loss: upscaled_render vs gt_full
# =============================================================================

import math
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from optimizer import Adan
from gsplat2d.rasterize import rasterize_gaussians
from gsplat2d.project_gaussians_cholesky import project_gaussians_cholesky
from gsplat2d.upscale import gradient_aware_upscale
from upscaler_torch import torch_gradient_aware_upscale


# ============ CONFIG ============
IMAGE_PATH = r"x:\_ai\_gsplat\datasets\DIV2K_valid_HR\0900.png"
NUM_POINTS = 500_000
ITERATIONS = 10000
LR = 0.015
USE_UPSCALE = True
USE_TORCH_UPSCALE = False  # True = torch impl, False = gsplat2d CUDA impl
MODEL_RESOLUTION = 1200  # train at this max dimension, upscale to full
MAX_TRAIN_RESOLUTION = None # if set, downscale gt to fit this max dimension before training
DEVICE = torch.device("cuda:0")
# ================================


def init_splats_simple(
    num_points: int, H: int, W: int, device,
    mode: str = 'grid',
    image: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode == 'grid':
        aspect = W / H
        ny = max(1, int(math.sqrt(num_points / aspect)))
        nx = max(1, int(num_points / ny))
        ys = torch.linspace(0.5, H - 0.5, ny, device=device)
        xs = torch.linspace(0.5, W - 0.5, nx, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        num_points = positions.shape[0]
    else:
        positions = torch.rand(num_points, 2, device=device) * torch.tensor([W, H], device=device)

    scales = torch.full((num_points,), math.sqrt(H * W / num_points) * 0.5, device=device)

    if image is not None:
        img = image.squeeze(0) if image.dim() == 4 else image
        x_coords = positions[:, 0].long().clamp(0, W - 1)
        y_coords = positions[:, 1].long().clamp(0, H - 1)
        rgbs = img[:, y_coords, x_coords].T
    else:
        rgbs = torch.full((num_points, 3), 0.5, device=device)

    return positions.contiguous(), rgbs.contiguous(), scales.contiguous()

# ================================


class Gaussian2DMinimal(nn.Module):
    def __init__(self, num_points: int, H: int, W: int, device,
                 positions: torch.Tensor | None = None,
                 rgbs: torch.Tensor | None = None,
                 scales: torch.Tensor | None = None):
        super().__init__()
        self.H = H
        self.W = W
        self.device = device
        self.B_SIZE = 16

        if positions is None:
            positions = torch.rand(num_points, 2, device=device) * torch.tensor([W, H], device=device)
        if rgbs is None:
            rgbs = torch.full((num_points, 3), 0.5, dtype=torch.float32, device=device)
        if scales is None:
            scales = torch.full((num_points,), math.sqrt(H * W / num_points) * 0.5, dtype=torch.float32, device=device)

        self.means = nn.Parameter(positions)
        self._rgb_logits = nn.Parameter(torch.logit(rgbs.clamp(0.001, 0.999)))
        self._opacity_logits = nn.Parameter(torch.zeros(num_points, device=device))

        # Circular splats: L11=scale, L21=0, L22=scale
        # _cholesky stores pre-softplus values, need inverse softplus
        diag = torch.log(torch.expm1(scales))
        x_L21 = torch.zeros(num_points, device=device)
        self._cholesky = nn.Parameter(torch.stack([diag, x_L21, diag], dim=1))

    @property
    def cholesky(self):
        L11 = F.softplus(self._cholesky[:, 0])
        L21 = self._cholesky[:, 1]  # no softplus
        L22 = F.softplus(self._cholesky[:, 2])
        return torch.stack([L11, L21, L22], dim=1)

    @property
    def rgbs(self):
        return torch.sigmoid(self._rgb_logits)

    @property
    def opacities(self):
        return torch.sigmoid(self._opacity_logits)

    def forward(self):
        xys, extents, conics, num_tiles_hit = project_gaussians_cholesky(
            self.cholesky, self.means, self.H, self.W, self.B_SIZE, self.opacities
        )
        out_img, out_wsum, dx, dy, dxy = rasterize_gaussians(
            xys, extents, conics, num_tiles_hit,
            self.rgbs, self.opacities,
            self.H, self.W, self.B_SIZE,
            compute_upscale_gradients=True
        )
        return {
            "render": out_img.permute(2, 0, 1),
            "render_hwc": out_img, "wsum": out_wsum, "dx": dx, "dy": dy, "dxy": dxy,
        }

    def scale_to(self, new_H: int, new_W: int):
        """Scale parameters to new resolution"""
        scale_x = new_W / self.W
        scale_y = new_H / self.H

        with torch.no_grad():
            self.means.data[:, 0] *= scale_x
            self.means.data[:, 1] *= scale_y

            # L11, L22: softplus -> scale -> inverse softplus
            L11 = F.softplus(self._cholesky.data[:, 0])
            L22 = F.softplus(self._cholesky.data[:, 2])
            L11_new = L11 * scale_x
            L22_new = L22 * scale_y
            self._cholesky.data[:, 0] = torch.log(torch.clamp(torch.exp(L11_new) - 1, min=1e-8))
            self._cholesky.data[:, 2] = torch.log(torch.clamp(torch.exp(L22_new) - 1, min=1e-8))
            # L21: no softplus, just scale
            self._cholesky.data[:, 1] *= scale_y

        self.H = new_H
        self.W = new_W


def upscale_fn(out, target_h, target_w, use_torch_upscale):
    render_hwc = out["render_hwc"]
    dx_hwc = out["dx"]
    dy_hwc = out["dy"]
    dxy_hwc = out["dxy"]
    if use_torch_upscale:
        return torch_gradient_aware_upscale(render_hwc, dx_hwc, dy_hwc, dxy_hwc, target_h, target_w)
    else:
        return gradient_aware_upscale(render_hwc, dx_hwc, dy_hwc, dxy_hwc, target_h, target_w)

def train(image_path: str, num_points: int, iterations: int, lr: float,
          use_upscale: bool, model_resolution: int, device,
          max_train_resolution: int | None = None,
          use_torch_upscale: bool = False):

    img = Image.open(image_path)
    gt_image = transforms.ToTensor()(img)[:3, ...].unsqueeze(0).to(device)  # [1, C, H, W]
    del img

    H_gt, W_gt = gt_image.shape[2], gt_image.shape[3]

    # Optional downscale to max_train_resolution
    if max_train_resolution is not None:
        max_dim = max(H_gt, W_gt)
        if max_dim > max_train_resolution:
            scale = max_train_resolution / max_dim
            H_gt, W_gt = round(H_gt * scale), round(W_gt * scale)
            gt_image = F.interpolate(gt_image, size=(H_gt, W_gt), mode='area')

    if use_upscale:
        scale = model_resolution / max(H_gt, W_gt)
        H_model = round(H_gt * scale)
        W_model = round(W_gt * scale)
    else:
        H_model, W_model = H_gt, W_gt

    gt_for_init = gt_image if gt_image.shape[2:] == (H_model, W_model) else F.interpolate(gt_image, size=(H_model, W_model), mode='area')

    positions, rgbs, scales = init_splats_simple(num_points, H_model, W_model, device, mode="grid", image=gt_for_init)
    actual_num_points = positions.shape[0]
    model = Gaussian2DMinimal(actual_num_points, H_model, W_model, device, positions=positions, rgbs=rgbs, scales=scales).to(device)

    pos_lr_scale = max(H_model, W_model) / 512.0
    optimizer = Adan([
        {'params': model._rgb_logits, 'lr': lr * 5},
        {'params': model.means, 'lr': lr * 2 * pos_lr_scale},
        {'params': model._cholesky, 'lr': lr * 1 * pos_lr_scale},
        {'params': model._opacity_logits, 'lr': lr}
    ], betas=(0.98, 0.92, 0.99), fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=iterations, T_mult=1, eta_min=lr * 0.01
    )

    gt_hwc = gt_image.squeeze(0).permute(1, 2, 0)  # [H_train, W_train, C]
    gt_scale = max(H_gt, W_gt) / max(H_model, W_model) if use_upscale else 1.0

    start = time.monotonic()
    psnr = 0.0
    pbar = tqdm(range(1, iterations + 1), desc="Training")
    for it in pbar:
        out = model()
        compare_img = out["render_hwc"] if not use_upscale else upscale_fn(out, H_gt, W_gt, use_torch_upscale)
        loss = F.mse_loss(compare_img, gt_hwc)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if it % 100 == 0:
            with torch.no_grad():
                mse = F.mse_loss(compare_img, gt_hwc)
                psnr = 10 * math.log10(1.0 / mse.item())
                chol = model.cholesky
                splat_sizes = torch.sqrt((chol[:, 0] * chol[:, 2]))  # sqrt(L11 * L22) ~ geom mean of axes
                p1, p50, p99 = torch.quantile(splat_sizes, torch.tensor([0.01, 0.5, 0.99], device=device))
                pbar.set_postfix(loss=f"{loss.item():.5f}", psnr=f"{psnr:.2f}", 
                                 sz=f"min:{p1.item() * gt_scale:.2f}px, mean:{p50.item() * gt_scale:.2f}px, max:{p99.item() * gt_scale:.2f}px")

    elapsed = time.monotonic() - start
    print(f"Training done in {elapsed:.2f}s")

    # Final eval
    model.eval()
    with torch.no_grad():
        out = model()
        if use_upscale:
            render_native = torch.clamp(out["render_hwc"], 0, 1)
            render_upscaled = torch.clamp(upscale_fn(out, H_gt, W_gt, use_torch_upscale), 0, 1)
            mse = F.mse_loss(render_upscaled, gt_hwc)
        else:
            render_native = torch.clamp(out["render_hwc"], 0, 1)
            mse = F.mse_loss(render_native, gt_hwc)
        psnr = 10 * math.log10(1.0 / mse.item())
        print(f"Final PSNR: {psnr:.4f}")

        n_suffix = f"{actual_num_points / 1_000_000:.0f}M" if actual_num_points >= 2_500_000 else f"{actual_num_points / 1000:.0f}k"
        transforms.ToPILImage()(render_native.permute(2, 0, 1)).save(f"result_native_{n_suffix}.png")
        if use_upscale:
            transforms.ToPILImage()(render_upscaled.permute(2, 0, 1)).save(f"result_upscaled_{n_suffix}.png")


if __name__ == "__main__":
    train(
        image_path=IMAGE_PATH,
        num_points=NUM_POINTS,
        iterations=ITERATIONS,
        lr=LR,
        use_upscale=USE_UPSCALE,
        model_resolution=MODEL_RESOLUTION,
        device=DEVICE,
        use_torch_upscale=USE_TORCH_UPSCALE,
        max_train_resolution=MAX_TRAIN_RESOLUTION,
    )