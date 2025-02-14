from gsplat2d.project_gaussians import project_gaussians
from gsplat2d.rasterize import rasterize_gaussians
from utils import *
import torch
import torch.nn as nn
import math
from optimizer import Adan

class LIG(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]

        self.n_scales = kwargs["n_scales"]
        self.allo_ratio = kwargs["allo_ratio"]
        self.level_models = []

        self.store_min = []
        self.store_max = []

        for s in range(self.n_scales):

            H = int(self.H * pow(2.0, -self.n_scales + s + 1))
            W = int(self.W * pow(2.0, -self.n_scales + s + 1))

            if s != self.n_scales - 1:
                num_points = int(kwargs["num_points"] * pow(2.0, (-self.n_scales + s + 1)*2) * self.allo_ratio)
            else:
                num_points = kwargs["num_points"]
                for i in range(s):
                    num_points -= int(kwargs["num_points"] * pow(2.0, (-self.n_scales + i + 1)*2) * self.allo_ratio)

            self.level_models.append(Gaussian2D(loss_type="L2", opt_type=kwargs['opt_type'], num_points=num_points, 
                                                   H=H, W=W, BLOCK_H=kwargs['BLOCK_H'], BLOCK_W=kwargs['BLOCK_W'],
                                                   device=kwargs['device'], lr=kwargs['lr']))

class Gaussian2D(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]

        self.B_SIZE = 16

        self.device = kwargs["device"]

        self.last_size = (self.H, self.W)

        w_init = torch.rand(self.init_num_points, 1, device=self.device) * self.W
        h_init = torch.rand(self.init_num_points, 1, device=self.device) * self.H
        self.means = nn.Parameter(torch.cat((w_init, h_init), dim=1))

        self.cov2d = nn.Parameter(torch.rand(self.init_num_points, 3, device=self.device))
        d = 3
        self.rgbs = nn.Parameter(torch.zeros(self.init_num_points, d, device=self.device))

        self.means.requires_grad = True
        self.cov2d.requires_grad = True
        self.rgbs.requires_grad = True

        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam([self.rgbs, self.means, self.cov2d], lr=kwargs["lr"])
        else:
            self.optimizer = Adan([self.rgbs, self.means, self.cov2d], lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=70000, gamma=0.7)

    def forward(self):
        (
            xys,
            radii,
            conics,
            num_tiles_hit,
        ) = project_gaussians(
            self.cov2d,
            self.means,
            self.H,
            self.W,
            self.B_SIZE,
        )
        out_img = rasterize_gaussians(
                xys,
                radii,
                conics,
                num_tiles_hit,
                self.rgbs,
                self.H,
                self.W,
                self.B_SIZE,
            )[..., :3]

        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def train_iter(self, gt_image):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)

        self.scheduler.step()
        return loss, psnr
