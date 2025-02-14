import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms

class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image_path: Path,
        log_path: str,
        num_points: int = 2000,
        model_name:str = "LIG",
        iterations:int = 30000,
        model_path = None,
        args = None,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = image_path_to_tensor(image_path).to(self.device)

        self.num_points = num_points
        image_path = Path(image_path)
        self.image_name = image_path.stem

        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        self.log_dir = Path(log_path + '/' + self.image_name)

        if model_name == "LIG":
            from gaussianlig import LIG
            self.gaussian_model = LIG(loss_type="L2", opt_type="adam", 
                                      num_points=self.num_points, n_scales=args.n_scales, allo_ratio=args.allo_ratio,
                                      H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
                                      device=self.device, lr=args.lr).to(self.device)

        self.logwriter = LogWriter(self.log_dir)

        if model_path is not None:
            print(f"loading model path:{model_path}")
            if not hasattr(self.gaussian_model, 'n_scales'):
                checkpoint = torch.load(model_path, map_location=self.device)
                model_dict = self.gaussian_model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.gaussian_model.load_state_dict(model_dict)
            else:
                checkpoint = torch.load(model_path, map_location=self.device)
                for level in range(self.gaussian_model.n_scales):
                    model_dict = self.gaussian_model.level_models[level].state_dict()
                    pretrained_dict = {k: v for k, v in checkpoint['state_dict'][level].items() if k in model_dict}
                    model_dict.update(pretrained_dict)
                    self.gaussian_model.level_models[level].load_state_dict(model_dict)
                self.gaussian_model.store_max = checkpoint['store_max']
                self.gaussian_model.store_min = checkpoint['store_min']

    def train(self):     
        psnr_list, iter_list = [], []
        if not hasattr(self.gaussian_model, 'n_scales'):
            progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
            self.gaussian_model.train()
            start_time = time.time()
            for iter in range(1, self.iterations+1):
                loss, psnr = self.gaussian_model.train_iter(self.gt_image)
                psnr_list.append(psnr)
                iter_list.append(iter)
                with torch.no_grad():
                    if iter % 10 == 0:
                        progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                        progress_bar.update(10)
            end_time = time.time() - start_time
            progress_bar.close()
            psnr_value, ms_ssim_value = self.test()
            with torch.no_grad():
                self.gaussian_model.eval()
                test_start_time = time.time()
                for i in range(100):
                    _ = self.gaussian_model()
                test_end_time = (time.time() - test_start_time)/100
        else:
            start_time = time.time()
            for scale_idx in range(self.gaussian_model.n_scales):
                if scale_idx != self.gaussian_model.n_scales - 1 and self.gaussian_model.n_scales > 1:
                    img_target = torch.nn.functional.interpolate(self.gt_image,
                                            scale_factor=pow(2.0, -self.gaussian_model.n_scales+scale_idx+1),
                                            mode='area')
                else:
                    img_target = self.gt_image
                
                if scale_idx != 0:
                    im_estim_prev = torch.nn.functional.interpolate(im_estim,
                                                                    size = (img_target.shape[2], img_target.shape[3]),
                                                                    mode='bilinear')
                    del im_estim
                    if self.save_imgs:
                        transform = transforms.ToPILImage()
                        im_estim_prev_img = transform(torch.clamp(im_estim_prev, 0, 1).squeeze(0))
                        name = self.image_name + f"_fitting_{scale_idx-1}.png" 
                        im_estim_prev_img.save(str(self.log_dir / name))

                    img_target = img_target - im_estim_prev
                    im_estim_prev = im_estim_prev.cpu()
                    img_target += 0.5

                    if self.save_imgs:
                        transform = transforms.ToPILImage()
                        img_target_img = transform(torch.clamp(img_target, 0, 1).squeeze(0))
                        name = self.image_name + f"_residual_{scale_idx-1}.png" 
                        img_target_img.save(str(self.log_dir / name))
                    

                    store_min = torch.min(img_target)
                    store_max = torch.max(img_target)
                    img_target = (img_target - store_min) / (store_max - store_min)

                    if self.save_imgs:
                        transform = transforms.ToPILImage()
                        img_target_img = transform(torch.clamp(img_target, 0, 1).squeeze(0))
                        name = self.image_name + f"_residual_scale_{scale_idx-1}.png" 
                        img_target_img.save(str(self.log_dir / name))

                    self.gaussian_model.store_min.append(store_min)
                    self.gaussian_model.store_max.append(store_max)

                progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
                self.gaussian_model.level_models[scale_idx].train()
                for iter in range(1, self.iterations+1):
                    # affect memory and speed
                    torch.cuda.empty_cache()
                    loss, psnr = self.gaussian_model.level_models[scale_idx].train_iter(img_target)
                    psnr_list.append(psnr)
                    iter_list.append(iter)
                    with torch.no_grad():
                        if iter % 10 == 0:
                            progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                            progress_bar.update(10)
                
                with torch.no_grad():
                    if scale_idx == 0:
                        im_estim = self.gaussian_model.level_models[scale_idx]()["render"].float()
                    else:
                        im_estim = self.gaussian_model.level_models[scale_idx]()["render"].float()*(store_max-store_min) + im_estim_prev.to(self.device) - 0.5 + store_min 

                    im_estim = im_estim.detach()
                    self.gaussian_model.level_models[scale_idx] = self.gaussian_model.level_models[scale_idx].to("cpu")

            end_time = time.time() - start_time
            progress_bar.close()
            psnr_value, ms_ssim_value = self.test()

            with torch.no_grad():
                self.gaussian_model.eval()
                test_start_time = time.time()
                for i in range(100):
                    for scale_idx in range(self.gaussian_model.n_scales):
                        _ = self.gaussian_model.level_models[scale_idx]()
                test_end_time = (time.time() - test_start_time)/100
 
        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, 1/test_end_time))
        if not hasattr(self.gaussian_model, 'n_scales'):
            torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        else:
            torch.save({'state_dict':[self.gaussian_model.level_models[scale_idx].state_dict() for scale_idx in range(self.gaussian_model.n_scales)], 
                        'store_max': self.gaussian_model.store_max, 'store_min': self.gaussian_model.store_min}, self.log_dir / "gaussian_model.pth.tar")
        np.save(self.log_dir / "training.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
        "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time})
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time

    def test(self):
        if not hasattr(self.gaussian_model, 'n_scales'):
            self.gaussian_model.eval()
            with torch.no_grad():
                out = self.gaussian_model()["render"].float()
        else:
            for scale_idx in range(self.gaussian_model.n_scales):
                self.gaussian_model.level_models[scale_idx].to(self.device)
                self.gaussian_model.level_models[scale_idx].eval()
                with torch.no_grad():
                    if scale_idx == 0:
                        out = self.gaussian_model.level_models[scale_idx]()["render"].float()
                    else:
                        next_estim = self.gaussian_model.level_models[scale_idx]()["render"].float()*(self.gaussian_model.store_max[scale_idx-1]-self.gaussian_model.store_min[scale_idx-1]) - 0.5 + self.gaussian_model.store_min[scale_idx-1]
                        out = torch.nn.functional.interpolate(out, size = (next_estim.shape[2], next_estim.shape[3]), mode='bilinear')
                        out = out + next_estim
        out = torch.clamp(out, 0, 1)
        mse_loss = F.mse_loss(out, self.gt_image.float())
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out, self.gt_image.float(), data_range=1, size_average=True).item()
        self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))
        if self.save_imgs:
            transform = transforms.ToPILImage()
            img = transform(out.squeeze(0))
            name = self.image_name + "_fitting.png" 
            img.save(str(self.log_dir / name))
        return psnr, ms_ssim_value

def image_path_to_tensor(image_path: Path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./dataset/DIV2K_valid_HR', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='DIV2K_valid_HR', help="Training dataset"
    )
    parser.add_argument(
        "--iterations", type=int, default=30000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="LIG", help="model selection: GaussianImage_Cholesky, GaussianImage_RS, 3DGS"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=500000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument("--n_scales", type=int, default=2)
    parser.add_argument("--allo_ratio", type=float, default=0.5)
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.018,
        help="Learning rate (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    if args.n_scales == 1:
        log_path = f"./checkpoints/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}_{args.n_scales}"
    elif args.n_scales > 1:
        log_path = f"./checkpoints/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}_{args.n_scales}_{args.allo_ratio}"
    
    logwriter = LogWriter(log_path)
    psnrs, ms_ssims, training_times, eval_times, eval_fpses = [], [], [], [], []
    image_h, image_w = 0, 0
    if args.data_name == "kodak":
        image_length, start = 24, 0
    elif args.data_name == "DIV2K_valid_LRX2" or args.data_name == "DIV2K_valid_HR":
        image_length, start = 100, 800
    elif args.data_name == "STimage":
        image_length, start = 15, 0
    elif args.data_name == "GF1":
        image_length, start = 4, 0
    for i in range(start, start+image_length):
        if args.data_name == "kodak":
            image_path = Path(args.dataset) / f'kodim{i+1:02}.png'
        elif args.data_name == "DIV2K_valid_LRX2":
            image_path = Path(args.dataset) /  f'{i+1:04}x2.png'
        elif args.data_name == "DIV2K_valid_HR":
            image_path = Path(args.dataset) /  f'{i+1:04}.png'
        elif args.data_name == "STimage":
            image_path = Path(args.dataset) / f'Human_Heart_{i}.png'
        elif args.data_name == "GF1":
            image_path = Path(args.dataset) / f'GF1_{i}.png'

        torch.cuda.empty_cache()
        trainer = SimpleTrainer2d(image_path=image_path, log_path=log_path, num_points=args.num_points, 
            iterations=args.iterations, model_name=args.model_name, args=args, model_path=args.model_path)
        psnr, ms_ssim, training_time, eval_time, eval_fps = trainer.train()
        psnrs.append(psnr)
        ms_ssims.append(ms_ssim)
        training_times.append(training_time) 
        eval_times.append(eval_time)
        eval_fpses.append(eval_fps)
        image_h += trainer.H
        image_w += trainer.W
        image_name = image_path.stem
        logwriter.write("{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
            image_name, trainer.H, trainer.W, psnr, ms_ssim, training_time, eval_time, eval_fps))

    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_time = torch.tensor(eval_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    avg_h = image_h//image_length
    avg_w = image_w//image_length

    logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_training_time, avg_eval_time, avg_eval_fps))    

if __name__ == "__main__":
    main(sys.argv[1:])
