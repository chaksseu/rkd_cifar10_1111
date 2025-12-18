#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a VAE (AutoencoderKL) on a Single GPU.
Full Evaluation: MSE, PSNR, SSIM + Latent Stats (Mean, Std, KL).
"""

import os
import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.utils as vutils

import wandb
from diffusers import AutoencoderKL

# Try importing torchmetrics
try:
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    HAS_TORCHMETRICS = True
except ImportError:
    print("[Warning] torchmetrics not installed. SSIM will be skipped.")
    HAS_TORCHMETRICS = False


# ------------------------- Utils -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_grid(images: torch.Tensor, nrow: int = 8) -> Image.Image:
    # images: [-1, 1] -> [0, 255] grid
    imgs = (images.clamp(-1, 1) + 1) / 2.0
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2)
    grid = (grid * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(grid)

def manual_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    return 10. * torch.log10((2.0 ** 2) / mse)


# ------------------------- Dataset -------------------------

class ImageFolderDataset(Dataset):
    def __init__(self, root: str, image_size: int = 32, center_crop: bool = False, horizontal_flip: bool = True):
        self.root = Path(root)
        exts = {".png", ".jpg", ".jpeg"}
        self.files = [p for p in self.root.rglob("*") if p.suffix.lower() in exts]
        if len(self.files) == 0:
            print(f"[Warning] No images found under {self.root}")
        
        tfms = [T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True)]
        if center_crop:
            tfms.append(T.CenterCrop(image_size))
        if horizontal_flip:
            tfms.append(T.RandomHorizontalFlip(p=0.5))
        tfms.append(T.ToTensor()) 
        self.to_tensor = T.Compose(tfms)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            x = self.to_tensor(img)
        return x * 2.0 - 1.0


# ------------------------- Evaluation -------------------------

@torch.no_grad()
def evaluate_reconstruction(
    model: AutoencoderKL,
    loader: DataLoader,
    device: torch.device,
    global_step: int,
    output_dir: Path,
    dataset_name: str,
    mixed_precision: str = "no"
):
    model.eval()
    
    # Accumulators for Reconstruction Metrics
    mse_accum = 0.0
    
    # Accumulators for Latent Stats
    z_mean_accum = 0.0
    z_std_accum = 0.0
    kl_loss_accum = 0.0
    
    if HAS_TORCHMETRICS:
        psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    else:
        psnr_accum = 0.0
        count = 0

    preview_batch = None
    preview_recon = None
    
    total_batches = len(loader)
    if total_batches == 0:
        return

    for i, x in enumerate(loader):
        x = x.to(device)
        
        # AMP Context for Eval
        with torch.amp.autocast('cuda', enabled=(mixed_precision == "fp16")):
            # 1. Encode
            posterior = model.encode(x).latent_dist
            z = posterior.sample()
            
            # 2. Latent Statistics (Test Set)
            z_mean_accum += z.mean().item()
            z_std_accum += z.std().item()
            kl_loss_accum += posterior.kl().mean().item()

            # 3. Decode
            recon = model.decode(z).sample
            recon = recon.clamp(-1, 1)

        # Metrics (Calculate in float32)
        recon_f32 = recon.float()
        x_f32 = x.float()

        # 4. Recon Loss (MSE)
        batch_mse = F.mse_loss(recon_f32, x_f32).item()
        mse_accum += batch_mse

        # 5. PSNR / SSIM
        if HAS_TORCHMETRICS:
            psnr_metric.update(recon_f32, x_f32)
            ssim_metric.update(recon_f32, x_f32)
        else:
            psnr_accum += manual_psnr(recon_f32, x_f32).item()
            count += 1
        
        # Keep first batch for visualization
        if i == 0:
            preview_batch = x_f32
            preview_recon = recon_f32

    # --- Compute Final Averages ---
    final_mse = mse_accum / total_batches
    final_z_mean = z_mean_accum / total_batches
    final_z_std = z_std_accum / total_batches
    final_kl = kl_loss_accum / total_batches
    
    metrics = {}
    
    # PSNR/SSIM Finalize
    if HAS_TORCHMETRICS:
        final_psnr = psnr_metric.compute().item()
        final_ssim = ssim_metric.compute().item()
        psnr_metric.reset()
        ssim_metric.reset()
    else:
        final_psnr = psnr_accum / max(1, count)
        final_ssim = 0.0 # skipped

    # Console Print
    print(f"[Eval-{dataset_name}] Step={global_step} | "
          f"MSE: {final_mse:.5f} | PSNR: {final_psnr:.2f}dB | SSIM: {final_ssim:.4f} | "
          f"z_std: {final_z_std:.3f} | KL: {final_kl:.4f}")

    # Logging Keys
    metrics[f"val/{dataset_name}/mse"] = final_mse
    metrics[f"val/{dataset_name}/psnr"] = final_psnr
    metrics[f"val/{dataset_name}/ssim"] = final_ssim
    
    # Latent Stats Logging (Check encoding quality)
    metrics[f"val/{dataset_name}/z_mean"] = final_z_mean
    metrics[f"val/{dataset_name}/z_std"] = final_z_std
    metrics[f"val/{dataset_name}/kl_loss"] = final_kl

    # Save Preview (Top: Real, Bottom: Recon)
    if preview_batch is not None:
        n_show = min(8, preview_batch.shape[0])
        comparison = torch.cat([preview_batch[:n_show], preview_recon[:n_show]], dim=0)
        grid = to_grid(comparison, nrow=n_show)
        
        save_path = output_dir / "previews" / f"{dataset_name}_step{global_step:06d}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(save_path)
        
        if wandb.run is not None:
            metrics[f"val/{dataset_name}/preview"] = wandb.Image(grid, caption=f"{dataset_name}: Top=Real, Bottom=Recon")

    if wandb.run is not None:
        wandb.log(metrics, step=global_step)
        
    model.train()


# ------------------------- Training -------------------------

def train(args):
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    # --- Data Loaders ---
    train_ds = ImageFolderDataset(args.train_dir, args.image_size, args.center_crop, not args.no_hflip)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    
    test_dl1 = None
    if args.test_dir1:
        test_ds1 = ImageFolderDataset(args.test_dir1, args.image_size, args.center_crop, False)
        if len(test_ds1) > 0:
            test_dl1 = DataLoader(test_ds1, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            print(f"[Info] Test Set 1 found: {len(test_ds1)} images.")

    test_dl2 = None
    if args.test_dir2:
        test_ds2 = ImageFolderDataset(args.test_dir2, args.image_size, args.center_crop, False)
        if len(test_ds2) > 0:
            test_dl2 = DataLoader(test_ds2, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            print(f"[Info] Test Set 2 found: {len(test_ds2)} images.")

    # --- Model Setup ---
    num_blocks = len(args.model_channels)
    down_block_types = ["DownEncoderBlock2D"] * num_blocks
    up_block_types = ["UpDecoderBlock2D"] * num_blocks
    
    print(f"[Info] Model Config: Channels={args.model_channels}, Blocks={num_blocks}")
    
    model = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=args.latent_channels,
        block_out_channels=tuple(args.model_channels),
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        norm_num_groups=32,
        layers_per_block=2,
    )
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.mixed_precision == "fp16"))

    global_step = 0
    print(f"[Info] Start Training... Epochs: {args.epochs}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        
        for step, x in enumerate(train_dl):
            x = x.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(args.mixed_precision == "fp16")):
                posterior = model.encode(x).latent_dist
                z = posterior.sample()
                recon = model.decode(z).sample

                rec_loss = F.l1_loss(recon, x)
                kl_loss = posterior.kl().mean()
                loss = rec_loss + args.kl_weight * kl_loss

            scaler.scale(loss).backward()
            
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            
            # Log Training Metrics
            if global_step % args.log_interval == 0:
                z_std = z.std().item()
                wandb.log({
                    "train/loss": loss.item(),
                    "train/rec_loss": rec_loss.item(),
                    "train/kl_loss": kl_loss.item(),
                    "train/z_std": z_std,
                    "train/epoch": epoch,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }, step=global_step)
                print(f"[Ep{epoch}] Step {global_step} | Loss: {loss.item():.4f} (Rec: {rec_loss.item():.4f}) | z_std: {z_std:.2f}")

            # Validation Loop
            if (args.eval_interval > 0) and (global_step % args.eval_interval == 0):
                if test_dl1:
                    evaluate_reconstruction(model, test_dl1, device, global_step, output_dir, "dataset1", args.mixed_precision)
                if test_dl2:
                    evaluate_reconstruction(model, test_dl2, device, global_step, output_dir, "dataset2", args.mixed_precision)

            # Checkpointing
            if (args.save_interval > 0) and (global_step % args.save_interval == 0):
                save_path = output_dir / f"checkpoint-{global_step}"
                model.save_pretrained(save_path)
                print(f"Saved model to {save_path}")

    # Final Save
    final_path = output_dir / "final_model"
    model.save_pretrained(final_path)
    print(f"Training Done. Saved final model to {final_path}")
    
    # Final Eval
    if test_dl1:
        evaluate_reconstruction(model, test_dl1, device, global_step, output_dir, "dataset1", args.mixed_precision)
    if test_dl2:
        evaluate_reconstruction(model, test_dl2, device, global_step, output_dir, "dataset2", args.mixed_precision)
    
    wandb.finish()


# ------------------------- Arguments -------------------------

DATE=1218
B=64
LR=1e-4
KL_W=1e-7
CUDA_NUM=3

def parse_args():
    parser = argparse.ArgumentParser(description="VAE Training Single GPU (2 Datasets Eval + Latent Stats)")
    parser.add_argument("--device", type=str, default=f"cuda:{CUDA_NUM}")
    
    parser.add_argument("--train_dir", type=str, default="cifar10_png_linear_only/rgb/train")
    parser.add_argument("--test_dir1", type=str, default="cifar10_png_linear_only/rgb/test", help="First test dataset")
    parser.add_argument("--test_dir2", type=str, default="cifar10_png_linear_only/gray3/test", help="Second test dataset (optional)")

    parser.add_argument("--output_dir", type=str, default=f"vae_out_dir/{DATE}_b{B}_lr{LR}_klW_{KL_W}_block_64_128")    
    parser.add_argument("--project", type=str, default=f"vae_training_{DATE}")
    parser.add_argument("--run_name", type=str, default=f"vae_b{B}_lr{LR}_klW_{KL_W}_64_128")
    
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=B)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16"])

    # Model Config
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--latent_channels", type=int, default=4)
    # parser.add_argument("--model_channels", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--model_channels", type=int, nargs="+", default=[64, 128])
    
    parser.add_argument("--kl_weight", type=float, default=KL_W)
    parser.add_argument("--no_hflip", action="store_true")
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=1000)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)