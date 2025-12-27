#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a Latent Diffusion Model (LDM) on a Single GPU.
Optimized for 16x16 Latent Space (from 32x32 Image / f=2 VAE).
"""

import os
import argparse
import shutil
from pathlib import Path
from typing import List, Optional
import random

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.utils as vutils

import wandb
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, AutoencoderKL

# ------------------------- Utils -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())

def to_grid(images: torch.Tensor, nrow: int = 8) -> Image.Image:
    # images: [-1, 1] -> [0, 255] grid
    imgs = (images.clamp(-1, 1) + 1) / 2.0
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2)
    grid = (grid * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(grid)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def collect_image_paths_recursive(root: Path, exts={".png", ".jpg", ".jpeg"}) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]

def flatten_real_cache(test_dir: Path, cache_dir: Path) -> int:
    """Flattens images for FID calculation."""
    ensure_dir(cache_dir)
    existing = list(cache_dir.glob("*"))
    if len(existing) > 0:
        return len(existing)

    paths = collect_image_paths_recursive(test_dir)
    print(f"[FID] Flattening test set ({len(paths)} imgs) to {cache_dir} ...")
    for i, src in enumerate(paths, 1):
        dst = cache_dir / f"real_{i:06d}{src.suffix.lower()}"
        try:
            if hasattr(os, 'symlink'):
                os.symlink(src.resolve(), dst)
            else:
                shutil.copy2(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    return len(paths)

def save_tensor_batch_to_dir(x: torch.Tensor, out_dir: Path, start_idx: int):
    ensure_dir(out_dir)
    x01 = (x.clamp(-1, 1) + 1) / 2.0
    x255 = (x01 * 255.0).clamp(0, 255).byte().cpu()
    for i in range(x255.shape[0]):
        arr = x255[i].permute(1, 2, 0).numpy()
        img = Image.fromarray(arr)
        img.save(out_dir / f"gen_{start_idx + i:06d}.png")


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


# ------------------------- Sampling (Latent DDIM) -------------------------

@torch.no_grad()
def sample_ldm_ddim(
    unet: UNet2DModel,
    vae: AutoencoderKL,
    ddim_scheduler: DDIMScheduler,
    num_images: int,
    latent_shape: tuple,  # (C, H, W)
    scale_factor: float,
    device: torch.device,
    steps: int = 50,
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    unet.eval()
    vae.eval()
    
    ddim_scheduler.set_timesteps(steps, device=device)
    
    # Start with random latent noise
    z = torch.randn((num_images, *latent_shape), device=device, generator=generator)

    for t in ddim_scheduler.timesteps:
        noise_pred = unet(z, t).sample
        z = ddim_scheduler.step(noise_pred, t, z, eta=eta, generator=generator).prev_sample

    # Rescale back and Decode
    z = z / scale_factor
    images = vae.decode(z).sample

    unet.train()
    return images


# ------------------------- FID -------------------------

def compute_fid(
    unet: UNet2DModel,
    vae: AutoencoderKL,
    ddim_scheduler: DDIMScheduler,
    args,
    device: torch.device,
    global_step: int,
    real_cache_dir: Path,
    num_test_imgs: int,
    latent_shape: tuple,
):
    if args.disable_fid:
        return

    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except ImportError:
        print("[FID] pytorch-fid not installed. Skipping.")
        return

    gen_dir = Path(args.output_dir) / "fid_gen" / f"step{global_step:06d}"
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    ensure_dir(gen_dir)

    print(f"[FID] Generating {num_test_imgs} samples to {gen_dir} ...")
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + global_step)

    remaining = num_test_imgs
    cursor = 0
    
    while remaining > 0:
        cur = min(args.fid_gen_batch, remaining)
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(args.mixed_precision == "fp16")):
                imgs = sample_ldm_ddim(
                    unet, vae, ddim_scheduler,
                    num_images=cur,
                    latent_shape=latent_shape,
                    scale_factor=args.latent_scale_factor,
                    device=device,
                    steps=args.sample_steps,
                    eta=args.sample_eta,
                    generator=gen
                )
        save_tensor_batch_to_dir(imgs, gen_dir, start_idx=cursor)
        cursor += cur
        remaining -= cur

    print(f"[FID] Computing FID against {real_cache_dir} ...")
    try:
        fid_score = calculate_fid_given_paths(
            [real_cache_dir.as_posix(), gen_dir.as_posix()],
            batch_size=args.fid_batch_size,
            device=device,
            dims=args.fid_dims,
        )
        print(f"[FID] Step={global_step} FID={fid_score:.4f}")
        
        if wandb.run is not None:
            wandb.log({"metrics/fid": fid_score}, step=global_step)
    except Exception as e:
        print(f"[FID Error] {e}")

    if not args.fid_keep_gen:
        shutil.rmtree(gen_dir, ignore_errors=True)


# ------------------------- Training -------------------------

def train(args):
    set_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    # 1. Load Pretrained VAE
    print(f"[Info] Loading VAE from {args.vae_path} ...")
    try:
        vae = AutoencoderKL.from_pretrained(args.vae_path)
    except OSError:
        print("[Error] Could not load VAE. Make sure the path is correct.")
        return

    vae.eval()
    vae.requires_grad_(False)
    vae.to(device)
    
    # Calculate Latent Shape Dynamically
    dummy_img = torch.zeros(1, 3, args.image_size, args.image_size).to(device)
    with torch.no_grad():
        dummy_z = vae.encode(dummy_img).latent_dist.sample()
    latent_shape = dummy_z.shape[1:] # (C, H, W)
    print(f"[Info] Latent Shape determined: {latent_shape} (from {args.image_size}x{args.image_size} input)")
    
    # # 2. Setup LDM UNet
    # print(f"[Info] Initializing UNet for Latent Space...")
    # unet = UNet2DModel(
    #     sample_size=latent_shape[1],  
    #     in_channels=latent_shape[0],  
    #     out_channels=latent_shape[0], 
    #     layers_per_block=2,
    #     block_out_channels=tuple(args.unet_channels),
    #     down_block_types=("DownBlock2D",) * len(args.unet_channels),
    #     up_block_types=("UpBlock2D",) * len(args.unet_channels),
    #     norm_num_groups=32,
    # )

    # 2. Setup LDM UNet (Attention 적용 버전)
    print(f"[Info] Initializing UNet with Attention...")
    unet = UNet2DModel(
        sample_size=latent_shape[1],  # 16
        in_channels=latent_shape[0],  # Latent Channel (보통 4)
        out_channels=latent_shape[0], 
        layers_per_block=2,
        block_out_channels=tuple(args.unet_channels),

        # 얕은 층은 DownBlock, 깊은 층(해상도 낮은 곳)은 AttnDownBlock 사용
        down_block_types=(
            "DownBlock2D",      # 16x16 -> 8x8 (여기선 CNN만 써서 특징 추출)
            "AttnDownBlock2D",  # 8x8 -> 4x4   (Self-Attention 추가)
            "AttnDownBlock2D",  # 4x4 -> 2x2   (Self-Attention 추가)
        ),
        up_block_types=(
            "AttnUpBlock2D",    # 2x2 -> 4x4
            "AttnUpBlock2D",    # 4x4 -> 8x8
            "UpBlock2D",        # 8x8 -> 16x16
        ),
        norm_num_groups=32,
    )


    unet.to(device)
    print(f"[Info] UNet Parameters: {count_parameters(unet):,}")

    # 3. Schedulers
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type="epsilon",
    )
    ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)

    # 4. Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.mixed_precision == "fp16"))

    # 5. Data Loader
    train_ds = ImageFolderDataset(args.train_dir, args.image_size, args.center_crop, not args.no_hflip)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    
    # 6. FID Cache
    fid_real_cache = Path(args.output_dir) / "fid_real_cache"
    num_test_imgs = 0
    if not args.disable_fid and args.test_dir:
        num_test_imgs = flatten_real_cache(Path(args.test_dir), fid_real_cache)
    
    # --- Training Loop ---
    global_step = 0
    print(f"[Info] Start Training LDM... Epochs: {args.epochs}")

    for epoch in range(1, args.epochs + 1):
        unet.train()
        
        for step, x in enumerate(train_dl):
            x = x.to(device) # [-1, 1]
            
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(args.mixed_precision == "fp16")):
                # A. Encode Images to Latents (VAE is frozen)
                with torch.no_grad():
                    posterior = vae.encode(x).latent_dist
                    z = posterior.sample() 
                    # B. Scale Latents
                    z = z * args.latent_scale_factor

                # C. Add Noise to Latents
                noise = torch.randn_like(z)
                bsz = z.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
                
                noisy_z = noise_scheduler.add_noise(z, noise, timesteps)

                # D. Predict Noise
                noise_pred = unet(noisy_z, timesteps).sample

                # E. Loss
                loss = F.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            
            # Logs
            if global_step % args.log_interval == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/epoch": epoch,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }, step=global_step)
                print(f"[Ep{epoch}] Step {global_step} | Loss: {loss.item():.4f}")

            # Preview Sampling
            if (args.sample_interval > 0) and (global_step % args.sample_interval == 0):
                unet.eval() 
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=(args.mixed_precision == "fp16")):
                        gen_imgs = sample_ldm_ddim(
                            unet, vae, ddim_scheduler,
                            num_images=16,
                            latent_shape=latent_shape,
                            scale_factor=args.latent_scale_factor,
                            device=device,
                            steps=args.sample_steps
                        )
                    
                    grid = to_grid(gen_imgs, nrow=4)
                    save_path = output_dir / f"samples_step{global_step:06d}.png"
                    grid.save(save_path)
                    
                    if wandb.run is not None:
                        wandb.log({"samples_ldm": wandb.Image(grid)}, step=global_step)
                unet.train()

            # FID Evaluation
            if (args.fid_interval > 0) and (global_step % args.fid_interval == 0):
                if num_test_imgs > 0:
                    compute_fid(
                        unet, vae, ddim_scheduler, args, device, global_step, 
                        fid_real_cache, num_test_imgs, latent_shape
                    )

            # Checkpointing
            if (args.save_interval > 0) and (global_step % args.save_interval == 0):
                save_path = output_dir / f"unet_step{global_step:06d}"
                unet.save_pretrained(save_path)
                print(f"Saved UNet to {save_path}")

    # Final Save
    final_path = output_dir / "final_unet"
    unet.save_pretrained(final_path)
    print(f"Training Done. Saved final UNet to {final_path}")
    
    if num_test_imgs > 0:
        compute_fid(unet, vae, ddim_scheduler, args, device, global_step, fid_real_cache, num_test_imgs, latent_shape)

    wandb.finish()


# ------------------------- Arguments -------------------------

DATE=1227
B=256
LR=1e-4
CUDA_NUM=7

# [USER REQUIRED] VAE 경로와 Scale Factor를 설정하세요.
# LATENT_SCALE = 1.0 / (VAE 학습 시의 z_std)
LATENT_SCALE = 1 / 1.5018
VAE_CHECKPOINT = "1227_b64_lr0.0001_MSE_klW_1e-07_block_64_128_256-checkpoint-1000000" # 수정 필요

def parse_args():
    parser = argparse.ArgumentParser(description="LDM Training Single GPU")
    parser.add_argument("--device", type=str, default=f"cuda:{CUDA_NUM}")
    
    # Paths
    parser.add_argument("--vae_path", type=str, default=VAE_CHECKPOINT, help="Path to pretrained VAE folder")
    parser.add_argument("--train_dir", type=str, default="cifar10_png_linear_only/rgb/train")
    parser.add_argument("--test_dir", type=str, default="cifar10_png_linear_only/rgb/test", help="For FID")
    parser.add_argument("--output_dir", type=str, default=f"ldm_out_dir/{DATE}_cifar10_attn_unet_128_256_256_b{B}_lr{LR}")
    
    # WandB
    parser.add_argument("--project", type=str, default=f"ddpm-attn-cifar10-{DATE}")
    parser.add_argument("--run_name", type=str, default=f"ldm_attn_128_256_256_cifar10_b{B}")

    # Training
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=B)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16"])
    parser.add_argument("--train_timesteps", type=int, default=400)
    parser.add_argument("--beta_schedule", type=str, default="linear")

    # Latent / VAE Config
    parser.add_argument("--image_size", type=int, default=32, help="Pixel image size")
    parser.add_argument("--latent_scale_factor", type=float, default=LATENT_SCALE, help="Scaling factor for latents (1/z_std)")
    
    # UNet Config (Optimized for 16x16 Latent)
    # [128, 256, 256] corresponds to:
    # 16x16 -> (Down) -> 8x8 -> (Down) -> 4x4 -> (Down) -> 2x2 (Bottleneck)
    # This is safe and effective for small resolution.
    parser.add_argument("--unet_channels", type=int, nargs="+", default=[128, 256, 256], help="UNet channels")

    # Sampling / FID
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_eta", type=float, default=0.0)
    parser.add_argument("--sample_interval", type=int, default=5000)
    parser.add_argument("--fid_interval", type=int, default=5000)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--disable_fid", action="store_true")
    parser.add_argument("--fid_batch_size", type=int, default=64)
    parser.add_argument("--fid_gen_batch", type=int, default=256)
    parser.add_argument("--fid_dims", type=int, default=2048)
    parser.add_argument("--fid_keep_gen", action="store_true")

    # Data
    parser.add_argument("--no_hflip", action="store_true")
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--log_interval", type=int, default=100)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)