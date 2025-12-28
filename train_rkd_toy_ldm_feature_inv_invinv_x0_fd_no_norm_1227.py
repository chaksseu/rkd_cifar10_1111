#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LDM Student (z0 predictor) distillation with Latent-based losses:
  - VAE Encode -> Latent Space Distillation -> VAE Decode
  - RKD (pdist) in Latent Space
  - INV (cdist) in Latent Space
  - INVINV (pdist) in Latent Space
  - Gaussian Loss (FID-like) in Latent Space
  - SAME loss in Latent Space

Dependencies:
  - pip install diffusers transformers torch torchvision pytorch-fid
"""

import os
import math
import argparse
import shutil
from pathlib import Path
from typing import List, Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image

# VAE 및 Scheduler
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DDIMInverseScheduler, AutoencoderKL

# ------------------------- Utils -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())

def to_grid(images: torch.Tensor, nrow: int = 4) -> Image.Image:
    # images: [-1, 1] -> [0, 255]
    imgs = (images.clamp(-1, 1) + 1) / 2.0
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2)
    grid = (grid * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(grid)

def save_tensor_batch_to_dir(x: torch.Tensor, out_dir: Path, start_idx: int):
    ensure_dir(out_dir)
    x01 = (x.clamp(-1, 1) + 1) / 2.0
    x255 = (x01 * 255.0).clamp(0, 255).byte().cpu()
    for i in range(x255.shape[0]):
        arr = x255[i].permute(1, 2, 0).numpy()
        Image.fromarray(arr).save(out_dir / f"gen_{start_idx + i:06d}.png")

def pdist_vec(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # x: (N, D) -> pdist
    return torch.pdist(x, p=2).clamp_min(eps)

def cdist_vec(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # x, y: (N, D) -> returns flattened cdist
    return torch.cdist(x, y, p=2).reshape(-1).clamp_min(eps)

def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def mean_from_vectors(vectors: List[torch.Tensor], device: torch.device, eps: float = 1e-12) -> torch.Tensor:
    if len(vectors) == 0:
        return torch.tensor(1.0, device=device)
    s = torch.zeros((), device=device, dtype=torch.float64)
    c = torch.zeros((), device=device, dtype=torch.float64)
    for vv in vectors:
        s = s + vv.sum().to(torch.float64)
        c = c + torch.tensor(float(vv.numel()), device=device, dtype=torch.float64)
    m = (s / c.clamp_min(1.0)).to(torch.float32)
    return m.clamp_min(eps)

def resolve_device(device_str: str) -> torch.device:
    try:
        dev = torch.device(device_str)
    except Exception:
        return torch.device("cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev

def collect_image_paths_recursive(root: Path, exts={".png", ".jpg", ".jpeg"}) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def flatten_real_cache(test_dir: Path, cache_dir: Path, use_symlink: bool = True) -> int:
    ensure_dir(cache_dir)
    existing = list(cache_dir.glob("*"))
    if len(existing) > 0:
        return len(existing)
    paths = collect_image_paths_recursive(test_dir)
    print(f"[FID] Flattening test set ({len(paths)} imgs) to {cache_dir} ...", flush=True)
    for i, src in enumerate(paths, 1):
        dst = cache_dir / f"real_{i:06d}{src.suffix.lower()}"
        try:
            if use_symlink and hasattr(os, 'symlink'):
                os.symlink(src.resolve(), dst)
            else:
                shutil.copy2(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    return len(paths)

# ------------------------- Gaussian Distance (Latent Space) -------------------------

def _mean_and_cov(X: torch.Tensor, eps: float = 1e-6):
    X = X.to(torch.float64)
    N, D = X.shape
    if N == 0:
        return torch.zeros(D, device=X.device), torch.eye(D, device=X.device)
    mu = X.mean(dim=0, keepdim=True)
    xc = X - mu
    denom = (N - 1) if N > 1 else 1
    C = (xc.t() @ xc) / denom
    I = torch.eye(D, dtype=X.dtype, device=X.device)
    C = 0.5 * (C + C.t()) + eps * I
    return mu.squeeze(0), C

def _sqrtm_psd(A: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    A = 0.5 * (A + A.t())
    evals, vecs = torch.linalg.eigh(A)
    evals = (evals + eps).clamp_min(0)
    return (vecs * evals.sqrt().unsqueeze(0)) @ vecs.t()

def fid_gaussian_torch(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # X, Y: (N, D) - Calculates Frechet Distance between two Gaussians
    mx, Cx = _mean_and_cov(X, eps)
    my, Cy = _mean_and_cov(Y, eps)
    mean_term = ((mx - my) ** 2).sum()
    Cy_sqrt = _sqrtm_psd(Cy, eps=eps)
    B = Cy_sqrt @ Cx @ Cy_sqrt
    B_sqrt = _sqrtm_psd(B, eps=eps)
    trace_term = torch.trace(Cx + Cy - 2.0 * B_sqrt)
    return (mean_term + trace_term).clamp_min(0.0).float()


# ------------------------- Dataset -------------------------

class StudentImageFolderDataset(Dataset):
    def __init__(self, root: str, image_size: int = 32, center_crop: bool = False, horizontal_flip: bool = True):
        self.root = Path(root)
        self.files = [p for p in self.root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if len(self.files) == 0:
            raise FileNotFoundError(f"No images found under {self.root}")

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
        return x * 2.0 - 1.0  # [-1, 1]


# ------------------------- Schedulers -------------------------

def load_scheduler(model_path: str, args) -> DDPMScheduler:
    try:
        return DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    except:
        return DDPMScheduler(
            num_train_timesteps=args.train_timesteps,
            beta_schedule=args.beta_schedule,
            prediction_type="epsilon"
        )

def make_ddim(ddpm: DDPMScheduler) -> DDIMScheduler:
    ddim = DDIMScheduler.from_config(ddpm.config)
    ddim.config.clip_sample = False 
    return ddim


# ------------------------- Sampling / Inversion (Latent Space) -------------------------

def teacher_predz0_seq(teacher, ddim_T, z_noisy, steps, eta, device) -> List[torch.Tensor]:
    # z_noisy: Latent Noise
    local = DDIMScheduler.from_config(ddim_T.config)
    local.set_timesteps(steps, device=device)
    z = z_noisy.to(device)

    teacher.eval()
    preds: List[torch.Tensor] = []
    for t in local.timesteps:
        z_in = local.scale_model_input(z, t)
        eps  = teacher(z_in, t).sample
        out  = local.step(model_output=eps, timestep=t, sample=z, eta=eta)
        z    = out.prev_sample
        preds.append(out.pred_original_sample) # z0 prediction
    return preds

def student_predz0_seq_with_grad(student, ddim_S, z_noisy, steps, eta, device) -> List[torch.Tensor]:
    local = DDIMScheduler.from_config(ddim_S.config)
    local.set_timesteps(steps, device=device)
    z = z_noisy.to(device)

    student.train()
    preds: List[torch.Tensor] = []
    for t in local.timesteps:
        z_in = local.scale_model_input(z, t)
        eps   = student(z_in, t).sample
        out  = local.step(model_output=eps, timestep=t, sample=z, eta=eta)
        z    = out.prev_sample
        preds.append(out.pred_original_sample) # z0 prediction
    return preds

def invert_z0_to_zT(model, ddim, z0, steps, device) -> torch.Tensor:
    # Deterministic DDIM Inversion in Latent Space
    inv = DDIMInverseScheduler.from_config(ddim.config)
    inv.set_timesteps(steps, device=device)
    
    # Normally inversion is done in eval mode
    was_train = model.training
    model.eval()
    
    zt = z0
    for t in inv.timesteps:
        latent_in = inv.scale_model_input(zt, t)
        with torch.no_grad():
            eps = model(latent_in, t).sample
        zt = inv.step(eps, t, zt).prev_sample
        
    if was_train: model.train()
    return zt

@torch.no_grad()
def sample_latents_ddim(
    unet: UNet2DModel,
    ddim: DDIMScheduler,
    num_images: int,
    latent_shape: tuple, # (C, H, W)
    device: torch.device,
    steps: int,
    eta: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    unet.eval()
    ddim.set_timesteps(steps, device=device)
    z = torch.randn((num_images, *latent_shape), device=device, generator=generator)

    for t in ddim.timesteps:
        z_in = ddim.scale_model_input(z, t)
        eps = unet(z_in, t).sample
        z = ddim.step(eps, t, z, eta=eta, generator=generator).prev_sample
    
    unet.train()
    return z


# ------------------------- Losses (Latent Space) -------------------------

def compute_losses(
    preds_T: List[torch.Tensor], # List of Latents [B, C, H, W]
    preds_S: List[torch.Tensor],
    z0_real: torch.Tensor,       # Latent of Real Image
    z0_inv_T: torch.Tensor,      # Teacher's generation from inverted noise
    args,
):
    eps = 1e-12
    device = z0_real.device
    
    # Flatten Latents to (B, D) for RKD
    # Dimensions: B x (C*H*W)
    def flatten_latent(z):
        return z.reshape(z.shape[0], -1)

    T_last_flat = flatten_latent(preds_T[-1])
    S_last_flat = flatten_latent(preds_S[-1])
    R_flat = flatten_latent(z0_real)
    I_flat = flatten_latent(z0_inv_T)

    # ---- RKD (Latent Space) ----
    rkd_s_list, rkd_t_list = [], []
    if args.w_rkd != 0.0:
        if args.rkd_teacher_ref == "last":
            T_ref_pdist = pdist_vec(T_last_flat, eps=eps)
            for k in range(0, len(preds_S), max(1, args.rkd_stride)):
                rkd_s_list.append(pdist_vec(flatten_latent(preds_S[k]), eps=eps))
                rkd_t_list.append(T_ref_pdist)
        else: # matched
            for k in range(0, len(preds_S), max(1, args.rkd_stride)):
                rkd_s_list.append(pdist_vec(flatten_latent(preds_S[k]), eps=eps))
                rkd_t_list.append(pdist_vec(flatten_latent(preds_T[k]), eps=eps))

    # ---- INV (Latent Space) ----
    inv_s = inv_t = None
    if args.w_inv != 0.0:
        inv_s = cdist_vec(S_last_flat, R_flat, eps=eps)
        inv_t = cdist_vec(T_last_flat, I_flat, eps=eps)

    # ---- INVINV (Latent Space) ----
    invinv_s = invinv_t = None
    if args.w_invinv != 0.0:
        invinv_s = pdist_vec(R_flat, eps=eps)
        invinv_t = pdist_vec(I_flat, eps=eps)

    # ---- Normalization ----
    student_parts, teacher_parts = [], []
    if args.w_rkd != 0.0 and len(rkd_s_list) > 0:
        student_parts.append(torch.cat(rkd_s_list))
        teacher_parts.append(torch.cat(rkd_t_list))
    if args.w_inv != 0.0 and inv_s is not None:
        student_parts.append(inv_s); teacher_parts.append(inv_t)
    if args.w_invinv != 0.0 and invinv_s is not None:
        student_parts.append(invinv_s); teacher_parts.append(invinv_t)

    if len(student_parts) > 0:
        student_mean = mean_from_vectors(student_parts, device, eps)
        teacher_mean = mean_from_vectors(teacher_parts, device, eps)
    else:
        student_mean = torch.tensor(1.0, device=device)
        teacher_mean = torch.tensor(1.0, device=device)

    # ---- Scalar Losses ----
    loss_rkd = torch.tensor(0.0, device=device)
    if args.w_rkd != 0.0:
        acc = 0.0
        for ds, dt in zip(rkd_s_list, rkd_t_list):
            acc += F.mse_loss(ds / student_mean, dt / teacher_mean)
        loss_rkd = acc / max(1, len(rkd_s_list))

    loss_inv = torch.tensor(0.0, device=device)
    if args.w_inv != 0.0:
        loss_inv = F.mse_loss(inv_s / student_mean, inv_t / teacher_mean)

    loss_invinv = torch.tensor(0.0, device=device)
    if args.w_invinv != 0.0:
        loss_invinv = F.mse_loss(invinv_s / student_mean, invinv_t / teacher_mean)

    # ---- Latent Gaussian Loss (Training FID proxy) ----
    loss_fid = torch.tensor(0.0, device=device)
    if args.w_fid != 0.0:
        # Calculate Gaussian distance on Latent Features
        fid_s = fid_gaussian_torch(S_last_flat, R_flat, eps=args.fid_eps)
        fid_t = fid_gaussian_torch(T_last_flat, I_flat, eps=args.fid_eps)
        loss_fid = fid_s + fid_t

    # ---- SAME (Latent Trajectory Regularization) ----
    loss_same = torch.tensor(0.0, device=device)
    if args.w_same != 0.0:
        zs = torch.stack(preds_S, dim=0) # [K, B, C, H, W]
        if args.same_mode == "mean":
            mu = zs.mean(dim=0, keepdim=True)
            loss_same = F.mse_loss(zs, mu.expand_as(zs))
        else:
            ref = zs[-1].detach()
            loss_same = F.mse_loss(zs[:-1], ref.unsqueeze(0).expand_as(zs[:-1]))

    total = (
        args.w_rkd * loss_rkd +
        args.w_inv * loss_inv +
        args.w_invinv * loss_invinv +
        args.w_fid * loss_fid +
        args.w_same * loss_same
    )

    stats = {
        "loss_rkd": loss_rkd.detach(),
        "loss_inv": loss_inv.detach(),
        "loss_invinv": loss_invinv.detach(),
        "loss_fid": loss_fid.detach(),
        "loss_same": loss_same.detach(),
        "s_mean": student_mean.detach()
    }
    return total, stats

# ------------------------- Eval: Pixel FID -------------------------

def compute_pixel_fid(real_dir, gen_dir, device, batch_size, dims):
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
        fid = calculate_fid_given_paths([str(real_dir), str(gen_dir)], batch_size, device, dims)
        return float(fid)
    except:
        return -1.0

@torch.no_grad()
def eval_and_sample(
    unet, vae, ddim, args, device, global_step, out_dir, latent_shape,
    wandb_run, fid_real_dir=None
):
    ensure_dir(out_dir / "samples")
    gen = torch.Generator(device=device).manual_seed(args.seed + global_step)
    
    # 1. Sample Latents
    z_gen = sample_latents_ddim(
        unet, ddim, args.sample_n, latent_shape, device, 
        args.sample_steps, args.sample_eta, gen
    )
    
    # 2. Decode to Pixel
    # Important: Latent rescale back
    z_gen = z_gen / args.latent_scale_factor
    imgs_gen = vae.decode(z_gen).sample # [-1, 1]
    
    # 3. Save Grid
    grid = to_grid(imgs_gen, nrow=int(math.sqrt(args.sample_n)))
    grid_path = out_dir / "samples" / f"step{global_step:08d}.png"
    grid.save(grid_path)
    
    if wandb_run:
        import wandb
        wandb.log({"eval/samples": wandb.Image(grid)}, step=global_step)
        
    # 4. FID (Pixel Space)
    if args.disable_fid or (fid_real_dir is None):
        return

    fid_gen_dir = out_dir / "fid_gen" / f"step{global_step}"
    ensure_dir(fid_gen_dir)
    
    # Generate needed amount
    needed = args.fid_num_samples
    done = 0
    while done < needed:
        cur = min(args.fid_gen_batch, needed - done)
        z = sample_latents_ddim(unet, ddim, cur, latent_shape, device, args.sample_steps, args.sample_eta)
        z = z / args.latent_scale_factor
        img = vae.decode(z).sample
        save_tensor_batch_to_dir(img, fid_gen_dir, done)
        done += cur
        
    fid_val = compute_pixel_fid(fid_real_dir, fid_gen_dir, device, args.fid_batch_size, 2048)
    print(f"[FID] Step {global_step}: {fid_val:.4f}")
    
    if wandb_run:
        wandb.log({"eval/fid": fid_val}, step=global_step)
        
    if not args.fid_keep_gen:
        shutil.rmtree(fid_gen_dir)


# ------------------------- Training -------------------------

def train(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[Info] Device: {device}")
    
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    # WandB
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            if args.wandb_offline: os.environ["WANDB_MODE"] = "offline"
            wandb_run = wandb.init(project=args.project, name=args.run_name, config=vars(args), dir=str(out_dir))
        except: pass

    # 1. Load VAE (Frozen)
    print(f"[Info] Loading VAE from {args.vae_path} ...")
    vae = AutoencoderKL.from_pretrained(args.vae_path)
    vae.eval()
    vae.requires_grad_(False)
    vae.to(device)
    
    # Determine Latent Shape
    with torch.no_grad():
        dummy_img = torch.zeros(1, 3, args.image_size, args.image_size).to(device)
        dummy_z = vae.encode(dummy_img).latent_dist.sample()
    latent_shape = dummy_z.shape[1:] # (C, H, W) e.g., (4, 16, 16)
    print(f"[Info] Latent Shape: {latent_shape}")

    # 2. Teacher (Latent UNet) - Load & Freeze
    print(f"[Info] Loading Teacher from {args.teacher_dir} ...")
    teacher = UNet2DModel.from_pretrained(args.teacher_dir, subfolder="unet" if args.use_subfolder else None).to(device)
    teacher.eval()
    teacher.requires_grad_(False)

    # 3. Student (Latent UNet)
    if args.student_dir:
        print(f"[Info] Loading Student from {args.student_dir} ...")
        student = UNet2DModel.from_pretrained(args.student_dir, subfolder="unet" if args.use_subfolder else None).to(device)
    else:
        print(f"[Info] Initializing New Student UNet ...")
        student = UNet2DModel(
            sample_size=latent_shape[1],
            in_channels=latent_shape[0], # Latent Channels
            out_channels=latent_shape[0],
            layers_per_block=args.layers_per_block,
            block_out_channels=tuple(args.unet_channels),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"), 
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
            norm_num_groups=args.norm_num_groups,
        ).to(device)
    
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Schedulers
    ddpm = load_scheduler(args.teacher_dir, args)
    ddim_T = make_ddim(ddpm)
    ddim_S = make_ddim(ddpm)

    # Data
    dataset = StudentImageFolderDataset(args.student_data_dir, args.image_size, args.center_crop, not args.no_hflip)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    
    # FID Cache
    fid_real_dir = None
    if (not args.disable_fid) and args.test_dir:
        fid_real_dir = out_dir / "fid_real_cache"
        flatten_real_cache(Path(args.test_dir), fid_real_dir)

    print(f"[Info] Start Training LDM Distillation...")
    global_step = 0
    scaler = torch.amp.GradScaler('cuda', enabled=(args.mixed_precision == "fp16"))
    
    for epoch in range(1, args.epochs + 1):
        for x_img in loader:
            x_img = x_img.to(device)
            
            # 1. Encode Pixel -> Latent (VAE)
            with torch.no_grad():
                # VAE Encode
                dist = vae.encode(x_img).latent_dist
                z0_real = dist.sample() * args.latent_scale_factor # Scale!
            
            # 2. Random Noise
            noise = torch.randn_like(z0_real)
            
            # 3. Determine Steps
            steps = torch.randint(args.ddim_steps_min, args.ddim_steps_max + 1, (1,)).item()
            
            # 4. Distillation Forward (Latent Space)
            with torch.amp.autocast('cuda', enabled=(args.mixed_precision == "fp16")):
                # Teacher Prediction
                with torch.no_grad():
                    preds_T = teacher_predz0_seq(teacher, ddim_T, noise, steps, args.ddim_eta, device)
                
                # Student Prediction (Gradients needed)
                preds_S = student_predz0_seq_with_grad(student, ddim_S, noise, steps, args.ddim_eta, device)
                
                # Inversion (on Real Latent)
                # Teacher inversion: z0_real -> zT
                # Note: For strict distillation, we often invert using Teacher or Student.
                # Here we use Student for inversion to get trajectory, or Teacher. 
                # Let's use Student for inversion as per common practice in some distillation papers,
                # or Teacher if we want to align with Teacher's space. 
                # Given RKD, let's invert using Student to align generation loop? 
                # Typically INV loss compares Teacher(Inverted_Real) vs Real.
                zT_real = invert_z0_to_zT(student, ddim_S, z0_real, steps, device)
                
                # Teacher Generation from Inverted Noise
                with torch.no_grad():
                    preds_T_inv = teacher_predz0_seq(teacher, ddim_T, zT_real, steps, args.ddim_eta, device)
                z0_inv_T = preds_T_inv[-1]

                # 5. Compute Losses (All in Latent Space)
                loss, stats = compute_losses(preds_T, preds_S, z0_real, z0_inv_T, args)

            # Optimization
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            global_step += 1
            
            if global_step % args.log_interval == 0:
                print(f"[Ep{epoch}] Step {global_step} | Loss: {loss.item():.4f} | RKD: {stats['loss_rkd']:.4f}")
                if wandb_run:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/rkd": stats['loss_rkd'].item(),
                        "train/inv": stats['loss_inv'].item(),
                        "train/fid_latent": stats['loss_fid'].item(),
                        "train/same": stats['loss_same'].item(),
                    }, step=global_step)
            
            if global_step % args.sample_interval == 0:
                eval_and_sample(
                    student, vae, ddim_S, args, device, global_step, out_dir, 
                    latent_shape, wandb_run, fid_real_dir
                )
            
            if global_step % args.save_interval == 0:
                save_path = out_dir / f"ckpt_step{global_step:06d}"
                student.save_pretrained(save_path)
                print(f"Saved student to {save_path}")

    final_path = out_dir / "final_student"
    student.save_pretrained(final_path)
    print("Done.")


# ------------------------- Args -------------------------

BATCH_SIZE = 8
CLASSN = 100
RKD_METRIC="pixel" # pixel inception clip
CUDA_NUM = 3
LR=1e-5

RKD_W = 0.1
INV_W = 0.1
INVINV_W = 1.0
FD_W = 0.0001
SAME_W = 0.1

def parse_args():
    parser = argparse.ArgumentParser(description="LDM RKD Distillation")
    # Paths
    parser.add_argument("--vae_path", type=str, default="1227_b64_lr0.0001_MSE_klW_1e-08_block_64_128-checkpoint-1000000", help="Pretrained VAE path")
    parser.add_argument("--teacher_dir", type=str, default="1228_cifar10_unet_64_128_b256_lr0.0001_rgb_unet_step150000", help="Pretrained Latent Teacher UNet")
    parser.add_argument("--student_dir", type=str, default="1228_cifar10_unet_64_128_b256_lr0.0001_rgb_unet_step150000", help="Pretrained Student (Optional)")
    parser.add_argument("--student_data_dir", type=str, default="cifar10_student_data_n100/gray3/train")
    parser.add_argument("--test_dir", type=str, default="cifar10_png_linear_only/gray3/test")
    parser.add_argument("--output_dir", type=str, default=f"out_1228_rkd_{RKD_METRIC}_LDM_feature_cifar10_rgb_to_gray_single_batch{BATCH_SIZE}_N{CLASSN}_LR{LR}-FD-rkdW{RKD_W}-invW{INV_W}-invinvW{INVINV_W}-fdW{FD_W}-sameW{SAME_W}-teacher-init-eps")
    parser.add_argument("--use_subfolder", action="store_true", help="If models are in subfolders like 'unet'")

    # LDM Config
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--latent_scale_factor", type=float, default=0.40365, help="Standard SD: 0.18215, User custom: 1/2.4774")
    parser.add_argument("--unet_channels", type=int, nargs="+", default=[128, 256, 256])
    parser.add_argument("--layers_per_block", type=int, default=2)
    parser.add_argument("--norm_num_groups", type=int, default=32)
    
    # Training
    parser.add_argument("--device", type=str, default=f"cuda:{CUDA_NUM}")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--train_timesteps", type=int, default=400)
    parser.add_argument("--beta_schedule", type=str, default="linear")

    # Distillation
    parser.add_argument("--ddim_steps_min", type=int, default=40)
    parser.add_argument("--ddim_steps_max", type=int, default=60)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    
    # Loss Weights
    parser.add_argument("--w_rkd", type=float, default=RKD_W)
    parser.add_argument("--w_inv", type=float, default=INV_W)
    parser.add_argument("--w_invinv", type=float, default=INVINV_W)
    parser.add_argument("--w_fid", type=float, default=FD_W)
    parser.add_argument("--w_same", type=float, default=SAME_W)
    
    parser.add_argument("--rkd_stride", type=int, default=1)
    parser.add_argument("--rkd_teacher_ref", type=str, default="last")
    parser.add_argument("--same_mode", type=str, default="mean")
    parser.add_argument("--fid_eps", type=float, default=1e-8)

    # Logging / Sampling
    parser.add_argument("--project", type=str, default="rkd-feature-cifar10-rgb-to-gray-1228")
    parser.add_argument("--run_name", type=str, default=f"student-{RKD_METRIC}-LDM-x0-pixel-rgb-to-gray-batch{BATCH_SIZE}-N{CLASSN}-LR{LR}-FD-rkdW{RKD_W}-invW{INV_W}-invinvW{INVINV_W}-fdW{FD_W}-sameW{SAME_W}-teacher-init-eps")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_offline", action="store_true")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--sample_interval", type=int, default=5000)
    parser.add_argument("--sample_n", type=int, default=64)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_eta", type=float, default=0.0)
    
    # FID
    parser.add_argument("--disable_fid", action="store_true")
    parser.add_argument("--fid_num_samples", type=int, default=1000)
    parser.add_argument("--fid_batch_size", type=int, default=64)
    parser.add_argument("--fid_gen_batch", type=int, default=128)
    parser.add_argument("--fid_keep_gen", action="store_true")
    
    # Data Aug
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--no_hflip", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Default VAE Scale Override if needed
    # args.latent_scale_factor = 1 / 2.4774 
    
    train(args)