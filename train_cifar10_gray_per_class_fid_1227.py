#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train an unconditional DDPM with Hugging Face Accelerate (multi-GPU ready),
and SAMPLE with DDIM. Logs losses & sample grids to Weights & Biases + always save local PNGs.
Additionally, computes FID (pytorch-fid) on sampling steps against a test set.
- Overall FID (all test images)
- Per-class FID (each test subdirectory vs same generated set)

Example:
  accelerate launch --mixed_precision fp16 --num_processes 2 train_ddpm_accel_ddim.py \
    --train_dir ./cifar10_png_linear_only/rgb/train \
    --test_dir  ./cifar10_png_linear_only/rgb/test \
    --output_dir ./ddpm_cifar10_rgb_accel \
    --project ddpm-cifar10-rgb \
    --run_name rgb-linear-ddpm-accel-ddim \
    --epochs 50 --batch_size 128 --lr 1e-4 \
    --sample_steps 50 --sample_eta 0.0 \
    --sample_interval 500 --save_interval 2000 \
    --grad_accum 1 --max_grad_norm 1.0
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"   # <= 이 줄 추가 (GPU 0만 보이게)


import os
import math
import argparse
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.utils as vutils

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler


# ------------------------- Utils -------------------------

def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())

def to_grid(images: torch.Tensor, nrow: int = 4) -> Image.Image:
    """
    images: [-1,1] tensor in shape (N,3,H,W)
    returns: PIL Image of grid in [0,255]
    """
    imgs = (images.clamp(-1, 1) + 1) / 2.0  # to [0,1]
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2)
    grid = (grid * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(grid)

def collect_image_paths_recursive(root: Path, exts={".png", ".jpg", ".jpeg"}) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def flatten_real_cache(test_dir: Path, cache_dir: Path, accelerator: Accelerator) -> int:
    """Flattens (symlink or copy) all images under test_dir into cache_dir for pytorch-fid."""
    ensure_dir(cache_dir)
    # If already populated, reuse
    existing = list(cache_dir.glob("*"))
    if len(existing) > 0:
        return len(existing)

    paths = collect_image_paths_recursive(test_dir)
    accelerator.print(f"[FID] Flattening test set ({len(paths)} imgs) to {cache_dir} ...")
    for i, src in enumerate(paths, 1):
        dst = cache_dir / f"real_{i:06d}{src.suffix.lower()}"
        try:
            # Prefer symlink to save disk
            os.symlink(src.resolve(), dst)
        except Exception:
            shutil.copy2(src, dst)
    return len(paths)

def save_tensor_batch_to_dir(x: torch.Tensor, out_dir: Path, start_idx: int):
    """
    x: [-1,1] (N,3,H,W)
    Save as PNG 8-bit.
    """
    ensure_dir(out_dir)
    x01 = (x.clamp(-1, 1) + 1) / 2.0
    x255 = (x01 * 255.0).clamp(0, 255).byte().cpu()
    for i in range(x255.shape[0]):
        arr = x255[i].permute(1, 2, 0).numpy()
        img = Image.fromarray(arr)
        img.save(out_dir / f"gen_{start_idx + i:06d}.png")


# ------------------------- Dataset -------------------------

class ImageFolderDataset(Dataset):
    """
    Recursively collects *.png (and *.jpg/*.jpeg) under root folder.
    Directory names can be class labels, but labels are ignored (unconditional).
    """
    def __init__(self, root: str, image_size: int = 32, center_crop: bool = False, horizontal_flip: bool = True):
        self.root = Path(root)
        exts = {".png", ".jpg", ".jpeg"}
        self.files: List[Path] = [p for p in self.root.rglob("*") if p.suffix.lower() in exts]
        if len(self.files) == 0:
            raise FileNotFoundError(f"No images found under {self.root}!")
        tfms = [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        ]
        if center_crop:
            tfms.append(T.CenterCrop(image_size))
        if horizontal_flip:
            tfms.append(T.RandomHorizontalFlip(p=0.5))
        tfms.append(T.ToTensor())  # -> [0,1]
        self.to_tensor = T.Compose(tfms)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            x01 = self.to_tensor(img)  # [0,1]
        x = x01 * 2.0 - 1.0           # [-1,1]
        return x


# ------------------------- Sampling (DDIM) -------------------------

@torch.no_grad()
def sample_images_ddim(
    model: UNet2DModel,
    ddim_scheduler: DDIMScheduler,
    num_images: int,
    image_size: int,
    device: torch.device,
    steps: int = 50,
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    DDIM sampling with given steps and eta.
    Returns a tensor in [-1,1] of shape (N,3,H,W).
    """
    was_training = model.training
    model.eval()
    ddim_scheduler.set_timesteps(steps, device=device)

    dtype = next(model.parameters()).dtype
    x = torch.randn((num_images, 3, image_size, image_size), device=device, dtype=dtype, generator=generator)

    for t in ddim_scheduler.timesteps:
        noise_pred = model(x, t).sample
        x = ddim_scheduler.step(noise_pred, t, x, eta=eta, generator=generator).prev_sample

    if was_training:
        model.train()
    return x


# ------------------------- FID -------------------------

def compute_and_log_fid(
    accelerator: Accelerator,
    unwrapped_model: UNet2DModel,
    ddim_scheduler: DDIMScheduler,
    args,
    device: torch.device,
    global_step: int,
    real_all_cache_dir: Path,
    num_test_imgs: int,
    per_class_cache_root: Optional[Path] = None,
    class_names: Optional[List[str]] = None,
    class_num_real_imgs: Optional[dict] = None,
):
    """
    Generate num_test_imgs samples into a temp dir, compute FID with pytorch-fid
    (overall + per-class), then log to W&B. Only called on main process.

    Per-class FID: each class's real cache vs the same generated set.
    """
    if args.disable_fid:
        return

    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except Exception as e:
        accelerator.print(f"[FID] pytorch-fid not available: {e}. Skipping FID.")
        return

    gen_dir = Path(args.output_dir) / "fid" / f"step{global_step:06d}"
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    ensure_dir(gen_dir)

    accelerator.print(f"[FID] Generating {num_test_imgs} samples to {gen_dir} ...")
    gen = torch.Generator(device=device)
    # Variation across steps while remaining deterministic given (seed, step)
    gen.manual_seed(int(args.seed) + int(global_step))

    remaining = num_test_imgs
    cursor = 0
    while remaining > 0:
        cur = min(args.fid_gen_batch, remaining)
        with torch.no_grad():
            autocast_device = "cuda" if device.type == "cuda" else "cpu"
            with torch.autocast(autocast_device, enabled=(accelerator.mixed_precision is not None)):
                xs = sample_images_ddim(
                    unwrapped_model,
                    ddim_scheduler,
                    num_images=cur,
                    image_size=args.image_size,
                    device=device,
                    steps=args.sample_steps,
                    eta=args.sample_eta,
                    generator=gen,
                )
        save_tensor_batch_to_dir(xs, gen_dir, start_idx=cursor)
        cursor += cur
        remaining -= cur

    # ---------- Overall FID ----------
    accelerator.print(f"[FID] Computing overall FID against {real_all_cache_dir} ...")
    fid_all = calculate_fid_given_paths(
        [real_all_cache_dir.as_posix(), gen_dir.as_posix()],
        batch_size=args.fid_batch_size,
        device=device,
        dims=args.fid_dims,
    )
    accelerator.print(f"[FID] step={global_step} FID(overall)={fid_all:.4f}")

    metrics = {
        "metrics/fid_all": float(fid_all),
    }

    # ---------- Per-class FID ----------
    per_class_fids = {}
    if per_class_cache_root is not None and class_names:
        accelerator.print(f"[FID] Computing per-class FID for {len(class_names)} classes ...")
        try:
            from pytorch_fid.fid_score import calculate_fid_given_paths
        except Exception as e:
            accelerator.print(f"[FID] pytorch-fid not available for per-class: {e}.")
        else:
            for cname in class_names:
                real_c_dir = per_class_cache_root / cname
                if not real_c_dir.exists():
                    accelerator.print(f"[FID]   [skip] Class '{cname}' cache dir {real_c_dir} not found.")
                    continue

                fid_c = calculate_fid_given_paths(
                    [real_c_dir.as_posix(), gen_dir.as_posix()],
                    batch_size=args.fid_batch_size,
                    device=device,
                    dims=args.fid_dims,
                )
                per_class_fids[cname] = fid_c
                metrics[f"metrics/fid_class/{cname}"] = float(fid_c)

            if per_class_fids:
                txt = ", ".join([f"{k}={v:.4f}" for k, v in per_class_fids.items()])
                accelerator.print(f"[FID] step={global_step} per-class FID: {txt}")

    # ---------- W&B logging ----------
    try:
        accelerator.log(metrics, step=global_step)
    except Exception:
        pass

    if not args.fid_keep_gen:
        # Clean generated images to save disk
        shutil.rmtree(gen_dir, ignore_errors=True)


# ------------------------- Training -------------------------

def train(args):
    torch.backends.cudnn.benchmark = True
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        log_with=["wandb"],
        kwargs_handlers=[ddp_kwargs],
        project_dir=args.output_dir,
        mixed_precision="fp16",  # or "bf16"
    )
    set_seed(args.seed)

    accelerator.init_trackers(
        project_name=args.project,
        config=vars(args),
        init_kwargs={"wandb": {"name": args.run_name, "resume": "allow"}},
    )

    device = accelerator.device
    accelerator.print(f"[Info] Using device: {device} | mixed_precision={accelerator.mixed_precision} | world_size={accelerator.num_processes}")

    # Dataset & DataLoader
    dataset = ImageFolderDataset(
        args.train_dir,
        image_size=args.image_size,
        center_crop=args.center_crop,
        horizontal_flip=not args.no_hflip
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
    )
    steps_per_epoch = math.ceil(len(dataset) / (args.batch_size * max(1, accelerator.num_processes) * max(1, args.grad_accum)))
    accelerator.print(f"[Info] Found {len(dataset)} training images under {args.train_dir}")
    accelerator.print(f"[Info] steps_per_epoch (optimizer steps) ≈ {steps_per_epoch}")

    # FID real cache (main only)
    num_test_imgs = 0
    fid_real_root_dir = Path(args.output_dir) / "fid" / "real_cache"
    fid_real_all_dir = fid_real_root_dir / "all"
    fid_real_per_class_root = fid_real_root_dir / "per_class"
    fid_class_names: List[str] = []
    fid_class_num_imgs: dict = {}

    if accelerator.is_main_process and (not args.disable_fid):
        if args.test_dir is None or len(args.test_dir.strip()) == 0:
            accelerator.print("[FID] --test_dir is empty; disabling FID.")
            args.disable_fid = True
        else:
            test_dir = Path(args.test_dir)
            if not test_dir.exists():
                accelerator.print(f"[FID] test_dir {test_dir} does not exist; disabling FID.")
                args.disable_fid = True
            else:
                # (1) Overall real cache (all classes)
                num_test_imgs = flatten_real_cache(test_dir, fid_real_all_dir, accelerator)
                accelerator.print(f"[FID] Real cache (all classes) ready: {num_test_imgs} images.")

                # (2) Per-class real caches
                class_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
                if len(class_dirs) == 0:
                    accelerator.print(f"[FID] No class subdirectories found under {test_dir}. Per-class FID will be skipped.")
                else:
                    accelerator.print(f"[FID] Preparing per-class real caches from {len(class_dirs)} subdirectories...")
                    for cd in sorted(class_dirs, key=lambda x: x.name):
                        cname = cd.name
                        cache_dir = fid_real_per_class_root / cname
                        n_c = flatten_real_cache(cd, cache_dir, accelerator)
                        fid_class_names.append(cname)
                        fid_class_num_imgs[cname] = n_c
                        accelerator.print(f"[FID]   Class '{cname}': {n_c} images.")

    accelerator.wait_for_everyone()
    # broadcast num_test_imgs to other ranks (for logs)
    num_test_imgs = accelerator.gather_for_metrics(torch.tensor([num_test_imgs], device=device)).max().item()

    # Model
    model = UNet2DModel(
        sample_size=args.image_size,
        in_channels=3,
        out_channels=3,
        block_out_channels=tuple(args.model_channels),  # e.g. (128, 256, 256)
        down_block_types=("DownBlock2D",) * len(args.model_channels),
        up_block_types=("UpBlock2D",) * len(args.model_channels),
        layers_per_block=2,
        norm_num_groups=32,
        attention_head_dim=None,
    )
    accelerator.print(f"[Info] Model parameters: {count_parameters(model):,}")

    # Schedulers
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type="epsilon",
    )
    ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)

    # Optimizer / LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Prepare
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        accelerator.print(f"[Epoch {epoch}] starting batches... len(loader)={len(loader)}")

        for step, x in enumerate(loader, start=1):
            with accelerator.accumulate(model):
                # x in [-1,1]
                noise = torch.randn_like(x)
                bsz = x.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=x.device, dtype=torch.long
                )
                noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
                pred = model(noisy_x, timesteps).sample
                loss = F.mse_loss(pred, noise)

                accelerator.backward(loss)

                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # accumulation 반영한 step 증가
            if (step % args.grad_accum) == 0 or (step == len(loader)):
                global_step += 1

                # -------- Logging --------
                if (global_step % args.log_interval == 0):
                    gathered = accelerator.gather_for_metrics({"loss": loss.detach()})
                    loss_detached = gathered["loss"].mean().item()
                    accelerator.log({
                        "train/loss": loss_detached,
                        "train/lr": float(args.lr),
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "data/test_num_images": int(num_test_imgs),
                    }, step=global_step)
                    accelerator.print(f"[Epoch {epoch:03d}] step={global_step:06d} loss={loss_detached:.4f} lr={args.lr:.2e}")

                # -------- Sampling (DDIM) + FID --------
                if (args.sample_interval > 0) and (global_step % args.sample_interval == 0):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(model)
                        with torch.no_grad():
                            autocast_device = "cuda" if device.type == "cuda" else "cpu"
                            with torch.autocast(autocast_device, enabled=(accelerator.mixed_precision is not None)):
                                imgs = sample_images_ddim(
                                    unwrapped, ddim_scheduler,
                                    num_images=args.sample_n, image_size=args.image_size, device=device,
                                    steps=args.sample_steps, eta=args.sample_eta
                                )
                                grid = to_grid(imgs, nrow=int(math.isqrt(args.sample_n)))

                        # ✅ 항상 로컬 저장 (preview grid)
                        out_path = Path(args.output_dir) / f"samples_step{global_step:06d}.png"
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        grid.save(out_path)

                        # (선택) W&B 업로드
                        try:
                            import wandb
                            accelerator.log({"samples_ddim": wandb.Image(grid)}, step=global_step)
                        except Exception:
                            pass

                        # ---- FID (overall + per-class) ----
                        if not args.disable_fid and num_test_imgs > 0:
                            compute_and_log_fid(
                                accelerator=accelerator,
                                unwrapped_model=unwrapped,
                                ddim_scheduler=ddim_scheduler,
                                args=args,
                                device=device,
                                global_step=global_step,
                                real_all_cache_dir=fid_real_all_dir,
                                num_test_imgs=int(num_test_imgs),
                                per_class_cache_root=fid_real_per_class_root,
                                class_names=fid_class_names,
                                class_num_real_imgs=fid_class_num_imgs,
                            )
                    accelerator.wait_for_everyone()

                # -------- Checkpointing --------
                if (args.save_interval > 0) and (global_step % args.save_interval == 0):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_dir = Path(args.output_dir) / f"ckpt_step{global_step:06d}"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.save_pretrained(save_dir.as_posix())
                        noise_scheduler.save_pretrained(save_dir.as_posix())
                        accelerator.print(f"[Info] Saved checkpoint to {save_dir}")
                    accelerator.wait_for_everyone()

        # -------- End-of-epoch sampling --------
        if args.sample_on_epoch_end:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                with torch.no_grad():
                    autocast_device = "cuda" if device.type == "cuda" else "cpu"
                    with torch.autocast(autocast_device, enabled=(accelerator.mixed_precision is not None)):
                        imgs = sample_images_ddim(
                            unwrapped, ddim_scheduler,
                            num_images=args.sample_n, image_size=args.image_size, device=device,
                            steps=args.sample_steps, eta=args.sample_eta
                        )
                        grid = to_grid(imgs, nrow=int(math.isqrt(args.sample_n)))

                # ✅ 항상 로컬 저장
                out_path = Path(args.output_dir) / f"samples_epoch_{epoch:03d}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                grid.save(out_path)

                # (선택) W&B 업로드
                try:
                    import wandb
                    accelerator.log({f"samples_ddim_epoch_{epoch:03d}": wandb.Image(grid)}, step=global_step)
                except Exception:
                    pass
            accelerator.wait_for_everyone()

        # Save "last" after each epoch
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            last_dir = Path(args.output_dir) / "last"
            last_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(last_dir.as_posix())
            noise_scheduler.save_pretrained(last_dir.as_posix())
        accelerator.wait_for_everyone()

    # -------- Final sampling & save (main only) --------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        with torch.no_grad():
            autocast_device = "cuda" if device.type == "cuda" else "cpu"
            with torch.autocast(autocast_device, enabled=(accelerator.mixed_precision is not None)):
                imgs = sample_images_ddim(
                    unwrapped, ddim_scheduler,
                    num_images=args.sample_n, image_size=args.image_size, device=device,
                    steps=args.sample_steps, eta=args.sample_eta
                )
                grid = to_grid(imgs, nrow=int(math.isqrt(args.sample_n)))

        # ✅ 항상 로컬 저장
        grid_path = Path(args.output_dir) / "final_samples_ddim.png"
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(grid_path)

        # (선택) W&B 업로드
        try:
            import wandb
            accelerator.log({"final_samples_ddim": wandb.Image(grid)}, step=global_step)
        except Exception:
            pass
        accelerator.print(f"[Done] Saved final DDIM samples to {grid_path}")

        final_dir = Path(args.output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(final_dir.as_posix())
        noise_scheduler.save_pretrained(final_dir.as_posix())
        accelerator.print(f"[Done] Saved model to {final_dir}")
    accelerator.wait_for_everyone()

    accelerator.end_training()

TT=400
DDIM_STEPS=50
def build_argparser():
    p = argparse.ArgumentParser(description="Accelerate-based unconditional DDPM training + DDIM sampling (W&B logging + local saves + FID overall & per-class)")
    # data / io
    # p.add_argument("--train_dir", type=str, default="./cifar10_png_linear_only/gray3/train", help="Folder with images (recursively reads *.png/*.jpg)")
    p.add_argument("--train_dir", type=str, default="cifar10_student_data_n100/gray3/train", help="Folder with images (recursively reads *.png/*.jpg)")
    p.add_argument("--test_dir",  type=str, default="./cifar10_png_linear_only/gray3/test",  help="Folder with class subdirs containing PNGs (used for FID)")
    p.add_argument("--output_dir", type=str, default=f"./ddpm_cifar10_gray3_T{TT}_DDIM{DDIM_STEPS}-CLASS_N100", help="Where to save checkpoints & final model")
    # logging
    p.add_argument("--project", type=str, default="ddpm-cifar10-gray3-1227", help="W&B project name")
    p.add_argument("--run_name", type=str, default="gray3-linear-ddpm-b256-lr1e4-CLASS_N100", help="W&B run name")
    p.add_argument("--wandb_offline", action="store_true", help="Use W&B offline mode (WANDB_MODE=offline)")
    # train
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--center_crop", action="store_true")
    p.add_argument("--no_hflip", action="store_true")
    p.add_argument("--train_timesteps", type=int, default=TT)
    p.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "scaled_linear", "squaredcos_cap_v2"])
    p.add_argument("--model_channels", type=int, nargs="+", default=[128, 256, 256],
                   help="UNet block_out_channels, e.g. --model_channels 128 256 256")
    # DDIM sampling
    p.add_argument("--sample_steps", type=int, default=DDIM_STEPS, help="DDIM sampling steps (typ. 25~100)")
    p.add_argument("--sample_eta", type=float, default=0.0, help="DDIM eta (0.0 = deterministic)")
    p.add_argument("--sample_interval", type=int, default=10000, help="Log samples every N optimizer steps (0 = never)")
    p.add_argument("--sample_n", type=int, default=64, help="How many images to sample for previews (make it a square number)")
    p.add_argument("--save_interval", type=int, default=10000, help="Save checkpoint every N optimizer steps (0 = never)")
    p.add_argument("--sample_on_epoch_end", action="store_true")
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    # FID
    p.add_argument("--disable_fid", action="store_true", help="Disable FID computation/logging")
    p.add_argument("--fid_batch_size", type=int, default=64, help="Batch size for Inception during FID")
    p.add_argument("--fid_gen_batch", type=int, default=512, help="Batch size for generating samples for FID")
    p.add_argument("--fid_dims", type=int, default=2048, help="Inception feature dims (default 2048)")
    p.add_argument("--fid_keep_gen", action="store_true", help="Keep generated images used for FID on disk")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
