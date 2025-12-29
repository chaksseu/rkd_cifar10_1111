#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune a pre-trained DDPM using LoRA (Low-Rank Adaptation) with Hugging Face Accelerate.
Loads a pre-trained UNet, freezes it, injects LoRA layers using peft.get_peft_model,
and trains only the LoRA weights on new data.

Dependencies:
  pip install diffusers accelerate torch torchvision peft
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 모든 import 보다 먼저!
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
from peft import LoraConfig, get_peft_model, PeftModel

# ------------------------- Utils -------------------------

def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())

def to_grid(images: torch.Tensor, nrow: int = 4) -> Image.Image:
    """
    images: [-1,1] tensor in shape (N,3,H,W)
    returns: PIL Image of grid in [0,255]
    """
    imgs = (images.clamp(-1, 1) + 1) / 2.0 
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2)
    grid = (grid * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(grid)

def collect_image_paths_recursive(root: Path, exts={".png", ".jpg", ".jpeg"}) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def flatten_real_cache(test_dir: Path, cache_dir: Path, accelerator: Accelerator) -> int:
    ensure_dir(cache_dir)
    existing = list(cache_dir.glob("*"))
    if len(existing) > 0:
        return len(existing)

    paths = collect_image_paths_recursive(test_dir)
    accelerator.print(f"[FID] Flattening test set ({len(paths)} imgs) to {cache_dir} ...")
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
        Image.fromarray(arr).save(out_dir / f"gen_{start_idx + i:06d}.png")


# ------------------------- Dataset -------------------------

class ImageFolderDataset(Dataset):
    def __init__(self, root: str, image_size: int = 32, center_crop: bool = False, horizontal_flip: bool = True):
        self.root = Path(root)
        self.files: List[Path] = [p for p in self.root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if len(self.files) == 0:
            raise FileNotFoundError(f"No images found under {self.root}!")
        tfms = [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        ]
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
            x01 = self.to_tensor(img)
        x = x01 * 2.0 - 1.0  
        return x


# ------------------------- Sampling (DDIM) -------------------------

@torch.no_grad()
def sample_images_ddim(
    model, # PEFT model or UNet2DModel
    ddim_scheduler: DDIMScheduler,
    num_images: int,
    image_size: int,
    device: torch.device,
    steps: int = 50,
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    ddim_scheduler.set_timesteps(steps, device=device)

    # PEFT 모델의 경우 model.parameters()에서 dtype을 가져옴
    dtype = next(model.parameters()).dtype
    x = torch.randn((num_images, 3, image_size, image_size), device=device, dtype=dtype, generator=generator)

    for t in ddim_scheduler.timesteps:
        # LoRA가 적용된 모델은 forward 시 자동으로 어댑터를 통과함
        noise_pred = model(x, t).sample
        x = ddim_scheduler.step(noise_pred, t, x, eta=eta, generator=generator).prev_sample

    if was_training:
        model.train()
    return x


# ------------------------- FID -------------------------

def compute_and_log_fid(
    accelerator: Accelerator,
    model, # PEFT model
    ddim_scheduler: DDIMScheduler,
    args,
    device: torch.device,
    global_step: int,
    real_all_cache_dir: Path,
    num_test_imgs: int,
):
    if args.disable_fid: return

    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except Exception as e:
        accelerator.print(f"[FID] pytorch-fid not available: {e}. Skipping.")
        return

    gen_dir = Path(args.output_dir) / "fid" / f"step{global_step:06d}"
    if gen_dir.exists(): shutil.rmtree(gen_dir)
    ensure_dir(gen_dir)

    accelerator.print(f"[FID] Generating {num_test_imgs} samples to {gen_dir} ...")
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + int(global_step))

    remaining = num_test_imgs
    cursor = 0
    while remaining > 0:
        cur = min(args.fid_gen_batch, remaining)
        with torch.no_grad():
            autocast_device = "cuda" if device.type == "cuda" else "cpu"
            with torch.autocast(autocast_device, enabled=(accelerator.mixed_precision is not None)):
                # Unwrap if it's a distributed model
                unwrapped = accelerator.unwrap_model(model)
                xs = sample_images_ddim(
                    unwrapped, ddim_scheduler, num_images=cur, image_size=args.image_size,
                    device=device, steps=args.sample_steps, eta=args.sample_eta, generator=gen,
                )
        save_tensor_batch_to_dir(xs, gen_dir, start_idx=cursor)
        cursor += cur
        remaining -= cur

    accelerator.print(f"[FID] Computing FID against {real_all_cache_dir} ...")
    try:
        fid_all = calculate_fid_given_paths(
            [real_all_cache_dir.as_posix(), gen_dir.as_posix()],
            batch_size=args.fid_batch_size, device=device, dims=args.fid_dims,
        )
        accelerator.print(f"[FID] step={global_step} FID={fid_all:.4f}")
        accelerator.log({"metrics/fid_all": float(fid_all)}, step=global_step)
    except Exception as e:
        accelerator.print(f"[FID Error] {e}")

    if not args.fid_keep_gen:
        shutil.rmtree(gen_dir, ignore_errors=True)


# ------------------------- Training -------------------------

def train(args):
    # args.cuda_num이 설정되어 있으므로 Accelerator가 해당 디바이스를 자동으로 잡습니다 (main에서 세팅됨)
    torch.backends.cudnn.benchmark = True
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        log_with=["wandb"],
        kwargs_handlers=[ddp_kwargs],
        project_dir=args.output_dir,
        mixed_precision="fp16",
    )
    set_seed(args.seed)

    accelerator.init_trackers(
        project_name=args.project,
        config=vars(args),
        init_kwargs={"wandb": {"name": args.run_name, "resume": "allow"}},
    )

    device = accelerator.device
    accelerator.print(f"[Info] Accelerator Device: {device} (Physical GPU: {args.cuda_num}) | LoRA Rank: {args.lora_rank}")

    # 1. Load Pre-trained Base Model
    accelerator.print(f"[Info] Loading pre-trained model from {args.pretrained_model_path} ...")
    try:
        base_model = UNet2DModel.from_pretrained(args.pretrained_model_path)
    except Exception:
        # 혹시 subfolder 'unet'이나 'final' 안에 있다면 경로 수정 필요
        accelerator.print("[Warn] Direct load failed, trying subfolders...")
        if (Path(args.pretrained_model_path) / "unet").exists():
            base_model = UNet2DModel.from_pretrained(Path(args.pretrained_model_path) / "unet")
        elif (Path(args.pretrained_model_path) / "final").exists():
             base_model = UNet2DModel.from_pretrained(Path(args.pretrained_model_path) / "final")
        else:
            raise ValueError("Cannot load pre-trained UNet.")

    # 2. Inject LoRA using get_peft_model
    accelerator.print("[Info] Injecting LoRA adapters...")
    
    # 1) Base Model Freeze
    base_model.requires_grad_(False)
    
    # 2) Config
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"], # Attention 레이어 타겟팅
        init_lora_weights="gaussian",
    )
    
    # 3) Wrap Model (Replaces add_adapter)
    # This creates a PeftModel where only LoRA params are trainable
    model = get_peft_model(base_model, lora_config)
    
    # 4. Check Parameters
    trainable_params = count_parameters(model)
    total_params = count_total_parameters(model)
    accelerator.print(f"[Info] Total Params: {total_params:,}")
    accelerator.print(f"[Info] Trainable (LoRA) Params: {trainable_params:,} ({trainable_params/total_params:.2%})")

    # 5. Dataset (New Data for Fine-tuning)
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
    )

    # 6. FID Setup
    num_test_imgs = 0
    fid_real_all_dir = Path(args.output_dir) / "fid" / "real_cache" / "all"
    if accelerator.is_main_process and (not args.disable_fid) and args.test_dir:
        test_dir = Path(args.test_dir)
        if test_dir.exists():
            num_test_imgs = flatten_real_cache(test_dir, fid_real_all_dir, accelerator)
    accelerator.wait_for_everyone()

    # 7. Optimizer (Only optimize LoRA params)
    # Use filter just to be safe, though PeftModel should handle it
    lora_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=args.weight_decay)

    # 8. Schedulers (Load from pretrained if possible, or create new compatible one)
    try:
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    except:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.train_timesteps,
            beta_schedule=args.beta_schedule,
            prediction_type="epsilon",
        )
    ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)

    # 9. Prepare with Accelerator
    # Wrap model/opt/loader with accelerator
    # Note: 'model' is already the PEFT wrapped model
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        
        for step, x in enumerate(loader, start=1):
            with accelerator.accumulate(model):
                noise = torch.randn_like(x)
                bsz = x.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=x.device, dtype=torch.long
                )
                noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
                
                # Model Forward (Base + LoRA)
                pred = model(noisy_x, timesteps).sample
                
                loss = F.mse_loss(pred, noise)
                accelerator.backward(loss)

                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if (step % args.grad_accum) == 0:
                global_step += 1

                if global_step % args.log_interval == 0:
                    loss_val = accelerator.gather(loss).mean().item()
                    accelerator.log({"train/loss": loss_val, "train/epoch": epoch}, step=global_step)
                    accelerator.print(f"[Epoch {epoch}] Step {global_step} | Loss: {loss_val:.4f}")

                # ---- Sampling & FID ----
                if (args.sample_interval > 0) and (global_step % args.sample_interval == 0):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        # Unwrap PEFT model for sampling
                        unwrapped = accelerator.unwrap_model(model)
                        
                        # Preview
                        with torch.no_grad():
                            xs = sample_images_ddim(
                                unwrapped, ddim_scheduler, args.sample_n, args.image_size, device,
                                args.sample_steps, args.sample_eta
                            )
                        grid = to_grid(xs, nrow=int(math.isqrt(args.sample_n)))
                        out_path = Path(args.output_dir) / f"samples_step{global_step:06d}.png"
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        grid.save(out_path)
                        
                        try:
                            import wandb
                            accelerator.log({"samples": wandb.Image(grid)}, step=global_step)
                        except: pass

                        # FID
                        if not args.disable_fid and num_test_imgs > 0:
                            compute_and_log_fid(
                                accelerator, unwrapped, ddim_scheduler, args, device,
                                global_step, fid_real_all_dir, num_test_imgs
                            )
                    accelerator.wait_for_everyone()

                # ---- Checkpointing (Save LoRA Adapters Only) ----
                if (args.save_interval > 0) and (global_step % args.save_interval == 0):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_dir = Path(args.output_dir) / f"lora_step{global_step:06d}"
                        save_dir.mkdir(parents=True, exist_ok=True)
                        
                        unwrapped = accelerator.unwrap_model(model)
                        # PEFT 모델 저장 방식 (LoRA 가중치만 저장됨)
                        unwrapped.save_pretrained(save_dir.as_posix())
                        noise_scheduler.save_pretrained(save_dir.as_posix())
                        
                        accelerator.print(f"[Info] Saved LoRA adapter to {save_dir}")
                    accelerator.wait_for_everyone()

    # Final Save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = Path(args.output_dir) / "final_lora"
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(final_dir.as_posix())
        noise_scheduler.save_pretrained(final_dir.as_posix())
        accelerator.print(f"[Done] Saved final LoRA to {final_dir}")
    accelerator.end_training()


# ------------------------- Args -------------------------

DATE=1229
B=32
LR=1e-5 
CUDA_NUM=1

TT=400 
DDIM_STEPS=50

def build_argparser():
    p = argparse.ArgumentParser(description="LoRA Fine-tuning of DDPM")
    
    # Device
    p.add_argument("--cuda_num", type=str, default=str(CUDA_NUM), help="GPU ID to use (e.g. '0' or '7')")

    # Model Loading (Base Model)
    p.add_argument("--pretrained_model_path", type=str, default="ddpm_cifar10_rgb_T400_DDIM50/ckpt_step150000", 
                   help="Path to the folder containing the pre-trained UNet and scheduler (e.g. ./ddpm_cifar10/final)")
    # 

    # LoRA Config
    p.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA update matrices")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling factor")

    # Data (New Dataset)
    # p.add_argument("--train_dir", type=str, default="cifar10_png_linear_only/gray3/train", help="Path to NEW training data")
    p.add_argument("--train_dir", type=str, default="cifar10_student_data_n10", help="Path to NEW training data")
    p.add_argument("--test_dir", type=str, default="cifar10_png_linear_only/gray3/test", help="Path to NEW test data (for FID)")
    
    # Training Config
    p.add_argument("--output_dir", type=str, default=f"{DATE}_ddpm/ddpm_LoRA_gray_r32_a32_cifar10_rgb_T{TT}_DDIM{DDIM_STEPS}-b{B}-lr{LR}_n10", help="Where to save checkpoints & final model")
    p.add_argument("--project", type=str, default=f"ddpm-cifar10-{DATE}", help="W&B project name")
    p.add_argument("--run_name", type=str, default=f"ddpm-LoRA_gray_r32_a32_-b{B}-lr{LR}_n10", help="W&B run name")
    p.add_argument("--wandb_offline", action="store_true")
    
    p.add_argument("--epochs", type=int, default=100000)
    p.add_argument("--batch_size", type=int, default=B)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--center_crop", action="store_true")
    p.add_argument("--no_hflip", action="store_true")
    
    # Scheduler Params (should match base model or intended usage)
    p.add_argument("--train_timesteps", type=int, default=400)
    p.add_argument("--beta_schedule", type=str, default="linear")

    # Sampling
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--sample_eta", type=float, default=0.0)
    p.add_argument("--sample_interval", type=int, default=5000)
    p.add_argument("--save_interval", type=int, default=5000)
    p.add_argument("--sample_n", type=int, default=64)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # FID
    p.add_argument("--disable_fid", action="store_true")
    p.add_argument("--fid_batch_size", type=int, default=64)
    p.add_argument("--fid_gen_batch", type=int, default=512)
    p.add_argument("--fid_dims", type=int, default=2048)
    p.add_argument("--fid_keep_gen", action="store_true")

    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    
    # Set CUDA device before accelerator init
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_num)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)