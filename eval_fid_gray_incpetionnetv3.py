#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate (generate + FID) for CIFAR-10 gray3 diffusion models.
- Supports:
  (A) Full UNet2DModel dir saved by diffusers (.save_pretrained)
  (B) LoRA adapter dir saved by peft (adapter_config.json present) + base UNet dir

Computes:
- Overall FID (real all vs generated)
- Per-class FID (each real class vs same generated set) if test_dir has class subfolders

NOTE:
- This version computes FID using YOUR trained Inception-v3 checkpoint (custom backbone),
  NOT the default pytorch-fid Inception.
- FID values are therefore "custom-FID" and not directly comparable to standard FID unless
  you use the same backbone + preprocessing everywhere.

Deps:
  pip install diffusers torch torchvision peft wandb numpy pillow

Example:
  python eval_generate_fid_custom_inception.py \
    --model_dirs out/runA/last out/runB/ckpts/ckpt_step020000 \
    --base_model_dir ddpm_cifar10_gray3_T400_DDIM50/ckpt_step150000 \
    --test_dir cifar10_png_linear_only/gray3/test \
    --output_dir eval_fid_out \
    --device cuda:0 --mixed_precision fp16 \
    --sample_steps 50 --sample_eta 0.0 \
    --fid_num_samples 0 --fid_gen_batch 512 --fid_batch_size 64 \
    --inception_ckpt ./inceptionv3_gray_imagenet_ckpt.pt \
    --inception_num_classes 1000 \
    --inception_input_size 299 \
    --inception_mean 0.45798322587856827 0.45798322587856827 0.45798322587856827 \
    --inception_std  0.2623006911570552  0.2623006911570552  0.2623006911570552 \
    --project cifar10-gray3-fid --run_name eval-jan02 --wandb
"""

import os
import re
import json
import math
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.utils as vutils
import torchvision.models as models
from PIL import Image

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler


# ------------------------- IO Utils -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_grid(images: torch.Tensor, nrow: int = 8) -> Image.Image:
    # images: [-1,1], (N,3,H,W)
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

def collect_image_paths_recursive(root: Path, exts={".png", ".jpg", ".jpeg"}) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def flatten_real_cache(src_dir: Path, cache_dir: Path, use_symlink: bool = True) -> int:
    """
    Flatten any nested images under src_dir into cache_dir for stable scanning.
    Uses symlink by default (fast, low disk); falls back to copy if symlink fails.
    """
    ensure_dir(cache_dir)
    existing = list(cache_dir.glob("*"))
    if len(existing) > 0:
        return len(existing)

    paths = collect_image_paths_recursive(src_dir)
    print(f"[FID] Flattening real set ({len(paths)} imgs) -> {cache_dir}", flush=True)

    for i, src in enumerate(paths, 1):
        dst = cache_dir / f"real_{i:06d}{src.suffix.lower()}"
        try:
            if use_symlink:
                os.symlink(src.resolve(), dst)
            else:
                shutil.copy2(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    return len(paths)

def list_class_dirs(test_dir: Path) -> List[Path]:
    return [d for d in test_dir.iterdir() if d.is_dir()]


# ------------------------- Model Loading (UNet / LoRA) -------------------------

def is_lora_adapter_dir(d: Path) -> bool:
    if not d.exists() or not d.is_dir():
        return False
    return (d / "adapter_config.json").exists() or (d / "adapter_model.bin").exists() or (d / "adapter_model.safetensors").exists()

def load_unet_any(
    model_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    base_model_dir: Optional[Path] = None,
    lora_merge: bool = False,
) -> Tuple[torch.nn.Module, str]:
    """
    Returns (model, kind) where kind in {"full", "lora"}.
    """
    if is_lora_adapter_dir(model_dir):
        if base_model_dir is None:
            raise ValueError(f"LoRA adapter detected at {model_dir} but --base_model_dir not provided.")
        from peft import PeftModel

        print(f"[Load] LoRA adapter: {model_dir}", flush=True)
        print(f"[Load] Base UNet:     {base_model_dir}", flush=True)

        base = UNet2DModel.from_pretrained(base_model_dir.as_posix())
        base.to(device=device, dtype=dtype)
        base.eval()

        model = PeftModel.from_pretrained(base, model_dir.as_posix(), is_trainable=False)
        model.to(device=device, dtype=dtype)
        model.eval()

        if lora_merge and hasattr(model, "merge_and_unload"):
            try:
                model = model.merge_and_unload()
                model.to(device=device, dtype=dtype)
                model.eval()
                print("[Load] LoRA merged into base weights (merge_and_unload).", flush=True)
            except Exception as e:
                print(f"[Warn] LoRA merge failed (will run unmerged): {e}", flush=True)

        return model, "lora"

    print(f"[Load] Full UNet: {model_dir}", flush=True)
    model = UNet2DModel.from_pretrained(model_dir.as_posix())
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, "full"

def load_scheduler_any(
    model_dir: Path,
    train_timesteps: int,
    beta_schedule: str,
    prediction_type: str,
    base_model_dir: Optional[Path] = None,
) -> DDPMScheduler:
    """
    Try loading scheduler from model_dir; else from base_model_dir; else fallback to fresh config.
    """
    for cand in [model_dir, base_model_dir]:
        if cand is None:
            continue
        try:
            return DDPMScheduler.from_pretrained(cand.as_posix())
        except Exception:
            pass

    return DDPMScheduler(
        num_train_timesteps=train_timesteps,
        beta_schedule=beta_schedule,
        prediction_type=prediction_type,
    )

def make_ddim(ddpm: DDPMScheduler) -> DDIMScheduler:
    ddim = DDIMScheduler.from_config(ddpm.config)
    ddim.config.clip_sample = False
    ddim.config.prediction_type = ddpm.config.prediction_type
    return ddim


# ------------------------- Sampling -------------------------

@torch.no_grad()
def sample_images_ddim(
    model,
    ddim: DDIMScheduler,
    num_images: int,
    image_size: int,
    device: torch.device,
    steps: int,
    eta: float,
    generator: Optional[torch.Generator],
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    was_training = model.training
    model.eval()

    local = DDIMScheduler.from_config(ddim.config)
    local.set_timesteps(steps, device=device)

    try:
        model_dtype = next(model.parameters()).dtype
    except Exception:
        model_dtype = torch.float32

    x = torch.randn((num_images, 3, image_size, image_size), device=device, dtype=model_dtype, generator=generator)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )

    with autocast_ctx:
        for t in local.timesteps:
            x_in = local.scale_model_input(x, t)
            model_out = model(x_in, t).sample
            x = local.step(model_output=model_out, timestep=t, sample=x, eta=eta, generator=generator).prev_sample

    if was_training:
        model.train()
    return x


# ------------------------- Custom Inception FID -------------------------

def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # for DataParallel/DistributedDataParallel checkpoints
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}

def _load_ckpt_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
            sd = load_file(path.as_posix())
            return dict(sd)
        except Exception as e:
            raise RuntimeError(f"Failed to load safetensors: {path} ({e})")

    obj = torch.load(path.as_posix(), map_location="cpu")
    if isinstance(obj, dict):
        # common patterns
        for key in ["state_dict", "model_state_dict", "model", "net", "ema", "student", "teacher"]:
            if key in obj and isinstance(obj[key], dict):
                return _strip_module_prefix(obj[key])
        # might be raw state_dict already
        if all(isinstance(k, str) for k in obj.keys()):
            return _strip_module_prefix(obj)
    raise RuntimeError(f"Unrecognized checkpoint format: {path}")

def build_inception_v3(num_classes: int, aux_logits: bool) -> nn.Module:
    # torchvision inception_v3
    # weights=None ensures no pretrained weights are loaded
    try:
        m = models.inception_v3(weights=None, aux_logits=aux_logits, transform_input=False)
    except TypeError:
        # older torchvision
        m = models.inception_v3(pretrained=False, aux_logits=aux_logits, transform_input=False)

    # replace final fc to match num_classes if needed
    if hasattr(m, "fc") and isinstance(m.fc, nn.Linear):
        if m.fc.out_features != num_classes:
            m.fc = nn.Linear(m.fc.in_features, num_classes, bias=True)
    return m

@torch.no_grad()
def inception_v3_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Return 2048-d pre-logits features of torchvision InceptionV3.
    x: (B,3,H,W) already normalized as in training.
    """
    # This mirrors torchvision.models.inception.Inception3 forward up to avgpool.
    # See torchvision source for exact module names.
    x = model.Conv2d_1a_3x3(x)
    x = model.Conv2d_2a_3x3(x)
    x = model.Conv2d_2b_3x3(x)
    x = model.maxpool1(x)

    x = model.Conv2d_3b_1x1(x)
    x = model.Conv2d_4a_3x3(x)
    x = model.maxpool2(x)

    x = model.Mixed_5b(x)
    x = model.Mixed_5c(x)
    x = model.Mixed_5d(x)

    x = model.Mixed_6a(x)
    x = model.Mixed_6b(x)
    x = model.Mixed_6c(x)
    x = model.Mixed_6d(x)
    x = model.Mixed_6e(x)

    x = model.Mixed_7a(x)
    x = model.Mixed_7b(x)
    x = model.Mixed_7c(x)

    # Inception3 uses adaptive_avg_pool2d((1,1)) in some versions, but avgpool exists.
    if hasattr(model, "avgpool"):
        x = model.avgpool(x)
    else:
        x = F.adaptive_avg_pool2d(x, (1, 1))

    x = torch.flatten(x, 1)  # (B,2048)
    return x

class ImageDirDataset(Dataset):
    def __init__(self, root_dir: Path, input_size: int, mean: List[float], std: List[float]):
        self.paths = sorted(collect_image_paths_recursive(root_dir))
        self.tf = T.Compose([
            T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),  # [0,1]
            T.Normalize(mean=mean, std=std),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.tf(img)

@torch.no_grad()
def compute_activation_stats_from_dir(
    img_dir: Path,
    inception: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    input_size: int,
    mean: List[float],
    std: List[float],
) -> Tuple[np.ndarray, np.ndarray, int]:
    ds = ImageDirDataset(img_dir, input_size=input_size, mean=mean, std=std)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    feats = []
    for x in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        f = inception_v3_features(inception, x)
        feats.append(f.cpu().numpy().astype(np.float64))  # float64 for stable cov
    feats = np.concatenate(feats, axis=0) if len(feats) > 0 else np.zeros((0, 2048), dtype=np.float64)

    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma, feats.shape[0]

def sqrtm_psd(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    # Symmetric PSD sqrt via eigh; clip eigenvalues for numerical stability.
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, eps, None)
    return (vecs * np.sqrt(vals)) @ vecs.T

def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    diff = mu1 - mu2
    # Stable trace(sqrtm(sigma1*sigma2)) via symmetric product:
    # Tr(sqrt(sigma1*sigma2)) == Tr(sqrt(sqrt(sigma1)*sigma2*sqrt(sigma1)))
    s1_sqrt = sqrtm_psd(sigma1)
    prod = s1_sqrt @ sigma2 @ s1_sqrt
    covmean = sqrtm_psd(prod)

    fid = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))
    # small negative due to numerical errors
    return max(0.0, fid)

def load_custom_inception(
    ckpt_path: Path,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    sd = _load_ckpt_state_dict(ckpt_path)
    aux_logits = any(("AuxLogits" in k) for k in sd.keys())
    model = build_inception_v3(num_classes=num_classes, aux_logits=aux_logits)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 0:
        print(f"[Warn] Inception missing keys: {len(missing)} (showing up to 5): {missing[:5]}", flush=True)
    if len(unexpected) > 0:
        print(f"[Warn] Inception unexpected keys: {len(unexpected)} (showing up to 5): {unexpected[:5]}", flush=True)

    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


# ------------------------- Evaluation Per Model -------------------------

def eval_one_model(
    model_dir: Path,
    tag: str,
    args,
    device: torch.device,
    wandb_run,
    wandb_mod,
    real_all_stats: Optional[Tuple[np.ndarray, np.ndarray, int]],
    real_class_stats: Dict[str, Tuple[np.ndarray, np.ndarray, int]],
    inception: Optional[nn.Module],
):
    # dtype / amp (for diffusion sampling)
    use_amp = (device.type == "cuda") and (args.mixed_precision != "no")
    if args.mixed_precision == "fp16":
        amp_dtype = torch.float16
        model_dtype = torch.float16 if use_amp else torch.float32
    elif args.mixed_precision == "bf16":
        amp_dtype = torch.bfloat16
        model_dtype = torch.bfloat16 if use_amp else torch.float32
    else:
        amp_dtype = torch.float16
        model_dtype = torch.float32

    base_model_dir = Path(args.base_model_dir) if args.base_model_dir else None

    model, kind = load_unet_any(
        model_dir=model_dir,
        device=device,
        dtype=model_dtype,
        base_model_dir=base_model_dir,
        lora_merge=args.lora_merge,
    )

    ddpm = load_scheduler_any(
        model_dir=model_dir,
        train_timesteps=args.train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
        base_model_dir=base_model_dir,
    )
    ddim = make_ddim(ddpm)

    # output dirs
    out_root = Path(args.output_dir) / tag
    samples_dir = out_root / "samples"
    fid_dir = out_root / "fid"
    ensure_dir(samples_dir)
    ensure_dir(fid_dir)

    # deterministic generator
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + (abs(hash(tag)) % 10_000))

    # ---- grid sample (keep) ----
    grid_n = args.grid_n
    nrow = int(math.isqrt(grid_n))
    if nrow * nrow != grid_n:
        nrow = min(grid_n, 8)

    imgs = sample_images_ddim(
        model=model,
        ddim=ddim,
        num_images=grid_n,
        image_size=args.image_size,
        device=device,
        steps=args.sample_steps,
        eta=args.sample_eta,
        generator=gen,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    grid = to_grid(imgs, nrow=nrow)
    grid_path = samples_dir / f"grid_{tag}.png"
    grid.save(grid_path)
    print(f"[Sample] Saved grid -> {grid_path}", flush=True)

    if wandb_run is not None and wandb_mod is not None:
        try:
            wandb_mod.log({f"{tag}/grid": wandb_mod.Image(grid)}, step=0)
        except Exception:
            pass

    # If FID disabled: stop here
    if args.disable_fid:
        print(f"[FID] --disable_fid set: skip FID for {tag}.", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return

    if inception is None or real_all_stats is None:
        raise RuntimeError("FID requested but inception/real stats are not prepared. Check --inception_ckpt and real cache steps.")

    # ---- FID generation (disk) ----
    fid_num = real_all_stats[2] if args.fid_num_samples <= 0 else int(min(args.fid_num_samples, real_all_stats[2]))
    gen_dir = fid_dir / "gen_tmp"
    if gen_dir.exists():
        shutil.rmtree(gen_dir, ignore_errors=True)
    ensure_dir(gen_dir)

    remaining = fid_num
    cursor = 0
    print(f"[FID] Generating {fid_num} images -> {gen_dir}", flush=True)
    while remaining > 0:
        cur = min(args.fid_gen_batch, remaining)
        xs = sample_images_ddim(
            model=model,
            ddim=ddim,
            num_images=cur,
            image_size=args.image_size,
            device=device,
            steps=args.sample_steps,
            eta=args.sample_eta,
            generator=gen,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        save_tensor_batch_to_dir(xs, gen_dir, start_idx=cursor)
        cursor += cur
        remaining -= cur

    # ---- compute gen stats + FID ----
    mu_r, sig_r, n_r = real_all_stats

    mu_g, sig_g, n_g = compute_activation_stats_from_dir(
        img_dir=gen_dir,
        inception=inception,
        device=device,
        batch_size=args.fid_batch_size,
        num_workers=args.fid_num_workers,
        input_size=args.inception_input_size,
        mean=args.inception_mean,
        std=args.inception_std,
    )
    fid_all = frechet_distance(mu_r, sig_r, mu_g, sig_g)

    results = {
        "tag": tag,
        "model_dir": model_dir.as_posix(),
        "kind": kind,
        "fid_num_samples": int(fid_num),
        "sample_steps": int(args.sample_steps),
        "sample_eta": float(args.sample_eta),
        "inception_ckpt": args.inception_ckpt,
        "inception_input_size": int(args.inception_input_size),
        "fid_all": float(fid_all),
    }
    print(f"[FID] {tag} | overall FID = {fid_all:.4f} (real={n_r}, gen={n_g})", flush=True)

    fid_per_class = {}
    if args.fid_per_class and len(real_class_stats) > 0:
        for cname, (mu_c, sig_c, n_c) in real_class_stats.items():
            fid_c = frechet_distance(mu_c, sig_c, mu_g, sig_g)
            fid_per_class[cname] = float(fid_c)
        results["fid_per_class"] = fid_per_class

        txt = ", ".join([f"{k}={v:.4f}" for k, v in sorted(fid_per_class.items(), key=lambda kv: kv[0])])
        print(f"[FID] {tag} | per-class: {txt}", flush=True)

    # ---- logging ----
    if wandb_run is not None and wandb_mod is not None:
        try:
            log_dict = {
                f"{tag}/fid_all": float(fid_all),
                f"{tag}/fid_num_samples": int(fid_num),
                f"{tag}/kind": kind,
            }
            if fid_per_class:
                for k, v in fid_per_class.items():
                    log_dict[f"{tag}/fid_class/{k}"] = float(v)
            wandb_mod.log(log_dict, step=0)
        except Exception:
            pass

    # ---- append results.jsonl ----
    ensure_dir(Path(args.output_dir))
    with (Path(args.output_dir) / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(results, ensure_ascii=False) + "\n")

    # ---- delete generated images used for FID ----
    if not args.fid_keep_gen:
        shutil.rmtree(gen_dir, ignore_errors=True)
        print(f"[FID] Deleted gen_tmp -> {gen_dir}", flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ------------------------- Model discovery helpers -------------------------

def discover_model_dirs(root: Path) -> List[Path]:
    cands = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        if (d / "adapter_config.json").exists():
            cands.append(d)
            continue
        if (d / "config.json").exists() and (
            (d / "diffusion_pytorch_model.bin").exists() or (d / "diffusion_pytorch_model.safetensors").exists()
        ):
            cands.append(d)
            continue
    uniq = []
    seen = set()
    for p in cands:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return sorted(uniq, key=lambda x: x.as_posix())

def sanitize_tag(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._\-]+", "_", s)
    return s[:160] if len(s) > 160 else s


# ------------------------- Main -------------------------

def main():
    p = argparse.ArgumentParser("Generate samples + compute custom-Inception FID (overall + per-class)")

    # Models
    p.add_argument("--model_dirs", type=str, nargs="*", default=[])
    p.add_argument("--model_root", type=str, default="")
    p.add_argument("--base_model_dir", type=str, default="ddpm_cifar10_rgb_T400_DDIM50/ckpt_step150000", help="Required for LoRA adapters: base UNet directory.")
    p.add_argument("--lora_merge", action="store_true")

    # Data / output
    p.add_argument("--test_dir", type=str, default="cifar10_png_linear_only/gray3/test")
    p.add_argument("--output_dir", type=str, default="eval_fid_out")
    p.add_argument("--fid_cache_dir", type=str, default="")
    p.add_argument("--fid_symlink_real", action="store_true")
    p.add_argument("--fid_copy_real", action="store_true")

    # Device / precision
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # Sampling
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--sample_eta", type=float, default=0.0)
    p.add_argument("--grid_n", type=int, default=36)
    p.add_argument("--seed", type=int, default=42)

    # Scheduler fallback
    p.add_argument("--train_timesteps", type=int, default=400)
    p.add_argument("--beta_schedule", type=str, default="linear")
    p.add_argument("--prediction_type", type=str, default="epsilon", choices=["epsilon", "sample", "v_prediction"])

    # FID
    p.add_argument("--disable_fid", action="store_true")
    p.add_argument("--fid_per_class", action="store_true")
    p.set_defaults(fid_per_class=True)
    p.add_argument("--fid_num_samples", type=int, default=0, help="0 => use ALL real images; else generate this many samples.")
    p.add_argument("--fid_gen_batch", type=int, default=512)
    p.add_argument("--fid_batch_size", type=int, default=64)
    p.add_argument("--fid_num_workers", type=int, default=8)
    p.add_argument("--fid_keep_gen", action="store_true")

    # Custom Inception for FID
    p.add_argument("--inception_ckpt", type=str, default="0102_inceptionv3_sgd_gray3/ckpts/best.pt", help="Path to YOUR trained InceptionV3 checkpoint.")
    p.add_argument("--inception_num_classes", type=int, default=1000)
    p.add_argument("--inception_input_size", type=int, default=299)
    p.add_argument("--inception_mean", type=float, nargs=3, default=[0.45798322587856827] * 3)
    p.add_argument("--inception_std", type=float, nargs=3, default=[0.2623006911570552] * 3)

    # W&B
    p.add_argument("--wandb", action="store_false")
    p.add_argument("--wandb_offline", action="store_true")
    p.add_argument("--project", type=str, default="cifar10-gray3-customfid")
    p.add_argument("--run_name", type=str, default="eval")

    args = p.parse_args()

    # device
    try:
        device = torch.device(args.device)
    except Exception:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device.index)

    # W&B
    wandb_run = None
    wandb_mod = None
    if args.wandb:
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"
        try:
            import wandb as _wandb
            wandb_mod = _wandb
            wandb_run = wandb_mod.init(project=args.project, name=args.run_name, config=vars(args))
        except Exception as e:
            print(f"[Warn] wandb init failed; continue without wandb. ({e})", flush=True)
            wandb_run = None
            wandb_mod = None

    # resolve model list
    model_dirs: List[Path] = [Path(x) for x in args.model_dirs]
    if args.model_root:
        root = Path(args.model_root)
        if root.exists():
            model_dirs.extend(discover_model_dirs(root))

    # dedup + existence
    uniq = []
    seen = set()
    for md in model_dirs:
        if not md.exists():
            print(f"[Skip] Not found: {md}", flush=True)
            continue
        rp = str(md.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(md)
    model_dirs = uniq

    if len(model_dirs) == 0:
        raise SystemExit("No valid --model_dirs and no models discovered under --model_root.")

    print(f"[Info] Device={device} | mixed_precision={args.mixed_precision}", flush=True)
    print(f"[Info] Models to evaluate: {len(model_dirs)}", flush=True)
    for md in model_dirs:
        print(f"  - {md}", flush=True)

    # prepare real caches
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"--test_dir not found: {test_dir}")

    cache_root = Path(args.fid_cache_dir) if args.fid_cache_dir else (Path(args.output_dir) / "fid_real_cache")
    ensure_dir(cache_root)

    use_symlink = True
    if args.fid_copy_real:
        use_symlink = False
    if args.fid_symlink_real:
        use_symlink = True

    real_all_cache = cache_root / "all"
    num_real_all = flatten_real_cache(test_dir, real_all_cache, use_symlink=use_symlink)

    real_class_caches: Dict[str, Path] = {}
    class_dirs = list_class_dirs(test_dir)
    if args.fid_per_class and len(class_dirs) > 0:
        per_class_root = cache_root / "per_class"
        ensure_dir(per_class_root)
        for cd in sorted(class_dirs, key=lambda x: x.name):
            cname = cd.name
            ccache = per_class_root / cname
            flatten_real_cache(cd, ccache, use_symlink=use_symlink)
            real_class_caches[cname] = ccache

    print(f"[FID] Real cache ready: all={num_real_all} @ {real_all_cache}", flush=True)
    if args.fid_per_class:
        print(f"[FID] Per-class caches: {len(real_class_caches)}", flush=True)

    # prepare inception + real stats (once)
    inception = None
    real_all_stats = None
    real_class_stats: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}

    if not args.disable_fid:
        if not args.inception_ckpt:
            raise SystemExit("FID enabled but --inception_ckpt is empty. Provide your trained InceptionV3 checkpoint.")
        inception_ckpt = Path(args.inception_ckpt)
        if not inception_ckpt.exists():
            raise FileNotFoundError(f"--inception_ckpt not found: {inception_ckpt}")

        print(f"[FID] Loading custom InceptionV3 from: {inception_ckpt}", flush=True)
        inception = load_custom_inception(
            ckpt_path=inception_ckpt,
            num_classes=args.inception_num_classes,
            device=device,
        )

        print("[FID] Computing REAL stats (all) ...", flush=True)
        real_all_stats = compute_activation_stats_from_dir(
            img_dir=real_all_cache,
            inception=inception,
            device=device,
            batch_size=args.fid_batch_size,
            num_workers=args.fid_num_workers,
            input_size=args.inception_input_size,
            mean=args.inception_mean,
            std=args.inception_std,
        )
        print(f"[FID] REAL(all) stats ready: n={real_all_stats[2]}", flush=True)

        if args.fid_per_class and len(real_class_caches) > 0:
            for cname, cdir in real_class_caches.items():
                print(f"[FID] Computing REAL stats (class={cname}) ...", flush=True)
                real_class_stats[cname] = compute_activation_stats_from_dir(
                    img_dir=cdir,
                    inception=inception,
                    device=device,
                    batch_size=args.fid_batch_size,
                    num_workers=args.fid_num_workers,
                    input_size=args.inception_input_size,
                    mean=args.inception_mean,
                    std=args.inception_std,
                )
            print(f"[FID] REAL(per-class) stats ready: {len(real_class_stats)} classes", flush=True)
    else:
        print("[FID] --disable_fid set: grid-only mode.", flush=True)

    # evaluate each model
    for md in model_dirs:
        tag = sanitize_tag(md.name)
        if sum(1 for x in model_dirs if x.name == md.name) > 1:
            tag = sanitize_tag(md.parent.name + "__" + md.name)

        print(f"\n===== Evaluating: {tag} =====", flush=True)

        eval_one_model(
            model_dir=md,
            tag=tag,
            args=args,
            device=device,
            wandb_run=wandb_run,
            wandb_mod=wandb_mod,
            real_all_stats=real_all_stats,
            real_class_stats=real_class_stats,
            inception=inception,
        )

    if wandb_run is not None and wandb_mod is not None:
        try:
            wandb_mod.finish()
        except Exception:
            pass

    print("\n[Done] Evaluation finished. See:", flush=True)
    print(f"  - {Path(args.output_dir) / 'results.jsonl'}", flush=True)
    print(f"  - per-model grids under {Path(args.output_dir) / '<tag>/samples'}", flush=True)


if __name__ == "__main__":
    main()
