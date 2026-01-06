#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate FID sample images (ONLY generation, NO FID computation) for ALL model dirs under a root.

Supports:
  (A) Full UNet2DModel dir saved by diffusers (config.json + diffusion_pytorch_model.*)
  (B) LoRA adapter dir saved by PEFT (adapter_config.json / adapter_model.*) + base UNet dir

Behavior:
  - Discover model directories recursively under --model_root
  - For each model dir, generate N images via DDIM and save as flat PNG files:
        <output_dir>/<tag>/fid_gen/gen_000000.png ...
  - Optionally save a small grid image for sanity check:
        <output_dir>/<tag>/samples/grid_<tag>.png
  - Save generation metadata:
        <output_dir>/<tag>/gen_meta.json

Example:
  python generate_fid_samples_from_root.py \
    --model_root /workspace/1229_ddpm/ddpm_cifar10_gray3_T400_DDIM50_B32_LR1e-05_teacher_init_n10 \
    --base_model_dir /workspace/1229_ddpm/ddpm_cifar10_gray3_T400_DDIM50/ckpt_step150000 \
    --test_dir /workspace/cifar10_png_linear_only/gray3/test \
    --output_dir /workspace/eval_fid_gen_only \
    --device cuda:0 --mixed_precision fp16 \
    --sample_steps 50 --sample_eta 0.0 \
    --num_samples 0 --gen_batch 2048 \
    --seed 42

Notes:
  - If --num_samples=0, it will count real images under --test_dir recursively and generate that many.
  - If --test_dir is not provided and --num_samples=0, the script will error out.
"""

import os
import re
import json
import math
import shutil
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
from contextlib import nullcontext

import torch
import torchvision.utils as vutils
from PIL import Image

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler


# ------------------------- IO Utils -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sanitize_tag(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._\-]+", "_", s)
    return s[:160] if len(s) > 160 else s

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
    if root is None:
        return []
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def count_existing_gen_images(gen_dir: Path) -> int:
    if not gen_dir.exists():
        return 0
    return len([p for p in gen_dir.glob("gen_*.png") if p.is_file()])


# ------------------------- Model discovery -------------------------

def is_lora_adapter_dir(d: Path) -> bool:
    if not d.exists() or not d.is_dir():
        return False
    return (
        (d / "adapter_config.json").exists()
        or (d / "adapter_model.bin").exists()
        or (d / "adapter_model.safetensors").exists()
    )

def is_diffusers_unet_dir(d: Path) -> bool:
    if not d.exists() or not d.is_dir():
        return False
    if not (d / "config.json").exists():
        return False
    return (d / "diffusion_pytorch_model.bin").exists() or (d / "diffusion_pytorch_model.safetensors").exists()

def discover_model_dirs(root: Path) -> List[Path]:
    """
    Recursively discover candidate model directories.
    """
    cands: List[Path] = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        if is_lora_adapter_dir(d) or is_diffusers_unet_dir(d):
            cands.append(d)

    # de-dup by resolved path
    uniq: List[Path] = []
    seen = set()
    for p in cands:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)

    return sorted(uniq, key=lambda x: x.as_posix())


# ------------------------- Load UNet / Scheduler -------------------------

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

    if not is_diffusers_unet_dir(model_dir):
        raise ValueError(f"Not a recognized diffusers UNet dir: {model_dir}")

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


# ------------------------- Per-model generation -------------------------

def generate_for_one_model(model_dir: Path, tag: str, args, device: torch.device):
    # precision setup (sampling only)
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

    # output layout
    out_root = Path(args.output_dir) / tag
    samples_dir = out_root / "samples"
    gen_dir = out_root / "fid_gen"
    ensure_dir(samples_dir)
    ensure_dir(gen_dir)

    # handle overwrite / resume
    if args.overwrite and gen_dir.exists():
        shutil.rmtree(gen_dir, ignore_errors=True)
        ensure_dir(gen_dir)

    existing = count_existing_gen_images(gen_dir)
    if existing >= args.num_samples:
        print(f"[Skip] {tag}: already has {existing} images (>= {args.num_samples}) at {gen_dir}", flush=True)
        return

    # load model + scheduler
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

    # deterministic generator
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + (abs(hash(tag)) % 10_000))

    # save one grid (optional)
    if args.grid_n > 0:
        grid_n = int(args.grid_n)
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

    # generation loop (resume-safe)
    remaining = int(args.num_samples) - existing
    cursor = existing
    print(f"[Gen] {tag}: generating {remaining} more images (total target={args.num_samples}) -> {gen_dir}", flush=True)

    while remaining > 0:
        cur = min(int(args.gen_batch), remaining)
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

    # save meta
    meta = {
        "tag": tag,
        "model_dir": model_dir.as_posix(),
        "kind": kind,
        "base_model_dir": (base_model_dir.as_posix() if base_model_dir is not None else None),
        "lora_merge": bool(args.lora_merge),
        "device": str(device),
        "mixed_precision": str(args.mixed_precision),
        "image_size": int(args.image_size),
        "sample_steps": int(args.sample_steps),
        "sample_eta": float(args.sample_eta),
        "seed": int(args.seed),
        "num_samples": int(args.num_samples),
        "gen_batch": int(args.gen_batch),
        "scheduler_source": "from_pretrained(model_dir/base_model_dir) or fallback",
        "train_timesteps_fallback": int(args.train_timesteps),
        "beta_schedule_fallback": str(args.beta_schedule),
        "prediction_type_fallback": str(args.prediction_type),
    }
    with (out_root / "gen_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Done] {tag}: saved {args.num_samples} images at {gen_dir}", flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ------------------------- Main -------------------------

def main():
    p = argparse.ArgumentParser("Generate FID sample images for all model dirs under a root (NO FID computation).")

    # discovery
    p.add_argument("--model_root", type=str, default="/workspace/1229_ddpm/ddpm_cifar10_gray3_T400_DDIM50_B32_LR1e-05_teacher_init_n10_no_airplane_automobile_bird_deer_dog", help="Root folder to recursively discover model directories.")
    p.add_argument("--base_model_dir", type=str, default="ddpm_cifar10_rgb_T400_DDIM50/ckpt_step150000", help="Required for LoRA adapters: base UNet directory.")
    p.add_argument("--lora_merge", action="store_true", help="If set, try merge_and_unload() for LoRA.")

    # output
    p.add_argument("--output_dir", type=str, default="eval_fid_out_0105/1229_ddpm", help="Where to save generated images per model.")

    # device / precision
    p.add_argument("--device", type=str, default="cuda:6")
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # sampling
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--sample_eta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gen_batch", type=int, default=2048)
    p.add_argument("--grid_n", type=int, default=36, help="Save a grid for sanity check. Set 0 to disable.")
    p.add_argument("--overwrite", action="store_true", help="If set, delete existing fid_gen and regenerate.")

    # how many to generate
    p.add_argument("--num_samples", type=int, default=0, help="0 => match number of real images under --test_dir.")
    p.add_argument("--test_dir", type=str, default="cifar10_png_linear_only/gray3/test", help="Real dataset dir for counting images when --num_samples=0.")

    # scheduler fallback (only used if scheduler can't be loaded from model dirs)
    p.add_argument("--train_timesteps", type=int, default=400)
    p.add_argument("--beta_schedule", type=str, default="linear")
    p.add_argument("--prediction_type", type=str, default="epsilon", choices=["epsilon", "sample", "v_prediction"])

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

    # resolve num_samples
    if int(args.num_samples) <= 0:
        if not args.test_dir:
            raise SystemExit("--num_samples=0 requires --test_dir to count real images.")
        test_dir = Path(args.test_dir)
        if not test_dir.exists():
            raise FileNotFoundError(f"--test_dir not found: {test_dir}")
        n_real = len(collect_image_paths_recursive(test_dir))
        if n_real <= 0:
            raise SystemExit(f"No images found under --test_dir: {test_dir}")
        args.num_samples = int(n_real)
        print(f"[Info] --num_samples=0 -> counted real images: {args.num_samples} under {test_dir}", flush=True)

    # validate base_model_dir for LoRA usage (only if needed later; here we just warn early)
    if args.base_model_dir:
        b = Path(args.base_model_dir)
        if not b.exists():
            raise FileNotFoundError(f"--base_model_dir not found: {b}")

    # discover
    root = Path(args.model_root)
    if not root.exists():
        raise FileNotFoundError(f"--model_root not found: {root}")
    model_dirs = discover_model_dirs(root)
    if len(model_dirs) == 0:
        raise SystemExit(f"No model dirs discovered under: {root}")

    print(f"[Info] Device={device} | mixed_precision={args.mixed_precision}", flush=True)
    print(f"[Info] Discovered models: {len(model_dirs)}", flush=True)
    for md in model_dirs:
        print(f"  - {md}", flush=True)

    # output root
    ensure_dir(Path(args.output_dir))

    # generate
    name_counts = {}
    for md in model_dirs:
        base_tag = sanitize_tag(md.name)
        name_counts[base_tag] = name_counts.get(base_tag, 0) + 1

    for md in model_dirs:
        tag = sanitize_tag(md.name)
        if name_counts.get(tag, 0) > 1:
            tag = sanitize_tag(md.parent.name + "__" + md.name)

        print(f"\n===== Generate for: {tag} =====", flush=True)
        try:
            generate_for_one_model(model_dir=md, tag=tag, args=args, device=device)
        except Exception as e:
            print(f"[Error] {tag}: {e}", flush=True)
            # continue to next model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

    print("\n[Done] Generation finished.", flush=True)
    print(f"  - Output root: {Path(args.output_dir).resolve()}", flush=True)
    print("  - Each model: <output_dir>/<tag>/fid_gen/gen_*.png", flush=True)


if __name__ == "__main__":
    main()
