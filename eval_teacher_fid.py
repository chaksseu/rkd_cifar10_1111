#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Measure FID (pytorch-fid) of ORIGINAL TEACHER diffusion model vs gray3/test.

- Loads teacher UNet2DModel from diffusers (.from_pretrained)
- Loads scheduler from the same dir if available (DDPMScheduler.from_pretrained), else fallback
- Samples images with DDIM (teacher predicts epsilon)
- Flattens real test images into a cache dir (symlink by default)
- Computes FID using pytorch-fid (InceptionV3, dims=2048)

Deps:
  pip install diffusers torch torchvision pytorch-fid pillow
"""

import os
import argparse
import shutil
from pathlib import Path
from typing import List, Optional
from contextlib import nullcontext

import torch
from PIL import Image

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler


# ------------------------- Utils -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def resolve_device(device_str: str) -> torch.device:
    try:
        dev = torch.device(device_str)
    except Exception:
        print(f"[Warn] Invalid --device '{device_str}'. Falling back to 'cpu'.", flush=True)
        return torch.device("cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA not available. Falling back to CPU.", flush=True)
        return torch.device("cpu")
    return dev

def set_seed(seed: int):
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collect_image_paths_recursive(root: Path, exts={".png", ".jpg", ".jpeg"}) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def save_tensor_batch_to_dir(x: torch.Tensor, out_dir: Path, start_idx: int):
    """
    x: (B,3,H,W) in [-1,1]
    """
    ensure_dir(out_dir)
    x01 = (x.clamp(-1, 1) + 1) * 0.5
    x255 = (x01 * 255.0).clamp(0, 255).byte().cpu()
    for i in range(x255.shape[0]):
        arr = x255[i].permute(1, 2, 0).numpy()
        Image.fromarray(arr).save(out_dir / f"gen_{start_idx + i:06d}.png")

def flatten_real_cache(test_dir: Path, cache_dir: Path, use_symlink: bool = True) -> int:
    """
    Flatten (possibly class-subfolder) images into a single directory for pytorch-fid.
    """
    ensure_dir(cache_dir)
    existing = list(cache_dir.glob("*"))
    if len(existing) > 0:
        return len(existing)

    paths = collect_image_paths_recursive(test_dir)
    print(f"[FID] Flattening test set ({len(paths)} imgs) -> {cache_dir}", flush=True)

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

def compute_fid_pytorch_fid(real_dir: Path, gen_dir: Path, device: torch.device, batch_size: int, dims: int) -> float:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    fid = calculate_fid_given_paths(
        [real_dir.as_posix(), gen_dir.as_posix()],
        batch_size=batch_size,
        device=device,
        dims=dims,
    )
    return float(fid)

def load_teacher_scheduler_or_fallback(teacher_dir: Path, train_timesteps: int, beta_schedule: str) -> DDPMScheduler:
    try:
        return DDPMScheduler.from_pretrained(teacher_dir.as_posix())
    except Exception:
        print("[Warn] Scheduler config not found in teacher_dir. Using fallback DDPMScheduler.", flush=True)
        return DDPMScheduler(
            num_train_timesteps=train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type="epsilon",
        )

def make_ddim(ddpm: DDPMScheduler, prediction_type: str = "epsilon") -> DDIMScheduler:
    ddim = DDIMScheduler.from_config(ddpm.config)
    ddim.config.clip_sample = False
    ddim.config.prediction_type = prediction_type
    return ddim


# ------------------------- Sampling (Teacher, epsilon-pred) -------------------------

@torch.no_grad()
def sample_images_ddim_epspred(
    model: UNet2DModel,
    ddim: DDIMScheduler,
    num_images: int,
    image_size: int,
    device: torch.device,
    steps: int,
    eta: float,
    generator: Optional[torch.Generator] = None,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Returns x0 in [-1,1], shape (num_images,3,H,W)
    """
    model.eval()
    local = DDIMScheduler.from_config(ddim.config)
    local.set_timesteps(steps, device=device)

    dtype = next(model.parameters()).dtype
    x = torch.randn((num_images, 3, image_size, image_size), device=device, dtype=dtype, generator=generator)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )

    with autocast_ctx:
        for t in local.timesteps:
            x_in = local.scale_model_input(x, t)
            eps = model(x_in, t).sample
            x = local.step(model_output=eps, timestep=t, sample=x, eta=eta, generator=generator).prev_sample

    return x


# ------------------------- Main -------------------------

def main(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    fid_root = out_dir / "fid"
    ensure_dir(fid_root)

    # 1) Real cache (flatten)
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(
            f"--test_dir not found: {test_dir}\n"
            f"Override example: --test_dir /path/to/gray3/test"
        )

    real_cache_dir = fid_root / "real_cache" / "all"
    num_real = flatten_real_cache(test_dir, real_cache_dir, use_symlink=not args.fid_no_symlink)
    print(f"[FID] Real cache ready: N={num_real} @ {real_cache_dir}", flush=True)

    # 2) Load teacher + scheduler
    teacher_dir = Path(args.teacher_dir)
    if not teacher_dir.exists():
        raise FileNotFoundError(
            f"--teacher_dir not found: {teacher_dir}\n"
            f"Override example: --teacher_dir /path/to/teacher_ckpt"
        )

    print(f"[Info] Loading Teacher UNet: {teacher_dir}", flush=True)
    teacher = UNet2DModel.from_pretrained(teacher_dir.as_posix()).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    ddpm = load_teacher_scheduler_or_fallback(teacher_dir, args.train_timesteps, args.beta_schedule)
    ddim = make_ddim(ddpm, prediction_type="epsilon")

    # 3) Decide sample count
    fid_num = num_real if args.fid_num_samples <= 0 else int(min(args.fid_num_samples, num_real))
    print(f"[Info] Will generate N={fid_num} images for FID.", flush=True)

    # 4) Generate images to disk
    gen_dir = fid_root / f"gen_teacher_ddim{args.sample_steps}_eta{args.sample_eta}_N{fid_num}"
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    ensure_dir(gen_dir)

    use_amp = (device.type == "cuda") and (args.mixed_precision != "no")
    amp_dtype = None
    if use_amp:
        amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    remaining = fid_num
    cursor = 0
    while remaining > 0:
        cur = min(args.gen_batch, remaining)
        xs = sample_images_ddim_epspred(
            model=teacher,
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
        if cursor % max(1, args.progress_every) == 0 or remaining == 0:
            print(f"[Gen] saved {cursor}/{fid_num} ...", flush=True)

    # 5) Compute FID
    print("[FID] Computing pytorch-fid ...", flush=True)
    fid = compute_fid_pytorch_fid(
        real_dir=real_cache_dir,
        gen_dir=gen_dir,
        device=device,
        batch_size=args.fid_batch_size,
        dims=args.fid_dims,
    )
    print(f"[RESULT] Teacher FID = {fid:.6f}  (real N={fid_num}, gen N={fid_num})", flush=True)

    # 6) Log to file
    with (out_dir / "teacher_fid_result.txt").open("a", encoding="utf-8") as f:
        f.write(
            f"teacher_dir={teacher_dir.as_posix()} test_dir={test_dir.as_posix()} "
            f"steps={args.sample_steps} eta={args.sample_eta} N={fid_num} "
            f"fid_dims={args.fid_dims} fid_batch={args.fid_batch_size} "
            f"FID={fid:.6f}\n"
        )

    if not args.keep_gen:
        shutil.rmtree(gen_dir, ignore_errors=True)
        print(f"[Info] Deleted gen_dir (use --keep_gen to keep): {gen_dir}", flush=True)


def build_argparser():
    p = argparse.ArgumentParser("Measure teacher FID vs gray3/test using pytorch-fid (defaults set to your paths)")

    # ---- Your path defaults (requested) ----
    p.add_argument(
        "--teacher_dir",
        type=str,
        default="ddpm_cifar10_rgb_T400_DDIM50/ckpt_step150000",
        help="Teacher UNet2DModel directory (diffusers from_pretrained)",
    )
    p.add_argument(
        "--test_dir",
        type=str,
        default="cifar10_png_linear_only/gray3/test",
        help="Real test images root (e.g., gray3/test)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="teacher_fid_gray3_test",
        help="Output directory",
    )

    # ---- Requested device default ----
    p.add_argument("--device", type=str, default="cuda:7")
    p.add_argument("--seed", type=int, default=42)

    # Sampling
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--sample_steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--sample_eta", type=float, default=0.0, help="DDIM eta (0.0 = deterministic)")

    # Generation
    p.add_argument("--gen_batch", type=int, default=256, help="How many images to generate per batch")
    p.add_argument("--progress_every", type=int, default=1024, help="Print progress every K saved images")

    # AMP
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # Scheduler fallback (only used if teacher_dir has no scheduler config)
    p.add_argument("--train_timesteps", type=int, default=400)
    p.add_argument("--beta_schedule", type=str, default="linear")

    # FID
    p.add_argument("--fid_batch_size", type=int, default=64)
    p.add_argument("--fid_dims", type=int, default=2048)
    p.add_argument("--fid_num_samples", type=int, default=10000, help="0 => use all real images; else min(this, num_real)")
    p.add_argument("--fid_no_symlink", action="store_true", help="Copy real images instead of symlink for cache")
    p.add_argument("--keep_gen", action="store_true", help="Keep generated images directory")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
