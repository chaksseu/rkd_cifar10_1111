#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from PIL import Image

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler


# ------------------------- Utils -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def collect_image_paths_recursive(root: Path, exts={".png", ".jpg", ".jpeg"}) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def flatten_real_cache(test_dir: Path, cache_dir: Path, use_symlink: bool = True) -> int:
    ensure_dir(cache_dir)
    existing = list(cache_dir.glob("*"))
    if len(existing) > 0:
        return len(existing)

    paths = collect_image_paths_recursive(test_dir)
    print(f"[FID] Flattening real set ({len(paths)} imgs) -> {cache_dir}")
    for i, src in enumerate(paths, 1):
        dst = cache_dir / f"real_{i:06d}{src.suffix.lower()}"
        if use_symlink:
            try:
                os.symlink(src.resolve(), dst)
                continue
            except Exception:
                pass
        shutil.copy2(src, dst)
    return len(paths)

def save_tensor_batch_to_dir(x: torch.Tensor, out_dir: Path, start_idx: int):
    ensure_dir(out_dir)
    x01 = (x.clamp(-1, 1) + 1) / 2.0
    x255 = (x01 * 255.0).clamp(0, 255).byte().cpu()
    for i in range(x255.shape[0]):
        arr = x255[i].permute(1, 2, 0).numpy()
        Image.fromarray(arr).save(out_dir / f"gen_{start_idx + i:06d}.png")


# ------------------------- Sampling (DDIM) -------------------------

@torch.no_grad()
def sample_images_ddim(
    model: UNet2DModel,
    ddim_scheduler: DDIMScheduler,
    num_images: int,
    image_size: int,
    device: torch.device,
    steps: int,
    eta: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
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


# ------------------------- Summary Printer (+ Save) -------------------------

def build_summary_text(s: Dict[str, Any]) -> str:
    """
    Build a summary string (for printing and for saving to out_dir/summary.txt).
    """
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("FID EVALUATION SUMMARY")
    lines.append("=" * 72)

    lines.append("[Config]")
    lines.append(f"  ckpt_dir         : {s['ckpt_dir']}")
    lines.append(f"  test_dir         : {s['test_dir']}")
    lines.append(f"  out_dir          : {s['out_dir']}")
    lines.append(f"  device           : {s['device']}")
    lines.append(f"  fp16_autocast    : {s['fp16_autocast']}")
    lines.append(f"  seed             : {s['seed']}")
    lines.append(f"  image_size       : {s['image_size']}")

    lines.append("[Scheduler]")
    lines.append(f"  train_timesteps  : {s['train_timesteps']}")
    lines.append(f"  beta_schedule    : {s['beta_schedule']}")
    lines.append(f"  prediction_type  : {s['prediction_type']}")
    lines.append(f"  ddim_steps       : {s['ddim_steps']}")
    lines.append(f"  ddim_eta         : {s['ddim_eta']}")

    lines.append("[Data]")
    lines.append(f"  num_real_all     : {s['num_real_all']}")
    lines.append(f"  num_gen          : {s['num_gen']}")
    lines.append(f"  fid_dims         : {s['fid_dims']}")
    lines.append(f"  fid_batch_size   : {s['fid_batch_size']}")
    lines.append(f"  gen_batch        : {s['gen_batch']}")
    lines.append(f"  keep_gen         : {s['keep_gen']}")
    lines.append(f"  per_class        : {s['per_class']}")
    lines.append(f"  gen_dir          : {s['gen_dir']}")

    lines.append("[Results]")
    fid_all = s.get("fid_all", None)
    if fid_all is None:
        lines.append("  fid_overall      : (not computed)")
    else:
        lines.append(f"  fid_overall      : {fid_all:.4f}")

    per_class_fids: Dict[str, float] = s.get("fid_per_class", {}) or {}
    if len(per_class_fids) == 0:
        lines.append("  fid_per_class    : (none)")
    else:
        for cname in sorted(per_class_fids.keys()):
            lines.append(f"  fid_class/{cname:<12}: {per_class_fids[cname]:.4f}")
        vals = list(per_class_fids.values())
        mean_val = sum(vals) / max(1, len(vals))
        lines.append(f"  fid_per_class_mean: {mean_val:.4f}")

    lines.append("=" * 72)
    return "\n".join(lines) + "\n"

def save_text(path: Path, text: str):
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


# ------------------------- Main -------------------------

TT = 400
DDIM_STEPS = 200

def build_argparser():
    p = argparse.ArgumentParser("FID test for saved DDPM UNet (DDIM sampling)")

    p.add_argument("--ckpt_dir", type=str, default=f"./ddpm_cifar10_rgb_accel_T{TT}/last")
    p.add_argument("--test_dir", type=str, default="./cifar10_png_linear_only/rgb/test")
    p.add_argument("--out_dir", type=str, default=f"./fid_eval_T{TT}_STEP{DDIM_STEPS}")

    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--ddim_steps", type=int, default=DDIM_STEPS)
    p.add_argument("--ddim_eta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--train_timesteps", type=int, default=TT)
    p.add_argument("--beta_schedule", type=str, default="linear",
                   choices=["linear", "scaled_linear", "squaredcos_cap_v2"])
    p.add_argument("--prediction_type", type=str, default="epsilon",
                   choices=["epsilon", "sample", "v_prediction"])

    p.add_argument("--num_gen", type=int, default=0)
    p.add_argument("--gen_batch", type=int, default=2048)

    p.add_argument("--fid_batch_size", type=int, default=64)
    p.add_argument("--fid_dims", type=int, default=2048)
    p.add_argument("--per_class", action="store_false")
    p.add_argument("--keep_gen", action="store_true")
    p.add_argument("--no_symlink", action="store_true")

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--fp16", action="store_true")

    # NEW: summary path override (optional)
    p.add_argument("--summary_path", type=str, default="",
                   help="If set, write summary to this path; otherwise {out_dir}/summary.txt")

    return p


def main():
    args = build_argparser().parse_args()

    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except Exception as e:
        raise RuntimeError(f"pytorch-fid import failed: {e}\nInstall: pip install pytorch-fid")

    ckpt_dir = Path(args.ckpt_dir)
    test_dir = Path(args.test_dir)
    out_dir = Path(args.out_dir)

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"--ckpt_dir not found: {ckpt_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"--test_dir not found: {test_dir}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    fp16_autocast = bool(args.fp16 and device.type == "cuda")

    ensure_dir(out_dir)

    summary: Dict[str, Any] = {
        "ckpt_dir": str(ckpt_dir),
        "test_dir": str(test_dir),
        "out_dir": str(out_dir),
        "device": str(device),
        "fp16_autocast": fp16_autocast,
        "seed": int(args.seed),
        "image_size": int(args.image_size),
        "train_timesteps": int(args.train_timesteps),
        "beta_schedule": str(args.beta_schedule),
        "prediction_type": str(args.prediction_type),
        "ddim_steps": int(args.ddim_steps),
        "ddim_eta": float(args.ddim_eta),
        "fid_dims": int(args.fid_dims),
        "fid_batch_size": int(args.fid_batch_size),
        "gen_batch": int(args.gen_batch),
        "keep_gen": bool(args.keep_gen),
        "per_class": bool(args.per_class),
        "num_real_all": None,
        "num_gen": None,
        "gen_dir": None,
        "fid_all": None,
        "fid_per_class": {},
    }

    print(f"[Info] device={device}")

    # ---- Load model ----
    print(f"[Load] UNet2DModel from {ckpt_dir}")
    model = UNet2DModel.from_pretrained(ckpt_dir.as_posix()).to(device)

    # ---- Build schedulers ----
    ddpm = DDPMScheduler(
        num_train_timesteps=int(args.train_timesteps),
        beta_schedule=str(args.beta_schedule),
        prediction_type=str(args.prediction_type),
    )
    ddim = DDIMScheduler.from_config(ddpm.config)
    print(f"[Sched] train_timesteps(T)={args.train_timesteps}, beta_schedule={args.beta_schedule}, pred={args.prediction_type}")
    print(f"[Sched] DDIM steps={args.ddim_steps}, eta={args.ddim_eta}")

    # ---- Real caches ----
    fid_root = out_dir / "fid"
    real_root = fid_root / "real_cache"
    real_all = real_root / "all"
    real_per_class = real_root / "per_class"

    n_real_all = flatten_real_cache(test_dir, real_all, use_symlink=(not args.no_symlink))
    summary["num_real_all"] = int(n_real_all)
    print(f"[Real] all: {n_real_all} images")

    class_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
    class_names = [d.name for d in sorted(class_dirs, key=lambda x: x.name)]

    if args.per_class and len(class_names) > 0:
        for cd in sorted(class_dirs, key=lambda x: x.name):
            n_c = flatten_real_cache(cd, real_per_class / cd.name, use_symlink=(not args.no_symlink))
            print(f"[Real] class {cd.name}: {n_c} images")
    elif args.per_class:
        print("[Per-class] No class subdirectories found; skipping per-class FID.")

    # ---- Decide #gen ----
    num_gen = int(args.num_gen) if int(args.num_gen) > 0 else int(n_real_all)
    summary["num_gen"] = int(num_gen)
    print(f"[Gen] num_gen={num_gen}")

    gen_dir = fid_root / (
        f"gen_T{args.train_timesteps}_ddim{args.ddim_steps}_eta{args.ddim_eta}"
        f"_{args.beta_schedule}_{args.prediction_type}_seed{args.seed}"
    )
    summary["gen_dir"] = str(gen_dir)

    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    ensure_dir(gen_dir)

    # ---- Generate ----
    g = torch.Generator(device=device)
    g.manual_seed(int(args.seed))

    remaining = num_gen
    cursor = 0

    while remaining > 0:
        cur = min(int(args.gen_batch), remaining)
        with torch.no_grad():
            if fp16_autocast:
                with torch.autocast("cuda", enabled=True):
                    xs = sample_images_ddim(
                        model, ddim, cur, args.image_size, device,
                        steps=int(args.ddim_steps), eta=float(args.ddim_eta), generator=g
                    )
            else:
                xs = sample_images_ddim(
                    model, ddim, cur, args.image_size, device,
                    steps=int(args.ddim_steps), eta=float(args.ddim_eta), generator=g
                )
        save_tensor_batch_to_dir(xs, gen_dir, start_idx=cursor)
        cursor += cur
        remaining -= cur
        print(f"[Gen] saved {cursor}/{num_gen}")

    # ---- Overall FID ----
    fid_all = calculate_fid_given_paths(
        [real_all.as_posix(), gen_dir.as_posix()],
        batch_size=int(args.fid_batch_size),
        device=device,
        dims=int(args.fid_dims),
    )
    summary["fid_all"] = float(fid_all)
    print(f"[FID] overall: {fid_all:.4f}")

    # ---- Per-class FID ----
    per_class_fids: Dict[str, float] = {}
    if args.per_class and len(class_names) > 0:
        print("[FID] per-class:")
        for cname in class_names:
            real_c = real_per_class / cname
            if not real_c.exists():
                print(f"  - {cname}: (skip, cache missing)")
                continue
            fid_c = calculate_fid_given_paths(
                [real_c.as_posix(), gen_dir.as_posix()],
                batch_size=int(args.fid_batch_size),
                device=device,
                dims=int(args.fid_dims),
            )
            per_class_fids[cname] = float(fid_c)
            print(f"  - {cname}: {fid_c:.4f}")

    summary["fid_per_class"] = per_class_fids

    # ---- Cleanup ----
    if not args.keep_gen:
        shutil.rmtree(gen_dir, ignore_errors=True)
        print(f"[Cleanup] removed {gen_dir}")

    # ---- Final Summary (print + save) ----
    summary_text = build_summary_text(summary)
    print("\n" + summary_text, end="")

    summary_path = Path(args.summary_path) if args.summary_path.strip() else (out_dir / "summary.txt")
    save_text(summary_path, summary_text)
    print(f"[Summary] wrote: {summary_path}")

    print("[Done]")


if __name__ == "__main__":
    main()
