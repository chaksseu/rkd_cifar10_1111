#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate paired data with the SAME noise z for:
  - Teacher: RGB
  - Multiple Students: Depth (LoRA checkpoints)

Grid rows:
  Row 0: Teacher RGB
  Row i: Student_i Depth visualization (per-image minmax)

Saves:
  out_dir/
    rgb/000000.png ...
    depth_<tag>/000000.png ...
    depth_<tag>_npy/000000.npy ...
    grids/pair_grid_0000.png ...

Deps:
  pip install diffusers torch torchvision peft
"""

import os
import math
import argparse
from pathlib import Path
from contextlib import nullcontext
from typing import List, Tuple

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from peft import PeftModel


# ------------------------- Utils -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def resolve_device(device_str: str) -> torch.device:
    try:
        dev = torch.device(device_str)
    except Exception:
        return torch.device("cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev

def set_seed(seed: int):
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_grid(images: torch.Tensor, nrow: int) -> Image.Image:
    """
    images: (N,3,H,W) in [-1,1]
    """
    imgs = (images.clamp(-1, 1) + 1.0) * 0.5
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2)
    grid = (grid * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(grid)

def save_rgb_png(x_m11: torch.Tensor, path: Path):
    """
    x_m11: (3,H,W) in [-1,1]
    """
    x01 = (x_m11.clamp(-1, 1) + 1.0) * 0.5
    arr = (x01 * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(arr).save(path)

def x0_gray3_from_x0(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,C,H,W) with C in {1,3}, assumed [-1,1]
    return: (B,3,H,W) where 3 channels are identical
    """
    if x.shape[1] == 3:
        g = x.mean(dim=1, keepdim=True)
    elif x.shape[1] == 1:
        g = x
    else:
        raise ValueError(f"Unsupported channel count: {x.shape[1]}")
    return g.repeat(1, 3, 1, 1)

def pred_depth_1ch_from_student_x0(x0_pred_3ch: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x0_pred_3ch: (B,3,H,W) in [-1,1]
    -> (B,1,H,W) in (0,1]
    """
    x01 = (x0_pred_3ch.clamp(-1, 1) + 1.0) * 0.5
    d1 = x01.mean(dim=1, keepdim=True)
    return d1.clamp_min(eps)

def depth_to_vis_3ch_m11(depth_1ch: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    depth_1ch: (B,1,H,W) positive
    -> (B,3,H,W) in [-1,1] for visualization (per-image minmax)
    """
    B, _, H, W = depth_1ch.shape
    out = torch.zeros((B, 3, H, W), device=depth_1ch.device, dtype=depth_1ch.dtype)
    for i in range(B):
        d = depth_1ch[i, 0]
        mn = d.min()
        mx = d.max()
        if (mx - mn) < eps:
            d01 = torch.zeros_like(d)
        else:
            d01 = ((d - mn) / (mx - mn)).clamp(0, 1)
        out[i] = (d01.unsqueeze(0).repeat(3, 1, 1) * 2.0 - 1.0)
    return out.clamp(-1, 1)

def load_teacher_scheduler_or_fallback(teacher_dir: Path, train_timesteps: int, beta_schedule: str) -> DDPMScheduler:
    try:
        return DDPMScheduler.from_pretrained(teacher_dir.as_posix())
    except Exception:
        return DDPMScheduler(
            num_train_timesteps=train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type="epsilon",
        )

def make_ddim(ddpm: DDPMScheduler, prediction_type: str) -> DDIMScheduler:
    ddim = DDIMScheduler.from_config(ddpm.config)
    ddim.config.clip_sample = False
    ddim.config.prediction_type = prediction_type
    return ddim

@torch.no_grad()
def ddim_sample_x0(model: UNet2DModel, ddim: DDIMScheduler, z: torch.Tensor, steps: int, eta: float, device: torch.device):
    """
    DDIM sampling from noise z.
    Returns x0-like sample in [-1,1] (as produced by the sampler).
    """
    local = DDIMScheduler.from_config(ddim.config)
    local.set_timesteps(int(steps), device=device)

    x = z.to(device)
    model.eval()
    for t in local.timesteps:
        x_in = local.scale_model_input(x, t)
        eps = model(x_in, t).sample
        x = local.step(model_output=eps, timestep=t, sample=x, eta=float(eta)).prev_sample
    return x


def parse_student_specs(specs: List[str]) -> List[Tuple[str, str]]:
    """
    Parse --students entries.
    Accepts:
      - "tag=/path/to/lora"
      - "/path/to/lora" (auto tag: basename)
    Returns: [(tag, path), ...]
    """
    out = []
    for s in specs:
        s = s.strip()
        if not s:
            continue
        if "=" in s:
            tag, path = s.split("=", 1)
            tag = tag.strip()
            path = path.strip()
        else:
            path = s
            tag = Path(s).name
        tag = tag.replace(" ", "_")
        if tag == "":
            tag = "student"
        out.append((tag, path))
    if len(out) == 0:
        raise ValueError("No students provided. Use --students tag=/path OR /path ...")
    # ensure unique tags
    seen = {}
    uniq = []
    for tag, path in out:
        if tag not in seen:
            seen[tag] = 1
            uniq.append((tag, path))
        else:
            seen[tag] += 1
            uniq.append((f"{tag}_{seen[tag]}", path))
    return uniq


# ------------------------- Main -------------------------

def build_argparser():
    p = argparse.ArgumentParser("Generate paired RGB-DEPTH data (teacher + multiple students) with shared noise z")

    p.add_argument("--teacher_dir", type=str, required=True, help="Teacher RGB UNet2DModel checkpoint dir")
    p.add_argument(
        "--students",
        type=str,
        nargs="+",
        required=True,
        help='One or more student LoRA dirs. Format: "tag=/path/to/lora" or "/path/to/lora"'
    )
    p.add_argument("--output_dir", type=str, required=True)

    # generation config
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=64)

    # diffusion/sampling
    p.add_argument("--train_timesteps", type=int, default=400)
    p.add_argument("--beta_schedule", type=str, default="linear")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)

    # precision
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # saving / grid
    p.add_argument("--save_depth_png", action="store_true", default=True)
    p.add_argument("--save_depth_npy", action="store_true", default=True)
    p.add_argument("--grid_cols", type=int, default=8, help="Columns for grid (nrow)")
    p.add_argument("--depth_eps", type=float, default=1e-6)

    return p


def main(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index)

    student_specs = parse_student_specs(args.students)  # [(tag, path), ...]

    out_dir = Path(args.output_dir)
    rgb_dir = out_dir / "rgb"
    grid_dir = out_dir / "grids"
    ensure_dir(out_dir); ensure_dir(rgb_dir); ensure_dir(grid_dir)

    # per-student dirs
    student_dirs = {}
    for tag, _ in student_specs:
        d_png = out_dir / f"depth_{tag}"
        d_npy = out_dir / f"depth_{tag}_npy"
        if args.save_depth_png:
            ensure_dir(d_png)
        if args.save_depth_npy:
            ensure_dir(d_npy)
        student_dirs[tag] = (d_png, d_npy)

    # Load teacher
    teacher_dir = Path(args.teacher_dir)
    teacher = UNet2DModel.from_pretrained(teacher_dir.as_posix()).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Load student base (teacher init) once
    base_student = UNet2DModel.from_pretrained(teacher_dir.as_posix()).to(device)
    base_student.requires_grad_(False)

    # Load multiple students (LoRA)
    students = []
    for tag, lora_path in student_specs:
        m = PeftModel.from_pretrained(base_student, lora_path, is_trainable=False).to(device)
        m.eval()
        students.append((tag, m))

    # schedulers
    ddpm = load_teacher_scheduler_or_fallback(teacher_dir, args.train_timesteps, args.beta_schedule)
    ddim_T = make_ddim(ddpm, prediction_type="epsilon")
    ddim_S = make_ddim(ddpm, prediction_type="epsilon")

    # AMP
    use_amp = (device.type == "cuda") and (args.mixed_precision != "no")
    amp_dtype = None
    if use_amp:
        amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )

    N = int(args.num_samples)
    B = int(args.batch_size)
    H = W = int(args.image_size)

    # Deterministic noise stream
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))

    num_batches = math.ceil(N / B)
    global_idx = 0
    num_students = len(students)

    for bi in range(num_batches):
        cur = min(B, N - global_idx)
        if cur <= 0:
            break

        # Shared noise z
        z = torch.randn((cur, 3, H, W), device=device, generator=gen)

        # Teacher + all students from same z
        with torch.no_grad():
            with autocast_ctx:
                x0_rgb = ddim_sample_x0(teacher, ddim_T, z, steps=args.steps, eta=args.eta, device=device)  # (cur,3,H,W)

                depth_vis_rows = []
                depth_1ch_all = {}  # tag -> (cur,1,H,W)

                for tag, stu in students:
                    x0_dep = ddim_sample_x0(stu, ddim_S, z, steps=args.steps, eta=args.eta, device=device)  # (cur,3,H,W)
                    x0_dep = x0_gray3_from_x0(x0_dep)
                    depth_1ch = pred_depth_1ch_from_student_x0(x0_dep, eps=args.depth_eps)  # (cur,1,H,W)
                    depth_1ch_all[tag] = depth_1ch
                    depth_vis_rows.append(depth_to_vis_3ch_m11(depth_1ch))  # (cur,3,H,W)

        # Save per-sample (teacher rgb + each student depth)
        for j in range(cur):
            idx = global_idx + j
            rgb_path = rgb_dir / f"{idx:06d}.png"
            save_rgb_png(x0_rgb[j], rgb_path)

            for tag, _ in student_specs:
                d_png_dir, d_npy_dir = student_dirs[tag]

                if args.save_depth_png:
                    vis = depth_to_vis_3ch_m11(depth_1ch_all[tag][j:j+1]).squeeze(0)  # (3,H,W)
                    dep_path = d_png_dir / f"{idx:06d}.png"
                    save_rgb_png(vis, dep_path)

                if args.save_depth_npy:
                    npy_path = d_npy_dir / f"{idx:06d}.npy"
                    np.save(npy_path.as_posix(), depth_1ch_all[tag][j, 0].detach().float().cpu().numpy())

        # Grid: rows = 1 + num_students
        # Build as concatenation: [teacher row images] + [student1 row images] + ...
        rows = [x0_rgb] + depth_vis_rows
        cat = torch.cat(rows, dim=0)  # ((1+S)*cur, 3, H, W)

        nrow = min(int(args.grid_cols), cur)
        grid_img = to_grid(cat, nrow=nrow)
        grid_path = grid_dir / f"pair_grid_{bi:04d}.png"
        grid_img.save(grid_path)

        # Logging
        tags_str = ", ".join([t for t, _ in student_specs])
        print(f"[Batch {bi+1:04d}/{num_batches:04d}] idx [{global_idx}..{global_idx+cur-1}] "
              f"rows=1+{num_students} ({tags_str}) grid={grid_path}")

        global_idx += cur

    print(f"[Done] Generated {global_idx} paired samples with {num_students} students at: {out_dir}")


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
