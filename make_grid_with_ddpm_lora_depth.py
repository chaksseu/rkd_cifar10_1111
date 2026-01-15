#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Save per-class grids for multiple student models using:
  RGB (teacher inversion) -> zT -> student denoise -> depth

Grid layout (UPDATED per request):
- "Vertical" layout: each CLASS is one ROW
- Each row has 3 columns: [RGB | GT | EST]

So overall grid has:
  rows = #classes
  cols = 3

Determinism:
- One sample per class is selected deterministically by --seed
- Teacher inversion zT is computed ONCE on that RGB batch
- Every student uses the SAME zT (same seed condition) for denoising

Deps:
  pip install diffusers torch torchvision peft pillow numpy
"""

import os
import re
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DDIMInverseScheduler
from peft import PeftModel


# ------------------------- Utils -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int):
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_device(device_str: str) -> torch.device:
    try:
        dev = torch.device(device_str)
    except Exception:
        print(f"[Warn] Invalid --device '{device_str}', fallback cpu", flush=True)
        return torch.device("cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev

def to_grid(images: torch.Tensor, nrow: int) -> Image.Image:
    # images: (N,3,H,W) in [-1,1]
    imgs = (images.clamp(-1, 1) + 1) / 2.0
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2)
    grid = (grid * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(grid)

def sanitize_tag(s: str) -> str:
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    return s[:180]

def stable_choice_index(seed: int, key: str, n: int) -> int:
    h = hashlib.md5(f"{seed}_{key}".encode("utf-8")).hexdigest()
    v = int(h, 16)
    return v % max(1, n)

def load_teacher_scheduler_or_fallback(teacher_dir: Path, train_timesteps: int, beta_schedule: str) -> DDPMScheduler:
    try:
        return DDPMScheduler.from_pretrained(teacher_dir.as_posix())
    except Exception:
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


# ------------------------- Paired loading -------------------------

def _resize_tensor(x: torch.Tensor, image_size: int, mode: str) -> torch.Tensor:
    # x: (1,H,W)
    if x.shape[-2:] == (image_size, image_size):
        return x
    x2 = F.interpolate(
        x.unsqueeze(0),
        size=(image_size, image_size),
        mode=mode,
        align_corners=False if mode in ["bilinear", "bicubic"] else None,
    ).squeeze(0)
    return x2

def load_paired_item(
    rgb_path: Path,
    depth_path: Path,
    rgb_root: Path,
    image_size: int,
    use_depth_raw_npy: bool,
    use_depth_npy: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
    """
    Returns:
      x_rgb:    (3,H,W) in [-1,1]
      gt_depth: (1,H,W) float32 (raw units or [0,1])
      gt_mask:  (1,H,W) float32 {0,1}
      rel:      relative path string
      cls:      class name (first component of rel)
    """
    rel = str(rgb_path.relative_to(rgb_root)).replace("\\", "/")
    cls = rel.split("/")[0] if "/" in rel else "unknown"

    tf_rgb = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
    ])
    tf_dep = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),  # for L: (1,H,W) in [0,1]
    ])

    with Image.open(rgb_path) as im_rgb:
        im_rgb = im_rgb.convert("RGB")
        x_rgb01 = tf_rgb(im_rgb)
        x_rgb = x_rgb01 * 2.0 - 1.0

    # (1) raw + mask npy
    if use_depth_raw_npy:
        rawp = Path(depth_path.as_posix() + ".raw.npy")
        maskp = Path(depth_path.as_posix() + ".mask.npy")
        if rawp.exists() and maskp.exists():
            raw = np.load(rawp.as_posix()).astype(np.float32)  # (H,W)
            msk = np.load(maskp.as_posix())                    # (H,W)
            raw_t = torch.from_numpy(raw).unsqueeze(0)         # (1,H,W)
            msk_t = torch.from_numpy(msk.astype(np.float32)).unsqueeze(0)

            raw_t = _resize_tensor(raw_t, image_size, mode="bilinear")
            msk_t = _resize_tensor(msk_t, image_size, mode="nearest")
            msk_t = (msk_t > 0.5).to(torch.float32)
            msk_t = msk_t * (raw_t > 0.0).to(torch.float32)
            return x_rgb, raw_t, msk_t, rel, cls

    # (2) normalized .png.npy
    if use_depth_npy:
        npy = Path(depth_path.as_posix() + ".npy")
        if npy.exists():
            arr = np.load(npy.as_posix()).astype(np.float32)   # (H,W) in [0,1]
            x_d = torch.from_numpy(arr).unsqueeze(0)           # (1,H,W)
            x_d = _resize_tensor(x_d, image_size, mode="bilinear").clamp(0.0, 1.0)
            msk = (x_d > 0.0).to(torch.float32)
            return x_rgb, x_d, msk, rel, cls

    # (3) PNG
    with Image.open(depth_path) as im_d:
        im_d = im_d.convert("L")
        x_d01 = tf_dep(im_d)
    msk = (x_d01 > 0.0).to(torch.float32)
    return x_rgb, x_d01, msk, rel, cls


def build_one_per_class_batch(
    eval_rgb_dir: str,
    eval_depth_dir: str,
    image_size: int,
    seed: int,
    use_depth_raw_npy: bool,
    use_depth_npy: bool,
    class_order: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]:
    """
    Select one sample per class deterministically (seeded) and return a single batch:
      x_rgb:   (C,3,H,W) in [-1,1]
      gt:      (C,1,H,W)
      mask:    (C,1,H,W)
      classes: list length C (ordered)
      rels:    list length C
    """
    rgb_root = Path(eval_rgb_dir)
    dep_root = Path(eval_depth_dir)
    exts = {".png", ".jpg", ".jpeg"}

    rgb_files = sorted([p for p in rgb_root.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    pairs_by_class: Dict[str, List[Tuple[Path, Path]]] = {}

    for rp in rgb_files:
        rel = rp.relative_to(rgb_root)
        dp = dep_root / rel
        if not dp.exists():
            continue
        rel_str = str(rel).replace("\\", "/")
        cls = rel_str.split("/")[0] if "/" in rel_str else "unknown"
        pairs_by_class.setdefault(cls, []).append((rp, dp))

    if len(pairs_by_class) == 0:
        raise FileNotFoundError(f"No paired files found. eval_rgb_dir={eval_rgb_dir}, eval_depth_dir={eval_depth_dir}")

    classes = sorted(pairs_by_class.keys()) if (class_order is None or len(class_order) == 0) else class_order
    classes = [c for c in classes if c in pairs_by_class]
    if len(classes) == 0:
        raise RuntimeError("No valid classes found after applying --class_order.")

    xs_rgb, xs_gt, xs_m, rels_out, classes_out = [], [], [], [], []

    for cls in classes:
        plist = pairs_by_class[cls]
        idx = stable_choice_index(seed, cls, len(plist))
        rp, dp = plist[idx]

        x_rgb, gt, msk, rel, _ = load_paired_item(
            rgb_path=rp,
            depth_path=dp,
            rgb_root=rgb_root,
            image_size=image_size,
            use_depth_raw_npy=use_depth_raw_npy,
            use_depth_npy=use_depth_npy,
        )
        xs_rgb.append(x_rgb)
        xs_gt.append(gt)
        xs_m.append(msk)
        rels_out.append(rel)
        classes_out.append(cls)

    x_rgb_b = torch.stack(xs_rgb, dim=0)   # (C,3,H,W)
    gt_b    = torch.stack(xs_gt, dim=0)    # (C,1,H,W)
    m_b     = torch.stack(xs_m, dim=0)     # (C,1,H,W)
    return x_rgb_b, gt_b, m_b, classes_out, rels_out


# ------------------------- Teacher inversion & Student denoise -------------------------

@torch.no_grad()
def invert_x0_to_zT_ddim_inverse_epspred(
    model: UNet2DModel,
    ddim: DDIMScheduler,
    x0: torch.Tensor,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    inv = DDIMInverseScheduler.from_config(ddim.config)
    inv.set_timesteps(int(steps), device=device)

    model.eval()
    xt = x0.to(device)
    for t in inv.timesteps:
        x_in = inv.scale_model_input(xt, t)
        eps = model(x_in, t).sample
        xt = inv.step(eps, t, xt).prev_sample
    return xt

@torch.no_grad()
def predx0_from_z_ddim(
    model,
    ddim: DDIMScheduler,
    z: torch.Tensor,
    steps: int,
    eta: float,
    device: torch.device,
) -> torch.Tensor:
    local = DDIMScheduler.from_config(ddim.config)
    local.set_timesteps(int(steps), device=device)

    x = z.to(device)
    model.eval()
    out = None
    for t in local.timesteps:
        x_in = local.scale_model_input(x, t)
        eps = model(x_in, t).sample
        out = local.step(model_output=eps, timestep=t, sample=x, eta=float(eta))
        x = out.prev_sample
    return out.pred_original_sample  # (B,3,H,W) in [-1,1]

def pred_depth_1ch_from_student_x0(x0_pred_3ch: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x01 = (x0_pred_3ch.clamp(-1, 1) + 1.0) * 0.5
    d1 = x01.mean(dim=1, keepdim=True)
    return d1.clamp_min(eps)

def solve_scale_shift_lstsq(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12):
    B = pred.shape[0]
    p = pred.view(B, -1).to(torch.float64)
    d = gt.view(B, -1).to(torch.float64)
    m = (mask.view(B, -1) > 0.5)

    s_out = torch.zeros((B, 1, 1, 1), device=pred.device, dtype=torch.float64)
    t_out = torch.zeros((B, 1, 1, 1), device=pred.device, dtype=torch.float64)

    for i in range(B):
        mi = m[i]
        n = int(mi.sum().item())
        if n < 2:
            s_out[i] = 0.0
            t_out[i] = 0.0
            continue

        pv = p[i, mi]
        dv = d[i, mi]

        sum_p2 = (pv * pv).sum()
        sum_p  = pv.sum()
        sum_d  = dv.sum()
        sum_pd = (pv * dv).sum()
        n_t = torch.tensor(float(n), device=pred.device, dtype=torch.float64)

        denom = (sum_p2 * n_t - sum_p * sum_p)
        if torch.abs(denom) < eps:
            s = torch.tensor(0.0, device=pred.device, dtype=torch.float64)
            t = (sum_d / n_t)
        else:
            s = (n_t * sum_pd - sum_p * sum_d) / denom
            t = (sum_p2 * sum_d - sum_p * sum_pd) / denom

        s_out[i, 0, 0, 0] = s
        t_out[i, 0, 0, 0] = t

    return s_out.to(torch.float32), t_out.to(torch.float32)

def depth_to_vis_3ch_m11(depth_1ch: torch.Tensor, mask_1ch: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    depth_1ch: (B,1,H,W)
    mask_1ch:  (B,1,H,W)
    returns:   (B,3,H,W) in [-1,1], per-image minmax over valid pixels
    """
    B = depth_1ch.shape[0]
    d = depth_1ch.clone()
    m = (mask_1ch > 0.5)
    out = torch.zeros((B, 3, d.shape[-2], d.shape[-1]), device=d.device, dtype=d.dtype)

    for i in range(B):
        vi = m[i, 0]
        if vi.sum() <= 0:
            di01 = torch.zeros_like(d[i, 0])
        else:
            dv = d[i, 0][vi]
            mn = dv.min()
            mx = dv.max()
            if (mx - mn) < eps:
                di01 = torch.zeros_like(d[i, 0])
            else:
                di01 = (d[i, 0] - mn) / (mx - mn)
                di01 = di01.clamp(0.0, 1.0)
        vis = di01.unsqueeze(0).repeat(3, 1, 1) * 2.0 - 1.0
        out[i] = vis
    return out.clamp(-1, 1)


# ------------------------- Student loading -------------------------

def is_lora_dir(p: Path) -> bool:
    return (p / "adapter_config.json").exists() or (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists()

def load_student_from_ckpt(teacher_dir: Path, student_ckpt: Path, device: torch.device) -> torch.nn.Module:
    """
    - If LoRA adapter dir: load base UNet from teacher_dir, then PeftModel.from_pretrained(base, adapter).
    - Else: load UNet2DModel directly from student_ckpt.
    """
    if is_lora_dir(student_ckpt):
        base = UNet2DModel.from_pretrained(teacher_dir.as_posix()).to(device)
        base.requires_grad_(False)
        student = PeftModel.from_pretrained(base, student_ckpt.as_posix(), is_trainable=False).to(device)
        student.eval()
        return student
    else:
        student = UNet2DModel.from_pretrained(student_ckpt.as_posix()).to(device)
        student.eval()
        return student


# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser("Save per-class vertical grids: each CLASS is a row with [RGB | GT | EST].")

    ap.add_argument("--teacher_dir", type=str, required=True)
    ap.add_argument("--student_ckpts", type=str, nargs="+", required=True)

    ap.add_argument("--eval_rgb_dir", type=str, required=True)
    ap.add_argument("--eval_depth_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    ap.add_argument("--gen_amp", action="store_true", help="Enable autocast for student generation (NOT inversion).")

    ap.add_argument("--image_size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.0)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--train_timesteps", type=int, default=400)
    ap.add_argument("--beta_schedule", type=str, default="linear")

    ap.add_argument("--use_depth_raw_npy", action="store_true", default=True)
    ap.add_argument("--use_depth_npy", action="store_true", default=True)

    ap.add_argument("--affine_align", action="store_true", default=True)
    ap.add_argument("--depth_eps", type=float, default=1e-6)

    ap.add_argument("--class_order", type=str, nargs="*", default=None,
                    help="Optional explicit class order list. If omitted, uses sorted folder names.")

    args = ap.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index)

    out_root = Path(args.output_dir)
    ensure_dir(out_root)

    teacher_dir = Path(args.teacher_dir)
    teacher = UNet2DModel.from_pretrained(teacher_dir.as_posix()).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    ddpm = load_teacher_scheduler_or_fallback(teacher_dir, args.train_timesteps, args.beta_schedule)
    ddim_T = make_ddim(ddpm, prediction_type="epsilon")
    ddim_S = make_ddim(ddpm, prediction_type="epsilon")

    # 1) select one per class (deterministic), build a batch
    x_rgb, gt_depth, gt_mask, classes, rels = build_one_per_class_batch(
        eval_rgb_dir=args.eval_rgb_dir,
        eval_depth_dir=args.eval_depth_dir,
        image_size=args.image_size,
        seed=args.seed,
        use_depth_raw_npy=args.use_depth_raw_npy,
        use_depth_npy=args.use_depth_npy,
        class_order=args.class_order,
    )

    gt_mask = gt_mask * (gt_depth > args.depth_eps).to(torch.float32)

    C = x_rgb.shape[0]
    print(f"[Info] Selected {C} classes:", classes, flush=True)
    for c, r in zip(classes, rels):
        print(f"  - {c}: {r}", flush=True)

    x_rgb = x_rgb.to(device, non_blocking=True)
    gt_depth = gt_depth.to(device, non_blocking=True)
    gt_mask = gt_mask.to(device, non_blocking=True)

    # 2) invert teacher on RGB ONCE -> zT (reused for all students)
    with torch.no_grad():
        zT = invert_x0_to_zT_ddim_inverse_epspred(
            model=teacher, ddim=ddim_T, x0=x_rgb, steps=args.steps, device=device
        )

    # autocast context for student generation only
    use_amp = (device.type == "cuda") and (args.mixed_precision != "no") and args.gen_amp
    amp_dtype = None
    if use_amp:
        amp_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16

    gen_autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
        if use_amp else nullcontext()
    )

    # GT visualization once (per-image minmax)
    gt_vis = depth_to_vis_3ch_m11(gt_depth.clamp_min(args.depth_eps), gt_mask, eps=args.depth_eps)

    # 3) each student -> EST -> build "class-rows" grid and save
    for ckpt in args.student_ckpts:
        ckpt_p = Path(ckpt)
        tag = sanitize_tag(ckpt_p.as_posix())
        save_dir = out_root / tag
        ensure_dir(save_dir)

        print(f"[Model] {ckpt_p} -> {save_dir}", flush=True)

        student = load_student_from_ckpt(teacher_dir=teacher_dir, student_ckpt=ckpt_p, device=device)

        with torch.no_grad():
            with gen_autocast_ctx:
                x0_depth_pred_3ch = predx0_from_z_ddim(
                    model=student, ddim=ddim_S, z=zT, steps=args.steps, eta=args.eta, device=device
                )

        pred_m = pred_depth_1ch_from_student_x0(x0_depth_pred_3ch, eps=args.depth_eps)

        if args.affine_align:
            s, t = solve_scale_shift_lstsq(pred_m, gt_depth, gt_mask, eps=1e-12)
            pred_a = (pred_m * s + t).clamp_min(args.depth_eps)
        else:
            pred_a = pred_m

        est_vis = depth_to_vis_3ch_m11(pred_a, gt_mask, eps=args.depth_eps)

        # -------------------------
        # GRID (UPDATED):
        # Each CLASS is one ROW with 3 columns [RGB | GT | EST]
        # => order images as: (rgb_i, gt_i, est_i) for i=0..C-1, and set nrow=3
        # -------------------------
        rows = []
        for i in range(C):
            rows.append(x_rgb[i])
            rows.append(gt_vis[i])
            rows.append(est_vis[i])
        cat = torch.stack(rows, dim=0)  # (3C,3,H,W)

        grid = to_grid(cat, nrow=3)

        out_path = save_dir / f"rgb_gt_est_grid_classrows_C{C}_steps{args.steps}_eta{args.eta}_seed{args.seed}.png"
        grid.save(out_path)
        print(f"[Saved] {out_path}", flush=True)

        del student
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("[Done]", flush=True)


if __name__ == "__main__":
    main()
