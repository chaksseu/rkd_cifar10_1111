#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate Depth Anything (Depth-Anything-V2-Small-hf) on a paired RGB->Depth test set
prepared in the "paper-style" format, and log evaluation metrics to Weights & Biases.

GT format priority:
  (1) <img>.png.raw.npy (float32 raw depth) + <img>.png.mask.npy (uint8 {0,1})
  (2) <img>.png.npy (float32 normalized [0,1])
  (3) <img>.png (L)

Protocol (Marigold/Lotus-style):
  1) Predict depth from RGB (DepthAnything V2 Small)
  2) Resize prediction to match evaluation resolution (image_size)
  3) Mask invalid pixels (mask AND gt_depth > min_depth_eval)
  4) Affine alignment per-image: gt ≈ s * pred + t (least squares on valid pixels)
  5) Compute AbsRel and δ1/δ2/δ3 on valid pixels only
  6) Log dataset averages + optional per-image CSV + optional visualization grids + WandB logging

Deps:
  pip install torch torchvision transformers pillow tqdm numpy wandb
"""

import os
import csv
import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

try:
    import wandb
except Exception as e:
    wandb = None
    _wandb_import_err = e


# ------------------------- Defaults (match your RKD argparser style) -------------------------

def build_argparser():
    p = argparse.ArgumentParser(
        "Eval DepthAnythingV2Small on paired RGB->Depth test set (affine-invariant, raw+mask preferred) + wandb"
    )

    DATE = "0111"
    CUDA_NUM = 4

    DEFAULT_RGB_ROOT = "/workspace/rkd_cifar10_1111/cifar10_png_linear_only/rgb"
    DEFAULT_DEPTH_ROOT = "/workspace/rkd_cifar10_1111/cifar10_png_linear_only/depth"

    # previously-required args -> defaults
    p.add_argument("--rgb_root", type=str, default=f"{DEFAULT_RGB_ROOT}/test",
                   help="RGB test root (e.g., .../rgb/test).")
    p.add_argument("--gt_depth_root", type=str, default=f"{DEFAULT_DEPTH_ROOT}/test",
                   help="GT depth test root with raw+mask (e.g., .../depth/test).")
    p.add_argument("--output_dir", type=str,
                   default=f"/workspace/rkd_cifar10_1111/{DATE}_eval_depthanything_v2_small_test",
                   help="Output dir for logs/vis/csv.")

    # model/device
    p.add_argument("--model_id", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--device", type=str, default=f"cuda:{CUDA_NUM}")

    # dataloader / resolution
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", type=int, default=32)

    # eval settings
    p.add_argument("--affine_align", action="store_true", default=True,
                   help="Affine align (scale+shift) pred->gt on valid pixels.")
    p.add_argument("--delta", type=float, default=1.25)
    p.add_argument("--depth_eps", type=float, default=1e-6,
                   help="Clamp eps for numerical stability (pred/gt).")
    p.add_argument("--min_depth_eval", type=float, default=1e-3,
                   help="Additional GT threshold for valid pixels (prevents AbsRel blow-up).")

    # GT loading toggles (default True)
    p.add_argument("--use_depth_raw_npy", action="store_true", default=True,
                   help="Prefer depth.png.raw.npy + depth.png.mask.npy when available.")
    p.add_argument("--use_depth_npy", action="store_true", default=True,
                   help="Fallback to depth.png.npy (normalized) if raw not present.")

    # speed/precision
    p.add_argument("--amp", action="store_true", help="Enable AMP for depth model forward.")
    p.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    # logging files
    p.add_argument("--save_per_image", action="store_true", help="Save per-image metrics CSV.")
    p.add_argument("--no_save_vis", action="store_true",
                   help="Disable saving one RGB/pred(raw)/pred(aligned)/GT grid (default: save).")
    p.add_argument("--vis_n", type=int, default=16, help="How many columns to visualize.")

    # wandb
    p.add_argument("--project", type=str, default=f"{DATE}_rkd-rgb2depth-cifar10")
    p.add_argument("--run_name", type=str, default=f"eval-depthanything-v2-small")
    p.add_argument("--wandb_offline", action="store_true", help="Use wandb offline mode.")
    p.add_argument("--wandb_entity", type=str, default="", help="Optional wandb entity.")
    p.add_argument("--wandb_group", type=str, default="", help="Optional wandb group.")
    p.add_argument("--wandb_tags", type=str, default="", help="Optional comma-separated tags.")
    p.add_argument("--no_wandb", action="store_true", help="Disable wandb entirely.")

    # progress logging cadence
    p.add_argument("--wandb_log_interval", type=int, default=1,
                   help="Log running metrics every N batches (>=1).")

    return p


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

def to_grid(images: torch.Tensor, nrow: int) -> Image.Image:
    """
    images: (N,3,H,W) in [-1,1]
    """
    imgs = (images.clamp(-1, 1) + 1) / 2.0
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2)
    grid = (grid * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(grid)

def depth_to_vis_3ch_m11(depth_1ch: torch.Tensor, mask_1ch: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    depth_1ch: (B,1,H,W) positive (raw / aligned)
    mask_1ch:  (B,1,H,W) {0,1}
    returns:   (B,3,H,W) in [-1,1] for visualization (per-image minmax over valid).
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


# ------------------------- Affine alignment + metrics -------------------------

def solve_scale_shift_lstsq(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12):
    """
    Per-image least squares fit:
      minimize || (s * pred + t) - gt ||^2 over mask==1 pixels

    pred, gt, mask: (B,1,H,W)
    Returns: s, t as (B,1,1,1)
    """
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

@torch.no_grad()
def compute_absrel_delta(
    pred: torch.Tensor,  # (B,1,H,W)
    gt: torch.Tensor,    # (B,1,H,W)
    mask: torch.Tensor,  # (B,1,H,W)
    delta: float = 1.25,
    depth_eps: float = 1e-6,
    reduce: str = "image",  # "image" or "pixel"
):
    """
    AbsRel = mean(|pred-gt|/gt) on valid pixels.
    delta accuracy: mean( max(pred/gt, gt/pred) < delta^k ) on valid pixels.
    """
    pred = pred.clamp_min(depth_eps)
    gt = gt.clamp_min(depth_eps)
    m = (mask > 0.5)
    B = pred.shape[0]

    ratio = torch.maximum(pred / gt, gt / pred)

    if reduce == "pixel":
        vp = m.sum().item()
        if vp <= 0:
            return None
        absrel = ((pred - gt).abs() / gt)[m].mean().item()
        d1 = (ratio < (delta))[m].float().mean().item()
        d2 = (ratio < (delta ** 2))[m].float().mean().item()
        d3 = (ratio < (delta ** 3))[m].float().mean().item()
        return {"absrel": absrel, "d1": d1, "d2": d2, "d3": d3,
                "valid_pixels": float(vp), "valid_images": float(B)}

    absrel_list, d1_list, d2_list, d3_list = [], [], [], []
    valid_imgs = 0
    valid_pixels = 0.0

    for i in range(B):
        mi = m[i]
        vp = mi.sum().item()
        if vp <= 0:
            continue
        valid_imgs += 1
        valid_pixels += float(vp)

        absrel_i = ((pred[i] - gt[i]).abs() / gt[i])[mi].mean().item()
        d1_i = (ratio[i] < (delta))[mi].float().mean().item()
        d2_i = (ratio[i] < (delta ** 2))[mi].float().mean().item()
        d3_i = (ratio[i] < (delta ** 3))[mi].float().mean().item()

        absrel_list.append(absrel_i)
        d1_list.append(d1_i)
        d2_list.append(d2_i)
        d3_list.append(d3_i)

    if valid_imgs == 0:
        return None

    return {"absrel": float(np.mean(absrel_list)),
            "d1": float(np.mean(d1_list)),
            "d2": float(np.mean(d2_list)),
            "d3": float(np.mean(d3_list)),
            "valid_pixels": float(valid_pixels),
            "valid_images": float(valid_imgs)}


# ------------------------- Dataset -------------------------

class PairedRGBDepthRawMaskDataset(Dataset):
    """
    Pairs:
      rgb_root/.../xxx.png
      gt_depth_root/.../xxx.png  (same rel path)

    GT loading priority:
      (1) depth.png.raw.npy + depth.png.mask.npy
      (2) depth.png.npy
      (3) depth.png (L)
    """
    def __init__(
        self,
        rgb_root: str,
        gt_depth_root: str,
        image_size: int = 32,
        use_depth_raw_npy: bool = True,
        use_depth_npy: bool = True,
    ):
        self.rgb_root = Path(rgb_root)
        self.depth_root = Path(gt_depth_root)
        exts = {".png", ".jpg", ".jpeg"}

        rgb_files = sorted([p for p in self.rgb_root.rglob("*") if p.is_file() and p.suffix.lower() in exts])
        pairs = []
        for rp in rgb_files:
            rel = rp.relative_to(self.rgb_root)
            dp = self.depth_root / rel
            if dp.exists():
                pairs.append((rp, dp, str(rel).replace("\\", "/")))
        if len(pairs) == 0:
            raise FileNotFoundError(f"No paired files found. rgb_root={self.rgb_root}, gt_depth_root={self.depth_root}")
        self.pairs = pairs

        self.S = int(image_size)
        self.use_depth_raw_npy = bool(use_depth_raw_npy)
        self.use_depth_npy = bool(use_depth_npy)

        self.tf_rgb_small = T.Compose([
            T.Resize(self.S, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
        ])
        self.tf_dep_small = T.Compose([
            T.Resize(self.S, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def _resize_tensor(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if x.shape[-2:] == (self.S, self.S):
            return x
        x2 = F.interpolate(
            x.unsqueeze(0),
            size=(self.S, self.S),
            mode=mode,
            align_corners=False if mode in ["bilinear", "bicubic"] else None,
        ).squeeze(0)
        return x2

    def __getitem__(self, idx: int):
        rp, dp, rel = self.pairs[idx]

        with Image.open(rp) as im_rgb:
            im_rgb = im_rgb.convert("RGB")
            im_rgb.load()
            rgb_pil = im_rgb.copy()

        x_rgb01 = self.tf_rgb_small(rgb_pil)          # (3,S,S) [0,1]
        x_rgb_small = x_rgb01 * 2.0 - 1.0             # [-1,1]

        # (1) raw+mask
        if self.use_depth_raw_npy:
            rawp = Path(dp.as_posix() + ".raw.npy")
            maskp = Path(dp.as_posix() + ".mask.npy")
            if rawp.exists() and maskp.exists():
                raw = np.load(rawp.as_posix()).astype(np.float32)
                msk = np.load(maskp.as_posix()).astype(np.uint8)
                raw_t = torch.from_numpy(raw).unsqueeze(0)  # (1,H,W)
                msk_t = torch.from_numpy(msk.astype(np.float32)).unsqueeze(0)

                raw_t = self._resize_tensor(raw_t, mode="bilinear")
                msk_t = self._resize_tensor(msk_t, mode="nearest")
                msk_t = (msk_t > 0.5).float()
                return rgb_pil, x_rgb_small, raw_t, msk_t, rel

        # (2) normalized npy
        if self.use_depth_npy:
            npy = Path(dp.as_posix() + ".npy")
            if npy.exists():
                arr = np.load(npy.as_posix()).astype(np.float32)
                x_d = torch.from_numpy(arr).unsqueeze(0)
                x_d = self._resize_tensor(x_d, mode="bilinear").clamp(0.0, 1.0)
                msk = (x_d > 0.0).float()
                return rgb_pil, x_rgb_small, x_d, msk, rel

        # (3) png fallback
        with Image.open(dp) as im_d:
            im_d = im_d.convert("L")
            x_d01 = self.tf_dep_small(im_d)  # (1,S,S) in [0,1]
        msk = (x_d01 > 0.0).float()
        return rgb_pil, x_rgb_small, x_d01, msk, rel


def collate_fn(batch):
    rgb_pils, x_rgbs, gts, masks, rels = zip(*batch)
    return list(rgb_pils), torch.stack(x_rgbs, 0), torch.stack(gts, 0), torch.stack(masks, 0), list(rels)


# ------------------------- WandB helpers -------------------------

def wandb_init_if_needed(args):
    if args.no_wandb:
        return None

    if wandb is None:
        raise RuntimeError(
            f"wandb import failed: {_wandb_import_err}\n"
            "Install with: pip install wandb"
        )

    mode = "offline" if args.wandb_offline else "online"
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()] if args.wandb_tags else None

    init_kwargs = dict(
        project=args.project,
        name=args.run_name,
        config=vars(args),
        mode=mode,
    )
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity
    if args.wandb_group:
        init_kwargs["group"] = args.wandb_group
    if tags is not None and len(tags) > 0:
        init_kwargs["tags"] = tags

    run = wandb.init(**init_kwargs)
    return run


# ------------------------- Main eval -------------------------

@torch.no_grad()
def main(args):
    run = wandb_init_if_needed(args)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index)

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "vis")
    summary_path = out_dir / "summary.txt"
    per_image_csv = out_dir / "per_image_metrics.csv"

    save_vis = (not args.no_save_vis)

    # model
    print(f"[Info] Loading model_id={args.model_id}", flush=True)
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_id).to(device)
    model.eval()

    # dataset
    ds = PairedRGBDepthRawMaskDataset(
        rgb_root=args.rgb_root,
        gt_depth_root=args.gt_depth_root,
        image_size=args.image_size,
        use_depth_raw_npy=bool(args.use_depth_raw_npy),
        use_depth_npy=bool(args.use_depth_npy),
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_fn,
        drop_last=False,
    )
    print(f"[Info] Pairs={len(ds)} | image_size={args.image_size}", flush=True)

    use_amp = (device.type == "cuda") and args.amp
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    absrel_img, d1_img, d2_img, d3_img = [], [], [], []
    absrel_pix, d1_pix, d2_pix, d3_pix = [], [], [], []

    rows: List[List] = []
    vis_done = False
    global_step = 0

    # write header
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"[Config]\n")
        f.write(f"model_id={args.model_id}\n")
        f.write(f"rgb_root={args.rgb_root}\n")
        f.write(f"gt_depth_root={args.gt_depth_root}\n")
        f.write(f"image_size={args.image_size}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"affine_align={int(args.affine_align)}\n")
        f.write(f"min_depth_eval={args.min_depth_eval}\n")
        f.write(f"depth_eps={args.depth_eps}\n")
        f.write(f"delta={args.delta}\n")
        f.write(f"use_depth_raw_npy={int(args.use_depth_raw_npy)}\n")
        f.write(f"use_depth_npy={int(args.use_depth_npy)}\n")
        f.write(f"save_vis={int(save_vis)}\n")
        f.write("\n")

    pbar = tqdm(loader, desc="EVAL", dynamic_ncols=True)
    for rgb_pils, x_rgb_small, gt_depth, gt_mask, rels in pbar:
        # move GT tensors -> device
        x_rgb_small = x_rgb_small.to(device, non_blocking=True)  # (B,3,S,S) [-1,1]
        gt_depth = gt_depth.to(device, non_blocking=True)        # (B,1,S,S)
        gt_mask = gt_mask.to(device, non_blocking=True)          # (B,1,S,S)

        # stable valid mask:
        gt_mask = gt_mask * (gt_depth > float(args.min_depth_eval)).float()

        # depth prediction
        inputs = processor(images=rgb_pils, return_tensors="pt")
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

        pred_depth = outputs.predicted_depth  # (B,H',W')

        S = int(args.image_size)
        pred = F.interpolate(
            pred_depth.unsqueeze(1).float(),
            size=(S, S),
            mode="bicubic",
            align_corners=False,
        ).clamp_min(float(args.depth_eps))

        if args.affine_align:
            s, t = solve_scale_shift_lstsq(pred, gt_depth, gt_mask, eps=1e-12)
            pred_a = (pred * s + t).clamp_min(float(args.depth_eps))
        else:
            pred_a = pred

        met_img = compute_absrel_delta(
            pred=pred_a, gt=gt_depth, mask=gt_mask,
            delta=float(args.delta), depth_eps=float(args.depth_eps), reduce="image"
        )
        met_pix = compute_absrel_delta(
            pred=pred_a, gt=gt_depth, mask=gt_mask,
            delta=float(args.delta), depth_eps=float(args.depth_eps), reduce="pixel"
        )

        if met_img is not None:
            absrel_img.append(met_img["absrel"])
            d1_img.append(met_img["d1"])
            d2_img.append(met_img["d2"])
            d3_img.append(met_img["d3"])

        if met_pix is not None:
            absrel_pix.append(met_pix["absrel"])
            d1_pix.append(met_pix["d1"])
            d2_pix.append(met_pix["d2"])
            d3_pix.append(met_pix["d3"])

        # per-image CSV rows
        if args.save_per_image and met_img is not None:
            ratio = torch.maximum(pred_a / gt_depth.clamp_min(args.depth_eps),
                                  gt_depth.clamp_min(args.depth_eps) / pred_a)
            B = pred_a.shape[0]
            for i in range(B):
                mi = (gt_mask[i] > 0.5)
                vp = mi.sum().item()
                if vp <= 0:
                    rows.append([rels[i], "", "", "", "", 0])
                    continue
                absrel_i = ((pred_a[i] - gt_depth[i]).abs() / gt_depth[i].clamp_min(args.depth_eps))[mi].mean().item()
                d1_i = (ratio[i] < (args.delta))[mi].float().mean().item()
                d2_i = (ratio[i] < (args.delta ** 2))[mi].float().mean().item()
                d3_i = (ratio[i] < (args.delta ** 3))[mi].float().mean().item()
                rows.append([rels[i], absrel_i, d1_i, d2_i, d3_i, int(vp)])

        # visualization (first batch)
        if save_vis and (not vis_done):
            vis_done = True
            n = min(int(args.vis_n), pred.shape[0])

            pred_vis = depth_to_vis_3ch_m11(pred[:n], gt_mask[:n], eps=float(args.depth_eps))
            preda_vis = depth_to_vis_3ch_m11(pred_a[:n], gt_mask[:n], eps=float(args.depth_eps))
            gt_vis = depth_to_vis_3ch_m11(gt_depth[:n].clamp_min(float(args.depth_eps)), gt_mask[:n], eps=float(args.depth_eps))

            cat = torch.cat([x_rgb_small[:n], pred_vis, preda_vis, gt_vis], dim=0)
            grid = to_grid(cat, nrow=n)
            grid_path = out_dir / "vis" / "rgb_pred_preda_gt.png"
            grid.save(grid_path)

            with summary_path.open("a", encoding="utf-8") as f:
                f.write(f"[VIS] saved={grid_path.as_posix()}\n\n")

            if run is not None:
                run.log({"eval/vis_rgb_pred_preda_gt": wandb.Image(grid_path.as_posix())}, step=global_step)

        # tqdm postfix + wandb running log
        if len(absrel_img) > 0:
            run_absrel = float(np.mean(absrel_img))
            run_d1 = float(np.mean(d1_img))
            pbar.set_postfix({"AbsRel(img)": f"{run_absrel:.4f}", "d1(img)": f"{run_d1:.4f}"})

            # if run is not None:
            #     if int(args.wandb_log_interval) < 1:
            #         args.wandb_log_interval = 1
            #     if (global_step % int(args.wandb_log_interval)) == 0:
            #         run.log(
            #             {
            #                 "eval/running_absrel_img": run_absrel,
            #                 "eval/running_d1_img": run_d1,
            #                 "eval/running_d2_img": float(np.mean(d2_img)) if len(d2_img) else float("nan"),
            #                 "eval/running_d3_img": float(np.mean(d3_img)) if len(d3_img) else float("nan"),
            #                 "eval/batch": global_step,
            #             },
            #             step=global_step,
            #         )

        global_step += 1

    if len(absrel_img) == 0:
        print("[Error] No valid images for metrics. Check masks/min_depth_eval.", flush=True)
        if run is not None:
            run.log({"eval/error_no_valid_images": 1}, step=global_step)
            run.finish()
        return

    # dataset averages
    absrel_m = float(np.mean(absrel_img))
    d1_m = float(np.mean(d1_img))
    d2_m = float(np.mean(d2_img))
    d3_m = float(np.mean(d3_img))

    absrel_p = float(np.mean(absrel_pix)) if len(absrel_pix) else float("nan")
    d1_p = float(np.mean(d1_pix)) if len(d1_pix) else float("nan")
    d2_p = float(np.mean(d2_pix)) if len(d2_pix) else float("nan")
    d3_p = float(np.mean(d3_pix)) if len(d3_pix) else float("nan")

    line = (
        f"[RESULT] (image-mean) AbsRel={absrel_m:.6f} d1={d1_m:.6f} d2={d2_m:.6f} d3={d3_m:.6f} | "
        f"(pixel-mean) AbsRel={absrel_p:.6f} d1={d1_p:.6f} d2={d2_p:.6f} d3={d3_p:.6f} | "
        f"affine_align={int(args.affine_align)} min_depth_eval={args.min_depth_eval} S={args.image_size}"
    )
    print(line, flush=True)

    with summary_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

    # save per-image CSV
    if args.save_per_image:
        with per_image_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rel_path", "absrel", "d1", "d2", "d3", "valid_pixels"])
            for r in rows:
                w.writerow(r)
        print(f"[Saved] per-image -> {per_image_csv.as_posix()}", flush=True)

    # wandb final logs
    if run is not None:
        final_metrics = {
            "eval/absrel_img_mean": absrel_m,
            "eval/d1_img_mean": d1_m,
            "eval/d2_img_mean": d2_m,
            "eval/d3_img_mean": d3_m,
            "eval/absrel_pix_mean": absrel_p,
            "eval/d1_pix_mean": d1_p,
            "eval/d2_pix_mean": d2_p,
            "eval/d3_pix_mean": d3_p,
            "eval/affine_align": int(args.affine_align),
            "eval/min_depth_eval": float(args.min_depth_eval),
            "eval/image_size": int(args.image_size),
            "eval/num_pairs": int(len(ds)),
        }
        run.log(final_metrics, step=global_step)

        # Optionally log per-image table
        if args.save_per_image and per_image_csv.exists():
            try:
                table = wandb.Table(columns=["rel_path", "absrel", "d1", "d2", "d3", "valid_pixels"])
                for r in rows:
                    table.add_data(*r)
                run.log({"eval/per_image_table": table}, step=global_step)
            except Exception as e:
                run.log({"eval/wandb_table_error": str(e)}, step=global_step)

        # Save local files into the run (works in offline too)
        try:
            wandb.save(summary_path.as_posix(), policy="now")
            if args.save_per_image and per_image_csv.exists():
                wandb.save(per_image_csv.as_posix(), policy="now")
            vis_path = out_dir / "vis" / "rgb_pred_preda_gt.png"
            if vis_path.exists():
                wandb.save(vis_path.as_posix(), policy="now")
        except Exception:
            pass

        run.finish()


if __name__ == "__main__":
    p = build_argparser()
    args = p.parse_args()

    ensure_dir(Path(args.output_dir))
    main(args)
