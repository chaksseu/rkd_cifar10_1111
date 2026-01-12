#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate depth datasets for:
  - Training: global-normalized depth in [0,1] saved as <img>.png.npy (float32)
  - Evaluation (paper-style): raw depth saved as <img>.png.raw.npy (float32) + valid mask <img>.png.mask.npy

Key differences vs old script:
  1) NO per-image minmax (it creates exact zeros -> AbsRel explodes).
  2) Save RAW depth floats for evaluation (do NOT evaluate on normalized/PNG).
  3) Normalize for training using GLOBAL percentiles computed on TRAIN split (p_low/p_high).
  4) Save mask to exclude invalid pixels (and optional depth range mask).

Intended evaluation protocol (Marigold/Lotus-style):
  - For each image, align prediction to GT with affine alignment (scale+shift) on valid pixels.
  - Compute AbsRel, δ1/δ2/δ3 on valid pixels only.
This script prepares the GT side correctly (raw + mask).
"""

import os
import glob
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# -----------------------------------------------------------------------------
# Dataset: returns (PIL_RGB, rel_path, (w,h))
# -----------------------------------------------------------------------------
class SplitImageDataset(Dataset):
    def __init__(self, split_root: str, exts=(".png", ".jpg", ".jpeg")):
        self.split_root = Path(split_root)
        paths = sorted(glob.glob(str(self.split_root / "**" / "*.*"), recursive=True))
        self.image_paths = [p for p in paths if p.lower().endswith(exts)]
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found under: {self.split_root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        p = Path(self.image_paths[idx])
        with Image.open(p) as img:
            img = img.convert("RGB")
            img.load()  # ensure loaded before file closes
            img = img.copy()
            w, h = img.size
        rel = os.path.relpath(p.as_posix(), self.split_root.as_posix())
        return img, rel.replace("\\", "/"), (w, h)


def collate_fn(batch):
    imgs, rels, sizes = zip(*batch)
    return list(imgs), list(rels), list(sizes)


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def resize_depth_to_size(depth_hw: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """depth_hw: (H',W') -> (h,w) on device"""
    d = depth_hw.unsqueeze(0).unsqueeze(0)  # (1,1,H',W')
    d = F.interpolate(d, size=(h, w), mode="bicubic", align_corners=False)
    return d.squeeze(0).squeeze(0)  # (h,w)

def out_paths_for(out_png: Path) -> Tuple[Path, Path, Path]:
    """
    For an output png path like xxx.png, return:
      raw:  xxx.png.raw.npy
      mask: xxx.png.mask.npy
      norm: xxx.png.npy
    """
    raw = Path(out_png.as_posix() + ".raw.npy")
    mask = Path(out_png.as_posix() + ".mask.npy")
    norm = Path(out_png.as_posix() + ".npy")
    return raw, mask, norm

def save_depth_png_uint16(norm01_hw: np.ndarray, out_png: Path):
    """Save normalized [0,1] depth visualization as uint16 PNG."""
    arr = (np.clip(norm01_hw, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    ensure_dir(out_png.parent)
    Image.fromarray(arr, mode="I;16").save(out_png)

def valid_mask_from_raw(raw_hw: np.ndarray, valid_min: float, finite_only: bool = True) -> np.ndarray:
    m = np.ones_like(raw_hw, dtype=np.uint8)
    if finite_only:
        m = (np.isfinite(raw_hw)).astype(np.uint8)
    if valid_min is not None:
        m = (m & (raw_hw > float(valid_min))).astype(np.uint8)
    return m

# -----------------------------------------------------------------------------
# Reservoir sampler for percentile stats (streaming)
# -----------------------------------------------------------------------------
class ReservoirSampler:
    """
    Keep up to max_samples float values (reservoir sampling).
    We sample k pixels per image from valid pixels to estimate global percentiles.
    """
    def __init__(self, max_samples: int, seed: int = 0):
        self.max_samples = int(max_samples)
        self.rng = np.random.default_rng(int(seed))
        self.buf: List[float] = []
        self.n_seen = 0

    def add_values(self, vals: np.ndarray):
        vals = vals.astype(np.float64).ravel()
        for v in vals:
            self.n_seen += 1
            if len(self.buf) < self.max_samples:
                self.buf.append(float(v))
            else:
                j = self.rng.integers(0, self.n_seen)
                if j < self.max_samples:
                    self.buf[int(j)] = float(v)

    def values(self) -> np.ndarray:
        if len(self.buf) == 0:
            return np.array([], dtype=np.float64)
        return np.array(self.buf, dtype=np.float64)

# -----------------------------------------------------------------------------
# Pass 1: generate RAW depth + MASK (and collect stats from TRAIN)
# -----------------------------------------------------------------------------
def generate_raw_and_masks(
    split_name: str,
    in_split_root: Path,
    out_split_root: Path,
    model,
    processor,
    device: torch.device,
    args,
    sampler: Optional[ReservoirSampler] = None,
):
    ds = SplitImageDataset(in_split_root.as_posix())
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_fn,
    )
    print(f"[Pass1] Split={split_name} | images={len(ds)} | in={in_split_root} | out={out_split_root}", flush=True)

    processed, skipped, failed = 0, 0, 0

    use_amp = (device.type == "cuda") and args.amp
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16

    pbar = tqdm(loader, desc=f"RAW[{split_name}]", dynamic_ncols=True)
    for images_pil, rel_paths, orig_sizes in pbar:
        out_png_paths: List[Path] = []
        need_mask: List[bool] = []

        for rel in rel_paths:
            relp = Path(rel)
            if args.output_ext:
                relp = relp.with_suffix("." + args.output_ext.lstrip("."))
            out_png = out_split_root / relp
            out_png_paths.append(out_png)

            raw_p, mask_p, norm_p = out_paths_for(out_png)
            if args.skip_existing and raw_p.exists() and mask_p.exists():
                need_mask.append(False)
            else:
                need_mask.append(True)

        if args.skip_existing and not any(need_mask):
            skipped += len(images_pil)
            pbar.set_postfix({"processed": processed, "skipped": skipped, "failed": failed})
            continue

        inputs = processor(images=images_pil, return_tensors="pt")
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            pred_depth = outputs.predicted_depth  # (B,H',W')

        for d, rel, (w, h), out_png, need in zip(pred_depth, rel_paths, orig_sizes, out_png_paths, need_mask):
            if not need:
                continue
            try:
                depth_hw = resize_depth_to_size(d, h=h, w=w)  # torch (h,w) on device
                depth_hw = depth_hw.float()

                raw_np = depth_hw.detach().cpu().numpy().astype(np.float32)
                mask_np = valid_mask_from_raw(raw_np, valid_min=args.valid_min, finite_only=True)

                raw_p, mask_p, _ = out_paths_for(out_png)
                ensure_dir(out_png.parent)
                np.save(raw_p.as_posix(), raw_np)
                np.save(mask_p.as_posix(), mask_np)

                # collect stats from train only
                if sampler is not None:
                    # sample pixels per image from valid area
                    valid_vals = raw_np[mask_np.astype(bool)]
                    if valid_vals.size > 0:
                        k = min(int(args.stat_pixels_per_image), int(valid_vals.size))
                        idx = np.random.choice(valid_vals.size, size=k, replace=False)
                        sampler.add_values(valid_vals[idx])

                processed += 1
            except Exception as e:
                failed += 1
                print(f"[Warn] Failed split={split_name} rel={rel} err={e}", flush=True)

        pbar.set_postfix({"processed": processed, "skipped": skipped, "failed": failed})

    print(f"[Pass1 Done] Split={split_name} processed={processed} skipped={skipped} failed={failed}", flush=True)


# -----------------------------------------------------------------------------
# Compute global normalization stats from TRAIN raw
# -----------------------------------------------------------------------------
def compute_and_save_stats(depth_root: Path, sampler: ReservoirSampler, args) -> dict:
    vals = sampler.values()
    if vals.size == 0:
        raise RuntimeError("No samples collected for stats. Check valid_min/stat_pixels_per_image.")

    p_low = float(np.percentile(vals, args.p_low))
    p_high = float(np.percentile(vals, args.p_high))
    if not np.isfinite(p_low) or not np.isfinite(p_high) or (p_high <= p_low):
        raise RuntimeError(f"Invalid percentiles: p_low={p_low}, p_high={p_high}")

    stats = {
        "norm_mode": "global_percentile",
        "p_low": float(args.p_low),
        "p_high": float(args.p_high),
        "clip_min": p_low,
        "clip_max": p_high,
        "valid_min": float(args.valid_min),
        "note": "Use raw(.raw.npy)+mask(.mask.npy) for evaluation; use norm(.npy) for training.",
    }

    out_json = depth_root / "_depth_stats.json"
    out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[Stats] Saved -> {out_json.as_posix()}", flush=True)
    return stats


# -----------------------------------------------------------------------------
# Pass 2: generate TRAINING normalized depth (png.npy) + visualization PNG
# -----------------------------------------------------------------------------
def normalize_split_from_raw(
    split_name: str,
    out_split_root: Path,
    stats: dict,
    args,
):
    clip_min = float(stats["clip_min"])
    clip_max = float(stats["clip_max"])
    denom = (clip_max - clip_min) + 1e-12

    # find all raw files under split
    raw_files = sorted(out_split_root.glob("**/*.png.raw.npy")) if args.output_ext.lower() == "png" else \
                sorted(out_split_root.glob(f"**/*.{args.output_ext}.raw.npy"))

    print(f"[Pass2] Split={split_name} raw_files={len(raw_files)} under {out_split_root}", flush=True)

    processed, skipped, failed = 0, 0, 0
    pbar = tqdm(raw_files, desc=f"NORM[{split_name}]", dynamic_ncols=True)
    for raw_p in pbar:
        # derive out_png (remove .raw.npy)
        # raw_p is like xxx.png.raw.npy  -> out_png is xxx.png
        raw_s = raw_p.as_posix()
        if not raw_s.endswith(".raw.npy"):
            continue
        out_png = Path(raw_s[:-len(".raw.npy")])
        raw_p2, mask_p, norm_p = out_paths_for(out_png)

        if args.skip_existing and norm_p.exists() and (not args.save_png_vis or out_png.exists()):
            skipped += 1
            pbar.set_postfix({"processed": processed, "skipped": skipped, "failed": failed})
            continue

        try:
            raw = np.load(raw_p.as_posix()).astype(np.float32)

            # normalize with global percentile clipping
            dn = (raw - clip_min) / denom
            dn = np.clip(dn, 0.0, 1.0).astype(np.float32)

            # optional: avoid exact 0/1 for numerical stability in downstream training
            if args.floor_eps is not None and args.floor_eps > 0:
                fe = float(args.floor_eps)
                dn = dn * (1.0 - 2.0 * fe) + fe
                dn = np.clip(dn, fe, 1.0 - fe).astype(np.float32)

            np.save(norm_p.as_posix(), dn)

            if args.save_png_vis:
                # save uint16 PNG for visualization (from normalized)
                save_depth_png_uint16(dn, out_png)

            processed += 1
        except Exception as e:
            failed += 1
            print(f"[Warn] Normalize failed: {raw_p} err={e}", flush=True)

        pbar.set_postfix({"processed": processed, "skipped": skipped, "failed": failed})

    print(f"[Pass2 Done] Split={split_name} processed={processed} skipped={skipped} failed={failed}", flush=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args):
    in_root = Path(args.input_root)
    out_root = Path(args.output_root)
    ensure_dir(out_root)

    # device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(device)
        print(f"[Info] Using GPU {args.gpu_id}: {torch.cuda.get_device_name(device)}", flush=True)
    else:
        device = torch.device("cpu")
        print("[Info] CUDA not available. Using CPU.", flush=True)

    # model
    print(f"[Info] Loading depth model: {args.model_id}", flush=True)
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_id).to(device)
    model.eval()

    # splits
    splits = args.splits
    if splits == "auto":
        cand = []
        if (in_root / "train").exists():
            cand.append("train")
        if (in_root / "test").exists():
            cand.append("test")
        if len(cand) == 0:
            raise FileNotFoundError("auto split mode: expected input_root/train or input_root/test to exist.")
        split_list = cand
    else:
        split_list = [s.strip() for s in splits.split(",") if len(s.strip()) > 0]

    print(f"[Info] input_root={in_root}", flush=True)
    print(f"[Info] output_root={out_root}", flush=True)
    print(f"[Info] splits={split_list}", flush=True)
    print(f"[Info] norm_mode=global_percentile p_low={args.p_low} p_high={args.p_high}", flush=True)

    # Pass 1: raw + mask (collect stats on train)
    sampler = ReservoirSampler(max_samples=args.stat_max_samples, seed=args.seed)

    for split in split_list:
        in_split = in_root / split
        if not in_split.exists():
            print(f"[Warn] split '{split}' not found under input_root, skip.", flush=True)
            continue
        out_split = out_root / split
        ensure_dir(out_split)

        if split == "train":
            generate_raw_and_masks(split, in_split, out_split, model, processor, device, args, sampler=sampler)
        else:
            generate_raw_and_masks(split, in_split, out_split, model, processor, device, args, sampler=None)

    # stats: either compute fresh or load existing
    stats_path = out_root / "_depth_stats.json"
    if stats_path.exists() and args.reuse_existing_stats:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        print(f"[Stats] Reusing existing stats: {stats_path.as_posix()}", flush=True)
    else:
        stats = compute_and_save_stats(out_root, sampler, args)

    # Pass 2: normalize each split from raw to training-ready norm + optional PNG
    for split in split_list:
        out_split = out_root / split
        if out_split.exists():
            normalize_split_from_raw(split, out_split, stats, args)

    print("[OK] Dataset generation finished.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser("Generate paper-protocol depth dataset (raw+mask for eval, global-norm for training)")

    p.add_argument("--input_root", type=str, required=True,
                   help="RGB root containing train/ and test/ (each with class subfolders).")
    p.add_argument("--output_root", type=str, required=True,
                   help="Depth root to create: depth/train/... and depth/test/...")

    p.add_argument("--splits", type=str, default="auto",
                   help="auto or comma-separated list (e.g., 'train,test' or 'test').")

    p.add_argument("--model_id", type=str, default="depth-anything/Depth-Anything-V2-LARGE-hf")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gpu_id", type=int, default=0)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    # output naming
    p.add_argument("--output_ext", type=str, default="png",
                   help="Output image extension for visualization (recommend png).")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip if raw+mask already exist (pass1) / norm+png exist (pass2).")
    p.add_argument("--save_png_vis", action="store_true",
                   help="Also save uint16 PNG for visualization (not used for metrics).")

    # validity / stability
    p.add_argument("--valid_min", type=float, default=1e-6,
                   help="Valid depth threshold for mask (raw > valid_min).")
    p.add_argument("--floor_eps", type=float, default=0.0,
                   help="Optional: push normalized depth away from exact 0/1 (e.g., 1/65535).")

    # global percentile normalization (train stats)
    p.add_argument("--p_low", type=float, default=1.0, help="Low percentile for global clipping.")
    p.add_argument("--p_high", type=float, default=99.0, help="High percentile for global clipping.")
    p.add_argument("--stat_pixels_per_image", type=int, default=512,
                   help="How many valid pixels to sample per image for percentile stats.")
    p.add_argument("--stat_max_samples", type=int, default=2_000_000,
                   help="Max reservoir samples for percentile stats.")
    p.add_argument("--reuse_existing_stats", action="store_true",
                   help="If _depth_stats.json exists, reuse it instead of recomputing.")

    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    main(args)
