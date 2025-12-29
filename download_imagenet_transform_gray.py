#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export Hugging Face ImageNet-1k (gated) to ImageFolder structure,
optionally converting to linear grayscale (gray3) via a strictly linear transform.

- Dataset: ILSVRC/imagenet-1k (gated; requires access approval + auth token)
- Output:
    out_root/
      gray3/train/<class>/*.png
      gray3/val/<class>/*.png
    (optional) out_root/rgb/...
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets import load_dataset


# ------------------------ Utils ------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def get_auth_kwargs():
    """
    datasets.load_dataset auth args differ by version.
    Prefer token=True (newer), fallback to use_auth_token=True (older).
    """
    import inspect
    sig = inspect.signature(load_dataset)
    if "token" in sig.parameters:
        return {"token": True}
    if "use_auth_token" in sig.parameters:
        return {"use_auth_token": True}
    # If neither exists, datasets will rely on huggingface_hub cached token (hf auth login / HF_TOKEN)
    return {}

def pil_to_rgb01(img: Image.Image) -> np.ndarray:
    # No gamma / colorspace conversion: we just read bytes and scale to [0,1].
    img = img.convert("RGB")
    arr_u8 = np.asarray(img, dtype=np.uint8)  # (H,W,3)
    return arr_u8.astype(np.float32) / 255.0

def rgb01_to_gray3_u8(rgb01: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    rgb01: float32 (H,W,3) in [0,1]
    w: float32 (3,), nonneg, sum=1 recommended
    Returns:
      gray01: float32 (H,W) in [0,1]
      gray3_u8: uint8 (H,W,3) in [0,255]
    """
    gray01 = (rgb01.reshape(-1, 3) @ w.reshape(3, 1)).reshape(rgb01.shape[0], rgb01.shape[1])
    gray3_01 = np.repeat(gray01[..., None], 3, axis=-1)
    gray3_u8 = np.clip(gray3_01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return gray01, gray3_u8

def save_png_u8(arr_u8: np.ndarray, path: Path, compress_level: int = 4):
    # arr_u8: (H,W,3) uint8
    img = Image.fromarray(arr_u8, mode="RGB")
    img.save(path, format="PNG", optimize=True, compress_level=int(compress_level))

def label_to_classname(ds, y: int) -> str:
    # ImageNet-1k dataset provides label:int, and ClassLabel mapping exists in features.
    try:
        feat = ds.features.get("label", None)
        if feat is not None and hasattr(feat, "int2str"):
            return feat.int2str(int(y))
        if feat is not None and hasattr(feat, "names"):
            return str(feat.names[int(y)])
    except Exception:
        pass
    return f"{int(y):04d}"


# ------------------------ Main Export ------------------------

def export_split(
    ds_name: str,
    split_name: str,
    out_root: Path,
    do_gray3: bool,
    do_rgb: bool,
    weights: np.ndarray,
    streaming: bool,
    shard_id: int,
    num_shards: int,
    max_images: int,
    compress_level: int,
):
    auth_kwargs = get_auth_kwargs()

    ds = load_dataset(ds_name, split=split_name, streaming=streaming, **auth_kwargs)

    # Shard for easy parallelization (run multiple processes with different shard_id)
    if num_shards > 1:
        ds = ds.shard(num_shards=num_shards, index=shard_id)

    # Split folder naming
    split_dir = "train" if split_name == "train" else "val"

    base_gray3 = out_root / "gray3" / split_dir
    base_rgb   = out_root / "rgb"   / split_dir
    if do_gray3:
        ensure_dir(base_gray3)
    if do_rgb:
        ensure_dir(base_rgb)

    # IterableDataset has no reliable __len__ in streaming mode
    it = iter(ds)
    pbar = tqdm(total=(max_images if max_images > 0 else None), desc=f"[Export] {split_name} shard {shard_id}/{num_shards}")

    n = 0
    errors = 0

    for ex in it:
        if max_images > 0 and n >= max_images:
            break

        try:
            img = ex["image"]  # dataset card: PIL.Image.Image :contentReference[oaicite:4]{index=4}
            y = int(ex["label"])

            cls = label_to_classname(ds, y)

            # Load as linear [0,1] float
            rgb01 = pil_to_rgb01(img)  # (H,W,3), float32

            # Filename: per-shard + running index (avoid collisions)
            fname = f"{split_name}_sh{shard_id:02d}_{n:08d}.png"

            if do_rgb:
                out_dir = base_rgb / cls
                ensure_dir(out_dir)
                rgb_u8 = np.clip(rgb01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
                save_png_u8(rgb_u8, out_dir / fname, compress_level=compress_level)

            if do_gray3:
                out_dir = base_gray3 / cls
                ensure_dir(out_dir)
                _, gray3_u8 = rgb01_to_gray3_u8(rgb01, weights)
                save_png_u8(gray3_u8, out_dir / fname, compress_level=compress_level)

            n += 1
            pbar.update(1)

        except Exception:
            errors += 1
            # continue; keep going on corrupt/edge cases

    pbar.close()
    return {"split": split_name, "shard_id": shard_id, "num_exported": n, "errors": errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="ILSVRC/imagenet-1k",
                    help="Hugging Face dataset name (gated).")
    ap.add_argument("--out_root", type=str, default="./imagenet1k_export",
                    help="Output root directory.")
    ap.add_argument("--no_gray3", action="store_true", help="Do not export gray3.")
    ap.add_argument("--export_rgb", action="store_true", help="Also export RGB images.")
    ap.add_argument("--weights", type=float, nargs=3, default=[0.2989, 0.5870, 0.1140],
                    help="Linear grayscale weights wR wG wB (nonneg, sum=1 recommended).")
    ap.add_argument("--streaming", action="store_true",
                    help="Use streaming=True (recommended to avoid huge RAM usage).")
    ap.add_argument("--shard_id", type=int, default=0,
                    help="Shard index for parallel export (0 <= shard_id < num_shards).")
    ap.add_argument("--num_shards", type=int, default=32,
                    help="Number of shards for parallel export.")
    ap.add_argument("--max_images", type=int, default=0,
                    help="Export only first N images per split (0 = all). Useful for testing.")
    ap.add_argument("--compress_level", type=int, default=4, help="PNG compress_level (0-9).")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ensure_dir(out_root)

    w = np.array(args.weights, dtype=np.float32)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    s = float(w.sum())
    if s <= 0:
        raise ValueError("weights sum must be > 0.")
    w = w / s  # keep linearity; just normalize to sum=1

    do_gray3 = not args.no_gray3
    do_rgb = bool(args.export_rgb)

    # Export train + validation (ImageNet-1k uses 'validation' split name) :contentReference[oaicite:5]{index=5}
    results = []
    for split in ["train", "validation"]:
        res = export_split(
            ds_name=args.dataset,
            split_name=split,
            out_root=out_root,
            do_gray3=do_gray3,
            do_rgb=do_rgb,
            weights=w,
            streaming=bool(args.streaming),
            shard_id=int(args.shard_id),
            num_shards=int(args.num_shards),
            max_images=int(args.max_images),
            compress_level=int(args.compress_level),
        )
        results.append(res)

    # Save a small manifest for bookkeeping
    manifest = {
        "dataset": args.dataset,
        "out_root": str(out_root.resolve()),
        "gray3": do_gray3,
        "rgb": do_rgb,
        "weights_normalized": w.tolist(),
        "streaming": bool(args.streaming),
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "results": results,
    }
    with (out_root / "_export_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\nDone.")

if __name__ == "__main__":
    main()
