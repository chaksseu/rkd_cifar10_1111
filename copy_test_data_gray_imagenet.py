#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline image preprocessing for FID:
- Recursively scan --in_dir for images
- For each image:
  1) Load & fully decode (fail-fast on corruption)
  2) Convert to RGB
  3) Resize so that the shorter edge becomes --size (default 256) using bicubic
  4) Center crop to --size x --size
  5) Save to --out_dir preserving directory structure (default PNG)

Fail-fast:
- If any image fails to load/decode/transform/save, terminate immediately.

Example:
  python preprocess_offline_256cc_mp.py \
    --in_dir  /workspace/.../imagenet1k_export/gray3/val \
    --out_dir /workspace/.../imagenet1k_export/gray3/val_256cc \
    --size 256 --workers 16 --overwrite
"""

import os
import sys
import argparse
from pathlib import Path
import multiprocessing as mp
from PIL import Image

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

IMG_EXTS_DEFAULT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

def collect_image_files(in_dir: Path, exts):
    exts = {e.lower() for e in exts}
    files = []
    for p in in_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files

def resize_shorter_edge(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {img.size}")

    short = min(w, h)
    if short == size:
        return img

    scale = size / float(short)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Safety: ensure at least size in both dims after rounding
    new_w = max(new_w, size)
    new_h = max(new_h, size)

    return img.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)

def center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w < size or h < size:
        # Should not happen if resize_shorter_edge succeeded, but keep it strict.
        raise ValueError(f"Image too small for center crop: got {img.size}, need >=({size},{size})")

    left = (w - size) // 2
    top = (h - size) // 2
    right = left + size
    bottom = top + size
    return img.crop((left, top, right, bottom))

def ensure_parent_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def atomic_save_png(img: Image.Image, dst: Path):
    # Atomic write: save to temp then replace
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    img.save(tmp, format="PNG")
    os.replace(tmp, dst)

def worker_init():
    # Make PIL safer with large images, but still decode fully (fail-fast).
    Image.MAX_IMAGE_PIXELS = None  # avoid DecompressionBombError for very large images

def process_one(args_tuple):
    src_str, in_root_str, out_root_str, size, overwrite = args_tuple
    src = Path(src_str)
    in_root = Path(in_root_str)
    out_root = Path(out_root_str)

    rel = src.relative_to(in_root)
    dst = (out_root / rel).with_suffix(".png")  # force PNG

    if dst.exists() and not overwrite:
        return ("skip", src_str, str(dst))

    try:
        # Load & fully decode (fail-fast)
        with Image.open(src) as im:
            im.load()  # force decode now; catches truncation/corruption early
            im = im.convert("RGB")
            im = resize_shorter_edge(im, size)
            im = center_crop(im, size)

            ensure_parent_dir(dst)
            atomic_save_png(im, dst)

        return ("ok", src_str, str(dst))

    except Exception as e:
        # Raise to trigger immediate termination in main
        raise RuntimeError(f"Failed on src={src_str} -> dst={str(dst)} | err={repr(e)}") from e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--exts", type=str, default=",".join(sorted(IMG_EXTS_DEFAULT)),
                    help="Comma-separated extensions, e.g. .jpg,.jpeg,.png")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if not in_dir.exists():
        print(f"[ERR] in_dir not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    if in_dir.resolve() == out_dir.resolve():
        print("[ERR] in_dir and out_dir must be different (to avoid overwriting originals).", file=sys.stderr)
        sys.exit(1)

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    files = collect_image_files(in_dir, exts)
    if len(files) == 0:
        print(f"[ERR] No images found under {in_dir} with exts={exts}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] in_dir  = {in_dir}")
    print(f"[Info] out_dir = {out_dir}")
    print(f"[Info] size    = {args.size} (resize shorter edge -> {args.size}, center crop -> {args.size}x{args.size})")
    print(f"[Info] workers = {args.workers}")
    print(f"[Info] images  = {len(files)}")
    print(f"[Info] overwrite = {args.overwrite}")

    # Prepare tasks
    task_args = [(str(p), str(in_dir), str(out_dir), args.size, args.overwrite) for p in files]

    # Multiprocessing (fail-fast): terminate pool on first exception
    ctx = mp.get_context("spawn")  # more robust than fork with PIL in some environments
    pool = ctx.Pool(processes=args.workers, initializer=worker_init)

    ok = skip = 0
    try:
        it = pool.imap_unordered(process_one, task_args, chunksize=32)

        if tqdm is not None:
            it = tqdm(it, total=len(task_args), desc="Preprocess")

        for status, src, dst in it:
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1

        pool.close()
        pool.join()

    except Exception as e:
        # Hard stop on any error
        pool.terminate()
        pool.join()
        print("\n[ERR] Preprocessing aborted due to an error (fail-fast).", file=sys.stderr)
        print(str(e), file=sys.stderr)
        sys.exit(2)

    print(f"[Done] ok={ok}, skip={skip}, err=0")
    print(f"[Done] output saved to: {out_dir}")

if __name__ == "__main__":
    main()
