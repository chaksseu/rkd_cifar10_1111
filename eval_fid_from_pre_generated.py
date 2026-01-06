#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute custom-Inception FID from PRE-GENERATED images on disk (NO diffusion sampling).

Expected generated layout (recursively under --gen_root):
  <...>/<tag>/fid_gen/gen_000000.png ...
Optionally also:
  <...>/<tag>/gen_meta.json

This script:
  - Discovers all fid_gen dirs under --gen_root
  - Prepares real-cache (flatten) once
  - Loads your trained Inception-v3 checkpoint
  - Computes real stats once (all + optional per-class)
  - For each fid_gen dir, computes gen stats (optionally cached) and FID
  - Logs to:
      - <report_dir>/results.jsonl
      - <report_dir>/results.csv
      - optional wandb

Deps:
  pip install torch torchvision numpy pillow
  (optional) pip install wandb safetensors

Example:
  python compute_fid_from_pre_generated.py \
    --gen_root /workspace/rkd_cifar10_1111/eval_fid_out_0105 \
    --test_dir /workspace/rkd_cifar10_1111/cifar10_png_linear_only/gray3/test \
    --inception_ckpt /workspace/rkd_cifar10_1111/0102-imagenet-gray3/output_adamw_bs256/ckpts/best.pt \
    --device cuda:7 \
    --fid_batch_size 256 --fid_num_workers 8 \
    --fid_num_samples 0 \
    --fid_per_class \
    --report_dir /workspace/rkd_cifar10_1111/eval_fid_out_0105
"""

import os
import re
import json
import csv
import math
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image


# ------------------------- utils -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def collect_image_paths_recursive(root: Path, exts={".png", ".jpg", ".jpeg"}) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def list_class_dirs(test_dir: Path) -> List[Path]:
    return [d for d in test_dir.iterdir() if d.is_dir()]

def sanitize_tag(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._\-]+", "_", s)
    return s[:160] if len(s) > 160 else s

def flatten_real_cache(src_dir: Path, cache_dir: Path, use_symlink: bool = True) -> int:
    """
    Flatten any nested images under src_dir into cache_dir for stable scanning.
    Uses symlink by default; falls back to copy if symlink fails.
    """
    ensure_dir(cache_dir)
    existing = list(cache_dir.glob("*"))
    if len(existing) > 0:
        return len(existing)

    paths = collect_image_paths_recursive(src_dir)
    print(f"[RealCache] Flatten real set ({len(paths)} imgs) -> {cache_dir}", flush=True)

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

def discover_fid_gen_dirs(gen_root: Path) -> List[Path]:
    """
    Recursively find directories named 'fid_gen' that contain gen_*.png images.
    Returns list of fid_gen directories.
    """
    out: List[Path] = []
    for d in gen_root.rglob("fid_gen"):
        if not d.is_dir():
            continue
        if any(p.is_file() for p in d.glob("gen_*.png")):
            out.append(d)
    # de-dup
    uniq = []
    seen = set()
    for p in out:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return sorted(uniq, key=lambda x: x.as_posix())

def count_gen_images(fid_gen_dir: Path) -> int:
    return len([p for p in fid_gen_dir.glob("gen_*.png") if p.is_file()])

def _sort_key_gen_name(p: Path) -> Tuple[int, str]:
    """
    Sort gen_000123.png numerically; fallback to name.
    """
    m = re.search(r"gen_(\d+)\.", p.name)
    if m:
        return (int(m.group(1)), p.name)
    return (10**18, p.name)

def load_json_if_exists(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ------------------------- custom inception loading -------------------------

def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}

def _load_ckpt_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(path.as_posix())
        return dict(sd)

    obj = torch.load(path.as_posix(), map_location="cpu")
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "ema", "student", "teacher"]:
            if key in obj and isinstance(obj[key], dict):
                return _strip_module_prefix(obj[key])
        if all(isinstance(k, str) for k in obj.keys()):
            return _strip_module_prefix(obj)
    raise RuntimeError(f"Unrecognized checkpoint format: {path}")

def build_inception_v3(num_classes: int, aux_logits: bool) -> nn.Module:
    try:
        m = models.inception_v3(weights=None, aux_logits=aux_logits, transform_input=False)
    except TypeError:
        m = models.inception_v3(pretrained=False, aux_logits=aux_logits, transform_input=False)

    if hasattr(m, "fc") and isinstance(m.fc, nn.Linear):
        if m.fc.out_features != num_classes:
            m.fc = nn.Linear(m.fc.in_features, num_classes, bias=True)
    return m

@torch.no_grad()
def inception_v3_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # forward to pre-logits 2048
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

    if hasattr(model, "avgpool"):
        x = model.avgpool(x)
    else:
        x = F.adaptive_avg_pool2d(x, (1, 1))

    x = torch.flatten(x, 1)
    return x

def load_custom_inception(ckpt_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    sd = _load_ckpt_state_dict(ckpt_path)
    aux_logits = any(("AuxLogits" in k) for k in sd.keys())
    model = build_inception_v3(num_classes=num_classes, aux_logits=aux_logits)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 0:
        print(f"[Warn] Inception missing keys: {len(missing)} (up to 5): {missing[:5]}", flush=True)
    if len(unexpected) > 0:
        print(f"[Warn] Inception unexpected keys: {len(unexpected)} (up to 5): {unexpected[:5]}", flush=True)

    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


# ------------------------- stats + fid -------------------------

class ImageDirDataset(Dataset):
    def __init__(self, root_dir: Path, input_size: int, mean: List[float], std: List[float],
                 max_items: int = 0, sort_gen_numeric: bool = False):
        paths = [p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if sort_gen_numeric:
            paths = sorted(paths, key=_sort_key_gen_name)
        else:
            paths = sorted(paths, key=lambda x: x.as_posix())

        if max_items and max_items > 0:
            paths = paths[:max_items]

        self.paths = paths
        self.tf = T.Compose([
            T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
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
    max_items: int = 0,
    sort_gen_numeric: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Streaming accumulation:
      sum_x = Σ f
      sum_xx = Σ f^T f
      mean = sum_x / n
      cov(unbiased) = (sum_xx - n*outer(mean,mean)) / (n-1)
    """
    ds = ImageDirDataset(
        img_dir,
        input_size=input_size,
        mean=mean,
        std=std,
        max_items=max_items,
        sort_gen_numeric=sort_gen_numeric,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    d = 2048
    n = 0
    sum_x = np.zeros((d,), dtype=np.float64)
    sum_xx = np.zeros((d, d), dtype=np.float64)

    for x in loader:
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        f = inception_v3_features(inception, x)          # (B,2048) float32 on GPU
        f_np = f.detach().cpu().numpy().astype(np.float64)

        n_b = f_np.shape[0]
        n += n_b
        sum_x += f_np.sum(axis=0)
        sum_xx += f_np.T @ f_np

    if n <= 1:
        raise RuntimeError(f"Not enough images to compute covariance (n={n}) for: {img_dir}")

    mu = sum_x / float(n)
    cov = (sum_xx - float(n) * np.outer(mu, mu)) / float(n - 1)
    cov = 0.5 * (cov + cov.T)  # symmetrize

    return mu, cov, n

def sqrtm_psd(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, eps, None)
    return (vecs * np.sqrt(vals)) @ vecs.T

def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    diff = mu1 - mu2
    s1_sqrt = sqrtm_psd(sigma1)
    prod = s1_sqrt @ sigma2 @ s1_sqrt
    covmean = sqrtm_psd(prod)

    fid = float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))
    return max(0.0, fid)

def save_stats_npz(path: Path, mu: np.ndarray, sigma: np.ndarray, n: int, extra: Optional[dict] = None):
    ensure_dir(path.parent)
    payload = {
        "mu": mu.astype(np.float64),
        "sigma": sigma.astype(np.float64),
        "n": np.int64(n),
    }
    if extra:
        for k, v in extra.items():
            # store as 0-d object array for strings/dicts
            payload[f"extra__{k}"] = np.array(v, dtype=object)
    np.savez_compressed(path.as_posix(), **payload)

def load_stats_npz(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, int, dict]]:
    if not path.exists():
        return None
    obj = np.load(path.as_posix(), allow_pickle=True)
    mu = obj["mu"]
    sigma = obj["sigma"]
    n = int(obj["n"])
    extra = {}
    for k in obj.files:
        if k.startswith("extra__"):
            extra[k[len("extra__"):]] = obj[k].item()
    return mu, sigma, n, extra


# ------------------------- main -------------------------

def main():
    p = argparse.ArgumentParser("Compute custom-Inception FID from pre-generated images under gen_root.")

    p.add_argument("--gen_root", type=str, required=True, help="Root to search for <tag>/fid_gen/gen_*.png")
    p.add_argument("--report_dir", type=str, default="", help="Where to write results + caches. Default: gen_root")

    p.add_argument("--test_dir", type=str, required=True, help="Real images directory (e.g., cifar10 .../test)")
    p.add_argument("--fid_cache_dir", type=str, default="", help="Real flatten cache dir. Default: <report_dir>/fid_real_cache")
    p.add_argument("--fid_symlink_real", action="store_true", help="Use symlink for real cache (default True).")
    p.add_argument("--fid_copy_real", action="store_true", help="Copy real images into cache.")
    p.add_argument("--fid_per_class", action="store_true", help="Also compute per-class FID if test_dir has class subfolders.")

    p.add_argument("--inception_ckpt", type=str, required=True, help="Path to YOUR trained InceptionV3 checkpoint.")
    p.add_argument("--inception_num_classes", type=int, default=1000)
    p.add_argument("--inception_input_size", type=int, default=299)
    p.add_argument("--inception_mean", type=float, nargs=3, default=[0.45798322587856827] * 3)
    p.add_argument("--inception_std", type=float, nargs=3, default=[0.2623006911570552] * 3)

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--fid_batch_size", type=int, default=256)
    p.add_argument("--fid_num_workers", type=int, default=4)

    p.add_argument("--fid_num_samples", type=int, default=0,
                   help="0 => use min(n_real, n_gen). Else min(fid_num_samples, n_real, n_gen).")

    p.add_argument("--cache_stats", action="store_true",
                   help="Cache real/gen activation stats to .npz to avoid recompute on re-runs.")
    p.add_argument("--recompute_stats", action="store_true",
                   help="Ignore cached stats and recompute.")

    # W&B
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_offline", action="store_true")
    p.add_argument("--project", type=str, default="cifar10-gray3-customfid")
    p.add_argument("--run_name", type=str, default="fid-from-pre-gen")

    args = p.parse_args()

    gen_root = Path(args.gen_root)
    if not gen_root.exists():
        raise FileNotFoundError(f"--gen_root not found: {gen_root}")

    report_dir = Path(args.report_dir) if args.report_dir else gen_root
    ensure_dir(report_dir)

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"--test_dir not found: {test_dir}")

    inception_ckpt = Path(args.inception_ckpt)
    if not inception_ckpt.exists():
        raise FileNotFoundError(f"--inception_ckpt not found: {inception_ckpt}")

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

    # discover fid_gen dirs
    fid_gen_dirs = discover_fid_gen_dirs(gen_root)
    if len(fid_gen_dirs) == 0:
        raise SystemExit(f"No fid_gen dirs found under: {gen_root}")

    print(f"[Info] Device={device}", flush=True)
    print(f"[Info] Found fid_gen dirs: {len(fid_gen_dirs)}", flush=True)
    for d in fid_gen_dirs[:10]:
        print(f"  - {d}", flush=True)
    if len(fid_gen_dirs) > 10:
        print("  ...", flush=True)

    # real cache
    cache_root = Path(args.fid_cache_dir) if args.fid_cache_dir else (report_dir / "fid_real_cache")
    ensure_dir(cache_root)

    use_symlink = True
    if args.fid_copy_real:
        use_symlink = False
    if args.fid_symlink_real:
        use_symlink = True

    real_all_cache = cache_root / "all"
    n_real_all = flatten_real_cache(test_dir, real_all_cache, use_symlink=use_symlink)
    print(f"[RealCache] all={n_real_all} @ {real_all_cache}", flush=True)

    real_class_caches: Dict[str, Path] = {}
    if args.fid_per_class:
        class_dirs = list_class_dirs(test_dir)
        if len(class_dirs) > 0:
            per_class_root = cache_root / "per_class"
            ensure_dir(per_class_root)
            for cd in sorted(class_dirs, key=lambda x: x.name):
                cname = cd.name
                ccache = per_class_root / cname
                flatten_real_cache(cd, ccache, use_symlink=use_symlink)
                real_class_caches[cname] = ccache
            print(f"[RealCache] per-class caches: {len(real_class_caches)}", flush=True)
        else:
            print("[Warn] --fid_per_class set but no class subdirs under test_dir; skipping per-class.", flush=True)

    # load inception once
    print(f"[FID] Loading custom InceptionV3 from: {inception_ckpt}", flush=True)
    inception = load_custom_inception(inception_ckpt, num_classes=args.inception_num_classes, device=device)

    # stats cache dir
    stats_cache_dir = report_dir / "fid_stats_cache"
    ensure_dir(stats_cache_dir)

    # real stats (all)
    real_all_stats_path = stats_cache_dir / "real_all_stats.npz"
    real_all_stats = None
    if args.cache_stats and (not args.recompute_stats):
        loaded = load_stats_npz(real_all_stats_path)
        if loaded is not None:
            mu_r, sig_r, n_r, _ = loaded
            real_all_stats = (mu_r, sig_r, n_r)
            print(f"[FID] Loaded REAL(all) stats cache: n={n_r} @ {real_all_stats_path}", flush=True)

    if real_all_stats is None:
        print("[FID] Computing REAL(all) stats ...", flush=True)
        mu_r, sig_r, n_r = compute_activation_stats_from_dir(
            img_dir=real_all_cache,
            inception=inception,
            device=device,
            batch_size=args.fid_batch_size,
            num_workers=args.fid_num_workers,
            input_size=args.inception_input_size,
            mean=args.inception_mean,
            std=args.inception_std,
            max_items=0,
            sort_gen_numeric=False,
        )
        real_all_stats = (mu_r, sig_r, n_r)
        print(f"[FID] REAL(all) stats ready: n={n_r}", flush=True)
        if args.cache_stats:
            save_stats_npz(real_all_stats_path, mu_r, sig_r, n_r, extra={"src": str(real_all_cache)})
            print(f"[FID] Saved REAL(all) stats cache -> {real_all_stats_path}", flush=True)

    # real stats (per-class)
    real_class_stats: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}
    if args.fid_per_class and len(real_class_caches) > 0:
        for cname, cdir in real_class_caches.items():
            cpath = stats_cache_dir / f"real_class__{sanitize_tag(cname)}.npz"
            loaded = None
            if args.cache_stats and (not args.recompute_stats):
                loaded = load_stats_npz(cpath)

            if loaded is not None:
                mu_c, sig_c, n_c, _ = loaded
                real_class_stats[cname] = (mu_c, sig_c, n_c)
                print(f"[FID] Loaded REAL(class={cname}) stats cache: n={n_c}", flush=True)
            else:
                print(f"[FID] Computing REAL(class={cname}) stats ...", flush=True)
                mu_c, sig_c, n_c = compute_activation_stats_from_dir(
                    img_dir=cdir,
                    inception=inception,
                    device=device,
                    batch_size=args.fid_batch_size,
                    num_workers=args.fid_num_workers,
                    input_size=args.inception_input_size,
                    mean=args.inception_mean,
                    std=args.inception_std,
                    max_items=0,
                    sort_gen_numeric=False,
                )
                real_class_stats[cname] = (mu_c, sig_c, n_c)
                print(f"[FID] REAL(class={cname}) stats ready: n={n_c}", flush=True)
                if args.cache_stats:
                    save_stats_npz(cpath, mu_c, sig_c, n_c, extra={"src": str(cdir)})
                    print(f"[FID] Saved REAL(class={cname}) stats cache -> {cpath}", flush=True)

    # result files
    results_jsonl = report_dir / "results.jsonl"
    results_csv = report_dir / "results.csv"

    csv_header = [
        "rel_tag_dir", "tag", "experiment",
        "gen_dir", "n_real_all", "n_gen_total", "n_gen_used",
        "fid_all", "inception_ckpt", "inception_input_size",
    ]

    # write csv header if not exists
    if not results_csv.exists():
        with results_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(csv_header)

    mu_r, sig_r, n_r = real_all_stats

    # evaluate each gen dir
    for fid_gen_dir in fid_gen_dirs:
        tag_dir = fid_gen_dir.parent
        rel_tag_dir = str(tag_dir.relative_to(gen_root))
        tag = tag_dir.name

        # experiment = first component under gen_root (if exists)
        parts = Path(rel_tag_dir).parts
        experiment = parts[0] if len(parts) >= 2 else ""

        n_gen_total = count_gen_images(fid_gen_dir)
        if n_gen_total <= 1:
            print(f"[Skip] {rel_tag_dir}: not enough gen images (n={n_gen_total})", flush=True)
            continue

        # choose how many samples to use
        if args.fid_num_samples <= 0:
            n_use = int(min(n_r, n_gen_total))
        else:
            n_use = int(min(args.fid_num_samples, n_r, n_gen_total))

        if n_use < 2:
            print(f"[Skip] {rel_tag_dir}: n_use < 2 (n_use={n_use})", flush=True)
            continue

        if n_gen_total != n_r and args.fid_num_samples <= 0:
            print(f"[Warn] {rel_tag_dir}: n_gen_total({n_gen_total}) != n_real_all({n_r}). Using n_use={n_use}.", flush=True)

        # gen stats cache (stored next to tag_dir for locality)
        gen_stats_path = tag_dir / "fid_stats_gen.npz"
        gen_stats = None
        if args.cache_stats and (not args.recompute_stats):
            loaded = load_stats_npz(gen_stats_path)
            if loaded is not None:
                mu_g, sig_g, n_g, extra = loaded
                # basic cache validity: n match
                if int(n_g) == int(n_use):
                    gen_stats = (mu_g, sig_g, n_g)
                    print(f"[FID] Loaded GEN stats cache: {rel_tag_dir} n={n_g} @ {gen_stats_path}", flush=True)

        if gen_stats is None:
            print(f"[FID] Computing GEN stats: {rel_tag_dir} (use {n_use}/{n_gen_total}) ...", flush=True)
            mu_g, sig_g, n_g = compute_activation_stats_from_dir(
                img_dir=fid_gen_dir,
                inception=inception,
                device=device,
                batch_size=args.fid_batch_size,
                num_workers=args.fid_num_workers,
                input_size=args.inception_input_size,
                mean=args.inception_mean,
                std=args.inception_std,
                max_items=n_use,
                sort_gen_numeric=True,   # important for gen_000xxx ordering
            )
            gen_stats = (mu_g, sig_g, n_g)
            if args.cache_stats:
                save_stats_npz(
                    gen_stats_path, mu_g, sig_g, n_g,
                    extra={"src": str(fid_gen_dir), "n_use": int(n_use), "n_gen_total": int(n_gen_total)}
                )
                print(f"[FID] Saved GEN stats cache -> {gen_stats_path}", flush=True)

        mu_g, sig_g, n_g = gen_stats
        fid_all = frechet_distance(mu_r, sig_r, mu_g, sig_g)
        print(f"[FID] {rel_tag_dir} | FID(all) = {fid_all:.4f} (real={n_r}, gen_used={n_g})", flush=True)

        fid_per_class = {}
        if args.fid_per_class and len(real_class_stats) > 0:
            for cname, (mu_c, sig_c, n_c) in real_class_stats.items():
                fid_c = frechet_distance(mu_c, sig_c, mu_g, sig_g)
                fid_per_class[cname] = float(fid_c)
            txt = ", ".join([f"{k}={v:.4f}" for k, v in sorted(fid_per_class.items(), key=lambda kv: kv[0])])
            print(f"[FID] {rel_tag_dir} | per-class: {txt}", flush=True)

        # merge gen_meta (if present)
        gen_meta = load_json_if_exists(tag_dir / "gen_meta.json")

        result = {
            "rel_tag_dir": rel_tag_dir,
            "tag": tag,
            "experiment": experiment,
            "gen_dir": fid_gen_dir.as_posix(),
            "n_real_all": int(n_r),
            "n_gen_total": int(n_gen_total),
            "n_gen_used": int(n_g),
            "fid_num_samples_arg": int(args.fid_num_samples),
            "fid_all": float(fid_all),
            "fid_per_class": fid_per_class if fid_per_class else None,
            "inception_ckpt": inception_ckpt.as_posix(),
            "inception_input_size": int(args.inception_input_size),
            "inception_mean": list(args.inception_mean),
            "inception_std": list(args.inception_std),
            "device": str(device),
            "fid_batch_size": int(args.fid_batch_size),
            "fid_num_workers": int(args.fid_num_workers),
            "gen_meta": gen_meta,
        }

        # append jsonl
        with results_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # append csv
        with results_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                rel_tag_dir, tag, experiment,
                fid_gen_dir.as_posix(), int(n_r), int(n_gen_total), int(n_g),
                float(fid_all), inception_ckpt.as_posix(), int(args.inception_input_size),
            ])

        # wandb log
        if wandb_run is not None and wandb_mod is not None:
            try:
                key_prefix = rel_tag_dir.replace("/", "__")
                log_dict = {
                    f"{key_prefix}/fid_all": float(fid_all),
                    f"{key_prefix}/n_gen_used": int(n_g),
                    f"{key_prefix}/n_gen_total": int(n_gen_total),
                }
                if fid_per_class:
                    for k, v in fid_per_class.items():
                        log_dict[f"{key_prefix}/fid_class/{k}"] = float(v)
                wandb_mod.log(log_dict, step=0)
            except Exception:
                pass

    if wandb_run is not None and wandb_mod is not None:
        try:
            wandb_mod.finish()
        except Exception:
            pass

    print("\n[Done] FID computation finished.", flush=True)
    print(f"  - results.jsonl: {results_jsonl}", flush=True)
    print(f"  - results.csv:   {results_csv}", flush=True)
    if args.cache_stats:
        print(f"  - stats cache:   {stats_cache_dir}", flush=True)


if __name__ == "__main__":
    main()
