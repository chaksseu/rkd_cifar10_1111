#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate ImageNet-1k val (ImageFolder) with torchvision Inception-v3 pretrained,
with CORRECT label mapping (ImageFolder alphabetical class order -> ImageNet class-id 0..999).

Fixes:
1) torchvision pretrained inception_v3 requires aux_logits=True (do NOT pass aux_logits=False)
2) Label-map ambiguity (e.g., "Cardigan" vs "cardigan") resolved via case-sensitive matching first.
"""

import argparse
import datetime as _dt
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torchvision.models import Inception_V3_Weights


def log(msg: str):
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _norm_keep_case(s: str) -> str:
    s = s.strip()
    s = s.replace("_", " ")
    s = s.replace("’", "'").replace("`", "'")
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_lower(s: str) -> str:
    return _norm_keep_case(s).lower()


def _strip_trailing_index_suffix(name: str) -> Tuple[str, Optional[int]]:
    # e.g., "crane2" -> ("crane", 2)
    m = re.match(r"^(.*?)(\d+)$", name.strip())
    if m:
        return m.group(1).strip(), int(m.group(2))
    return name.strip(), None


def build_name_maps(categories: List[str]):
    """
    Build reverse maps from (name variants) -> candidate ImageNet indices.

    We build:
      - case-sensitive map (whitespace-normalized, but case preserved)
      - lowercase map (fallback)
    for both full label string and primary token (before first comma).
    """
    cs = defaultdict(list)  # case-sensitive
    lc = defaultdict(list)  # lower-case

    for idx, cname in enumerate(categories):
        full_cs = _norm_keep_case(cname)
        prim_cs = _norm_keep_case(cname.split(",")[0])

        full_lc = full_cs.lower()
        prim_lc = prim_cs.lower()

        cs[full_cs].append(idx)
        cs[prim_cs].append(idx)

        lc[full_lc].append(idx)
        lc[prim_lc].append(idx)

    return dict(cs), dict(lc)


def pick_candidate(folder_raw: str, candidates: List[int], categories: List[str], suffix: Optional[int]) -> Tuple[int, str]:
    """
    Resolve candidate indices deterministically.
    Priority:
      1) exact case-sensitive match vs categories (normalized whitespace)
      2) exact lowercase match vs categories (normalized whitespace)
      3) if suffix provided (like crane2), use suffix-th candidate (1-indexed)
      4) if folder starts with upper, prefer candidate whose category starts with upper
         if folder starts with lower, prefer candidate whose category starts with lower
      5) fallback first candidate
    """
    fr_cs = _norm_keep_case(folder_raw)
    fr_lc = fr_cs.lower()

    # 1) exact cs match
    for i in candidates:
        if _norm_keep_case(categories[i]) == fr_cs:
            return i, "exact_cs"

    # 2) exact lc match
    for i in candidates:
        if _norm_lower(categories[i]) == fr_lc:
            return i, "exact_lc"

    # 3) suffix disambiguation
    if suffix is not None and 1 <= suffix <= len(candidates):
        return candidates[suffix - 1], f"suffix={suffix}"

    # 4) prefer case style
    if fr_cs and fr_cs[0].isalpha():
        want_upper = fr_cs[0].isupper()
        for i in candidates:
            cat = categories[i]
            if cat and cat[0].isalpha() and (cat[0].isupper() == want_upper):
                return i, "case_pref"

    # 5) fallback
    return candidates[0], "fallback_first"


def map_folder_to_imagenet_index(
    folder_name: str,
    cs_map: Dict[str, List[int]],
    lc_map: Dict[str, List[int]],
    categories: List[str],
) -> Tuple[Optional[int], str]:
    """
    Map ImageFolder class name -> ImageNet class-id index (0..999).
    Uses case-sensitive maps first, then lowercase fallback.
    """
    raw = folder_name.strip()
    base, suffix = _strip_trailing_index_suffix(raw)

    base_cs = _norm_keep_case(base)
    prim_cs = _norm_keep_case(base.split(",")[0])

    # Try case-sensitive full, then primary
    if base_cs in cs_map:
        cand = cs_map[base_cs]
        idx, why = pick_candidate(raw, cand, categories, suffix)
        return idx, f"cs_full:{why}"
    if prim_cs in cs_map:
        cand = cs_map[prim_cs]
        idx, why = pick_candidate(raw, cand, categories, suffix)
        return idx, f"cs_prim:{why}"

    # Lowercase fallback
    base_lc = base_cs.lower()
    prim_lc = prim_cs.lower()
    if base_lc in lc_map:
        cand = lc_map[base_lc]
        idx, why = pick_candidate(raw, cand, categories, suffix)
        return idx, f"lc_full:{why}"
    if prim_lc in lc_map:
        cand = lc_map[prim_lc]
        idx, why = pick_candidate(raw, cand, categories, suffix)
        return idx, f"lc_prim:{why}"

    return None, "unmapped"


class RemappedDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds: ImageFolder, orig_to_new: List[int], drop_unmapped: bool):
        self.base_ds = base_ds
        self.orig_to_new = orig_to_new
        self.drop_unmapped = drop_unmapped

        if drop_unmapped and any(v < 0 for v in orig_to_new):
            keep = []
            for i, (_, y) in enumerate(base_ds.samples):
                if orig_to_new[y] >= 0:
                    keep.append(i)
            self.keep_indices = keep
        else:
            self.keep_indices = None

    def __len__(self):
        return len(self.keep_indices) if self.keep_indices is not None else len(self.base_ds)

    def __getitem__(self, idx):
        if self.keep_indices is not None:
            idx = self.keep_indices[idx]
        x, y = self.base_ds[idx]
        ny = self.orig_to_new[y]
        return x, ny


def get_logits(out):
    # In eval with aux_logits=True, torchvision returns InceptionOutputs(logits=..., aux_logits=None)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--force_gray3", action="store_true")
    ap.add_argument("--resize_size", type=int, default=342)
    ap.add_argument("--crop_size", type=int, default=299)
    ap.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    ap.add_argument("--std", type=float, nargs=3, default=[0.229, 0.224, 0.225])
    ap.add_argument("--label_map", type=str, default="auto", choices=["auto", "none"])
    ap.add_argument("--strict_label_map", action="store_true",
                    help="If set, fail when any class cannot be mapped.")
    ap.add_argument("--drop_unmapped", action="store_true",
                    help="If set, drop samples whose class couldn't be mapped (only relevant if not strict).")
    ap.add_argument("--debug_map", action="store_true",
                    help="Print mapping for a few suspicious class names.")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    # ----- transforms -----
    def to_rgb(img):
        if args.force_gray3:
            return img.convert("L").convert("RGB")
        return img.convert("RGB")

    tf = T.Compose([
        T.Lambda(to_rgb),
        T.Resize(args.resize_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(args.crop_size),
        T.ToTensor(),
        T.Normalize(mean=args.mean, std=args.std),
    ])

    ds = ImageFolder(args.val_dir, transform=tf)

    log(f"[Info] Using torchvision Inception-v3 weights={Inception_V3_Weights.IMAGENET1K_V1}")
    log(f"Dataset: {len(ds)} val | classes={len(ds.classes)}")
    log(f"Class name head(20): {ds.classes[:20]}")
    log(f"Class name tail(5): {ds.classes[-5:]}")

    mapped_ds = ds
    orig_to_new = None

    if args.label_map == "auto":
        weights = Inception_V3_Weights.IMAGENET1K_V1
        categories = list(weights.meta.get("categories", []))
        if len(categories) != 1000:
            raise RuntimeError(f"weights.meta['categories'] length={len(categories)} (expected 1000).")

        cs_map, lc_map = build_name_maps(categories)

        orig_to_new = [-1] * len(ds.classes)
        unmapped = []
        ambiguous = []

        for orig_idx, folder_name in enumerate(ds.classes):
            idx, why = map_folder_to_imagenet_index(folder_name, cs_map, lc_map, categories)
            if idx is None:
                unmapped.append(folder_name)
            else:
                orig_to_new[orig_idx] = idx
                # mark ambiguous only if multiple candidates existed in the matched key
                if "fallback_first" in why:
                    ambiguous.append((folder_name, idx, why))

        n_ok = sum(1 for v in orig_to_new if v >= 0)
        log(f"[LabelMap:auto] mapped {n_ok}/{len(ds.classes)} class folders to ImageNet indices.")

        if ambiguous:
            log(f"[LabelMap:auto] ambiguous-like (fallback) matches: {len(ambiguous)} (show up to 10)")
            for a in ambiguous[:10]:
                log(f"  - {a[0]} -> {a[1]} ({a[2]})")

        if unmapped:
            log(f"[LabelMap:auto] UNMAPPED classes: {len(unmapped)} (show up to 20)")
            for u in unmapped[:20]:
                log(f"  - {u}")
            if args.strict_label_map:
                raise RuntimeError("Some classes could not be mapped. Fix folder names or disable strict mode.")

        if args.debug_map:
            for name in ["Cardigan, Cardigan Welsh corgi", "cardigan", "crane", "crane2"]:
                if name in ds.classes:
                    oi = ds.class_to_idx[name]
                    log(f"[DebugMap] '{name}' ImageFolderIdx={oi} -> ImageNetIdx={orig_to_new[oi]}")

        mapped_ds = RemappedDataset(ds, orig_to_new, drop_unmapped=(args.drop_unmapped and not args.strict_label_map))

    # ----- loader -----
    persistent = args.workers > 0
    loader = DataLoader(
        mapped_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=4 if args.workers > 0 else None,
        drop_last=False,
    )

    log(f"Device: {args.device} | amp={args.amp} | force_gray3={args.force_gray3} | label_map={args.label_map}")
    log(f"Loader: bs={args.batch_size} workers={args.workers} pin_memory=True persistent_workers={persistent} prefetch_factor={(4 if args.workers>0 else None)}")

    # ----- model -----
    weights = Inception_V3_Weights.IMAGENET1K_V1
    # IMPORTANT: do NOT pass aux_logits=False with pretrained weights (torchvision will error).
    model = models.inception_v3(weights=weights)  # aux_logits=True internally as required
    model.eval().to(device)

    # ----- eval -----
    top1 = 0
    top5 = 0
    n = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if args.amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = get_logits(model(x))
            else:
                logits = get_logits(model(x))

            # Top-1
            pred1 = logits.argmax(dim=1)
            top1 += (pred1 == y).sum().item()

            # Top-5
            k = 5
            pred5 = logits.topk(k, dim=1).indices
            top5 += pred5.eq(y.view(-1, 1)).any(dim=1).sum().item()

            n += y.numel()

    top1_acc = 100.0 * top1 / max(1, n)
    top5_acc = 100.0 * top5 / max(1, n)
    log(f"[Val] N={n} Top1={top1_acc:.2f}% Top5={top5_acc:.2f}%")


if __name__ == "__main__":
    main()
