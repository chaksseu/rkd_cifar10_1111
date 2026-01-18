#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repack existing teacher bank (.pt shards) into:
  OUT_ROOT/{metric}/shards/shard_XXXXXX.safetensors

- Input layout (example):
  IN_ROOT/ddim_steps040_eta0/shards/shard_*.pt
  ...
  IN_ROOT/ddim_steps060_eta0/shards/shard_*.pt

- Each input .pt is expected to contain:
  "z": (N,3,H,W) float32 (or float16 etc.)
  "feats_pixel":  (N,Dp) float32
  "feats_clip":   (N,Dc) float32
  "feats_dinov3": (N,Dd) float32
  optionally "steps": int

We do NOT regenerate anything. We just re-pack into safetensors with better sharding/mixing.

Requires:
  pip install safetensors tqdm
"""

import argparse
import json
import os
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from tqdm import tqdm
from safetensors.torch import save_file


METRIC2KEY = {
    "pixel":  "feats_pixel",
    "clip":   "feats_clip",
    "dinov3": "feats_dinov3",
}


def parse_steps_from_dirname(name: str) -> Optional[int]:
    # supports: ddim_steps040_eta0 / ddim_steps40_eta0 etc.
    m = re.search(r"ddim_steps(\d+)_eta", name)
    if m:
        return int(m.group(1))
    return None


def list_input_pt_shards(in_root: Path) -> List[Tuple[int, Path]]:
    """Return list of (steps, pt_path) across all step folders."""
    items: List[Tuple[int, Path]] = []
    for step_dir in sorted(in_root.glob("ddim_steps*_eta*")):
        if not step_dir.is_dir():
            continue
        steps = parse_steps_from_dirname(step_dir.name)
        if steps is None:
            continue
        shard_dir = step_dir / "shards"
        if not shard_dir.exists():
            continue
        for p in sorted(shard_dir.glob("shard_*.pt")):
            items.append((steps, p))
    return items


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


class TensorBuffer:
    """
    Buffer that stores list-of-tensors and can pop exactly n rows across boundaries.
    Keeps everything on CPU.
    """
    def __init__(self):
        self.parts: List[torch.Tensor] = []
        self.num_rows: int = 0

    def append(self, x: torch.Tensor):
        if x is None:
            return
        if not isinstance(x, torch.Tensor):
            raise TypeError("append expects torch.Tensor")
        if x.ndim == 0:
            x = x.view(1)
        self.parts.append(x)
        self.num_rows += int(x.shape[0])

    def pop_rows(self, n: int) -> torch.Tensor:
        """Pop exactly n rows (along dim=0) and return concatenated tensor."""
        n = int(n)
        if n <= 0:
            raise ValueError("n must be positive")
        if self.num_rows < n:
            raise RuntimeError(f"Buffer underflow: have {self.num_rows}, need {n}")

        out_parts: List[torch.Tensor] = []
        remain = n
        while remain > 0:
            cur = self.parts[0]
            cur_n = int(cur.shape[0])
            if cur_n <= remain:
                out_parts.append(cur)
                self.parts.pop(0)
                remain -= cur_n
            else:
                out_parts.append(cur[:remain])
                self.parts[0] = cur[remain:]
                remain = 0

        out = torch.cat(out_parts, dim=0) if len(out_parts) > 1 else out_parts[0]
        self.num_rows -= n
        return out

    def flush_all(self) -> Optional[torch.Tensor]:
        if self.num_rows == 0:
            return None
        out = torch.cat(self.parts, dim=0) if len(self.parts) > 1 else self.parts[0]
        self.parts = []
        self.num_rows = 0
        return out


def repack(
    in_root: Path,
    out_root: Path,
    metrics: List[str],
    out_shard_samples: int,
    seed: int,
    overwrite: bool,
    strict_keys: bool,
    keep_z: bool = True,
):
    assert out_shard_samples > 0

    # Collect all input shard paths
    shard_items = list_input_pt_shards(in_root)
    if len(shard_items) == 0:
        raise FileNotFoundError(f"No input shards found under: {in_root}")

    rng = random.Random(int(seed))
    rng.shuffle(shard_items)  # key: global shuffle to mix steps across output shards

    # Output dirs per metric
    for m in metrics:
        if m not in METRIC2KEY:
            raise ValueError(f"Unknown metric '{m}'. Choose from {list(METRIC2KEY.keys())}")
        ensure_dir(out_root / m / "shards")

    # Buffers per metric
    buf_z: Dict[str, TensorBuffer] = {m: TensorBuffer() for m in metrics}
    buf_f: Dict[str, TensorBuffer] = {m: TensorBuffer() for m in metrics}
    buf_s: Dict[str, TensorBuffer] = {m: TensorBuffer() for m in metrics}

    # Shard counters per metric
    out_idx: Dict[str, int] = {m: 0 for m in metrics}
    total_written: Dict[str, int] = {m: 0 for m in metrics}

    # Manifest info
    manifest: Dict[str, dict] = {m: {
        "metric": m,
        "feat_key": METRIC2KEY[m],
        "out_shard_samples": int(out_shard_samples),
        "num_shards": 0,
        "num_samples": 0,
        "dtype_z": "unknown",
        "dtype_feat": "unknown",
        "steps_dtype": "int16",
        "source_root": str(in_root),
        "note": "Random-mixed across steps by global shuffling of input pt shards. fp32 preserved unless input differs.",
    } for m in metrics}

    pbar = tqdm(shard_items, desc="Reading .pt shards", dynamic_ncols=True)
    for steps_from_dir, pt_path in pbar:
        d = torch.load(pt_path, map_location="cpu")

        if "z" not in d:
            raise KeyError(f"Missing 'z' in {pt_path}")
        z = d["z"]
        if not isinstance(z, torch.Tensor):
            raise TypeError(f"'z' is not a Tensor in {pt_path}")

        # steps: prefer per-file stored steps, fallback to directory name
        steps_file = int(d.get("steps", steps_from_dir))
        if steps_file <= 0:
            steps_file = steps_from_dir

        # shuffle within this input shard (improves mixing)
        n = int(z.shape[0])
        perm = torch.randperm(n)  # uses global torch RNG; fine for offline pack

        z = z[perm].contiguous()
        steps_vec = torch.full((n,), steps_file, dtype=torch.int16)

        for m in metrics:
            feat_key = METRIC2KEY[m]
            if feat_key not in d:
                if strict_keys:
                    raise KeyError(f"Missing '{feat_key}' in {pt_path}")
                else:
                    continue

            f = d[feat_key]
            if not isinstance(f, torch.Tensor):
                raise TypeError(f"'{feat_key}' is not a Tensor in {pt_path}")
            f = f[perm].contiguous()

            # Record dtype once
            if manifest[m]["dtype_z"] == "unknown":
                manifest[m]["dtype_z"] = str(z.dtype)
            if manifest[m]["dtype_feat"] == "unknown":
                manifest[m]["dtype_feat"] = str(f.dtype)

            if keep_z:
                buf_z[m].append(z)
            buf_f[m].append(f)
            buf_s[m].append(steps_vec)

            # Flush full output shards as much as possible
            while buf_f[m].num_rows >= out_shard_samples:
                take = out_shard_samples

                z_out = buf_z[m].pop_rows(take) if keep_z else None
                f_out = buf_f[m].pop_rows(take)
                s_out = buf_s[m].pop_rows(take)

                out_dir = out_root / m / "shards"
                out_path = out_dir / f"shard_{out_idx[m]:06d}.safetensors"
                if out_path.exists() and (not overwrite):
                    raise FileExistsError(f"Output exists (use --overwrite): {out_path}")

                tensors = {}
                if keep_z:
                    tensors["z"] = z_out
                tensors[feat_key] = f_out
                tensors["steps"] = s_out

                meta = {
                    "metric": m,
                    "feat_key": feat_key,
                    "packed_from": "pt_shards",
                    "source_pt_example": str(pt_path),
                    "mixed_steps": "true",
                }
                save_file(tensors, str(out_path), metadata=meta)

                out_idx[m] += 1
                total_written[m] += take
                manifest[m]["num_shards"] = int(out_idx[m])
                manifest[m]["num_samples"] = int(total_written[m])

        # help tqdm show something useful
        pbar.set_postfix({
            "example_steps": steps_file,
            "buf_clip": buf_f["clip"].num_rows if "clip" in buf_f else 0,
            "buf_dino": buf_f["dinov3"].num_rows if "dinov3" in buf_f else 0,
            "buf_pixel": buf_f["pixel"].num_rows if "pixel" in buf_f else 0,
        })

    # Flush remainders (last partial shards)
    for m in metrics:
        remain = buf_f[m].num_rows
        if remain == 0:
            continue

        z_out = buf_z[m].flush_all() if keep_z else None
        f_out = buf_f[m].flush_all()
        s_out = buf_s[m].flush_all()
        assert f_out is not None and s_out is not None

        out_dir = out_root / m / "shards"
        out_path = out_dir / f"shard_{out_idx[m]:06d}.safetensors"
        if out_path.exists() and (not overwrite):
            raise FileExistsError(f"Output exists (use --overwrite): {out_path}")

        feat_key = METRIC2KEY[m]
        tensors = {}
        if keep_z:
            tensors["z"] = z_out
        tensors[feat_key] = f_out
        tensors["steps"] = s_out

        meta = {
            "metric": m,
            "feat_key": feat_key,
            "packed_from": "pt_shards",
            "mixed_steps": "true",
            "note": "last_partial_shard",
        }
        save_file(tensors, str(out_path), metadata=meta)

        out_idx[m] += 1
        total_written[m] += int(remain)
        manifest[m]["num_shards"] = int(out_idx[m])
        manifest[m]["num_samples"] = int(total_written[m])

    # Write manifest per metric + global manifest
    ensure_dir(out_root)
    for m in metrics:
        meta_path = out_root / m / "manifest.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(manifest[m], f, indent=2, ensure_ascii=False)

    global_meta = {
        "in_root": str(in_root),
        "out_root": str(out_root),
        "metrics": metrics,
        "out_shard_samples": int(out_shard_samples),
        "seed": int(seed),
        "keep_z": bool(keep_z),
        "note": "metric-wise safetensors shards with randomized mixing across steps via global shuffle",
    }
    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(global_meta, f, indent=2, ensure_ascii=False)

    print("\n[Done] Repacked teacher bank.")
    for m in metrics:
        print(f"  - {m}: samples={manifest[m]['num_samples']:,}  shards={manifest[m]['num_shards']:,}  saved_to={out_root/m/'shards'}")


def build_argparser():
    p = argparse.ArgumentParser("Repack teacher bank (.pt shards) -> metric-wise safetensors shards (random-mixed)")

    p.add_argument("--in_root", type=str, required=True, help="Input bank root (contains ddim_stepsXXX_eta*/shards/shard_*.pt)")
    p.add_argument("--out_root", type=str, required=True, help="Output root for safetensors bank")
    p.add_argument("--metrics", type=str, nargs="+", default=["clip", "dinov3", "pixel"],
                   choices=["clip", "dinov3", "pixel"],
                   help="Which metrics to repack")
    p.add_argument("--out_shard_samples", type=int, default=8192,
                   help="How many samples per output safetensors shard. Smaller => more mixing across steps.")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed for input shard order")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist")
    p.add_argument("--strict_keys", action="store_true",
                   help="If set, error when an input pt shard misses a requested feats_* key. Otherwise skip.")
    p.add_argument("--no_keep_z", action="store_true",
                   help="If set, do NOT store z in output (feat only). Default stores z + feat + steps.")
    return p


def main():
    args = build_argparser().parse_args()
    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    if not in_root.exists():
        raise FileNotFoundError(f"in_root not found: {in_root}")

    repack(
        in_root=in_root,
        out_root=out_root,
        metrics=args.metrics,
        out_shard_samples=int(args.out_shard_samples),
        seed=int(args.seed),
        overwrite=bool(args.overwrite),
        strict_keys=bool(args.strict_keys),
        keep_z=(not args.no_keep_z),
    )


if __name__ == "__main__":
    main()
