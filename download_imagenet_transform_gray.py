#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export Hugging Face ImageNet-1k (gated) to ImageFolder structure,
optionally converting to linear grayscale (gray3) via a strictly linear transform.
Supports internal Multi-processing.
"""

import os
import json
import argparse
import multiprocessing  # [수정됨] 멀티프로세싱 모듈 추가
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets import load_dataset


# ------------------------ Utils ------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def get_auth_kwargs():
    import inspect
    sig = inspect.signature(load_dataset)
    if "token" in sig.parameters:
        return {"token": True}
    if "use_auth_token" in sig.parameters:
        return {"use_auth_token": True}
    return {}

def pil_to_rgb01(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    arr_u8 = np.asarray(img, dtype=np.uint8)
    return arr_u8.astype(np.float32) / 255.0

def rgb01_to_gray3_u8(rgb01: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray01 = (rgb01.reshape(-1, 3) @ w.reshape(3, 1)).reshape(rgb01.shape[0], rgb01.shape[1])
    gray3_01 = np.repeat(gray01[..., None], 3, axis=-1)
    gray3_u8 = np.clip(gray3_01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return gray01, gray3_u8

def save_png_u8(arr_u8: np.ndarray, path: Path, compress_level: int = 4):
    img = Image.fromarray(arr_u8, mode="RGB")
    img.save(path, format="PNG", optimize=True, compress_level=int(compress_level))

def label_to_classname(ds, y: int) -> str:
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
    """
    개별 프로세스가 수행하는 작업 함수
    """
    auth_kwargs = get_auth_kwargs()
    
    # 각 프로세스마다 별도의 데이터셋 인스턴스 로드
    ds = load_dataset(ds_name, split=split_name, streaming=streaming, **auth_kwargs)

    if num_shards > 1:
        ds = ds.shard(num_shards=num_shards, index=shard_id)

    split_dir = "train" if split_name == "train" else "val"
    base_gray3 = out_root / "gray3" / split_dir
    base_rgb   = out_root / "rgb"   / split_dir
    
    # 주의: 멀티프로세스 환경에서 makedirs 충돌 방지를 위해 exist_ok=True 필수 (이미 적용됨)
    if do_gray3:
        ensure_dir(base_gray3)
    if do_rgb:
        ensure_dir(base_rgb)

    it = iter(ds)
    
    # 멀티프로세스 환경에서 tqdm이 겹쳐 보일 수 있으나, position 인자를 자동 관리하도록 둠
    desc_str = f"[PID:{os.getpid()}] {split_name} shard {shard_id}/{num_shards}"
    pbar = tqdm(total=(max_images if max_images > 0 else None), desc=desc_str, position=shard_id, leave=False)

    n = 0
    errors = 0

    for ex in it:
        if max_images > 0 and n >= max_images:
            break

        try:
            img = ex["image"]
            y = int(ex["label"])
            cls = label_to_classname(ds, y)
            rgb01 = pil_to_rgb01(img)
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

    pbar.close()
    return {"split": split_name, "shard_id": shard_id, "num_exported": n, "errors": errors}

# [추가됨] multiprocessing.Pool.map용 래퍼 함수
def worker_wrapper(kwargs: Dict[str, Any]):
    return export_split(**kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="ILSVRC/imagenet-1k",
                    help="Hugging Face dataset name (gated).")
    ap.add_argument("--out_root", type=str, default="./imagenet1k_export",
                    help="Output root directory.")
    ap.add_argument("--no_gray3", action="store_true", help="Do not export gray3.")
    ap.add_argument("--export_rgb", action="store_true", help="Also export RGB images.")
    ap.add_argument("--weights", type=float, nargs=3, default=[0.2989, 0.5870, 0.1140],
                    help="Linear grayscale weights.")
    ap.add_argument("--streaming", action="store_true", default=True,
                    help="Use streaming=True (Default: True).")
    # shard_id 인자는 자동 생성되므로 제거하거나 무시해도 됨
    ap.add_argument("--num_shards", type=int, default=2,
                    help="Number of processes to run in parallel.")
    ap.add_argument("--max_images", type=int, default=0,
                    help="Export only first N images per split per shard.")
    ap.add_argument("--compress_level", type=int, default=4, help="PNG compress_level.")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ensure_dir(out_root)

    w = np.array(args.weights, dtype=np.float32)
    if np.any(w < 0): raise ValueError("weights must be non-negative.")
    s = float(w.sum())
    if s <= 0: raise ValueError("weights sum must be > 0.")
    w = w / s

    do_gray3 = not args.no_gray3
    do_rgb = bool(args.export_rgb)

    # 1. 실행할 작업 리스트 생성
    tasks = []
    
    # Train과 Validation을 모두 병렬로 처리
    splits_to_process = ["train", "validation"]
    
    print(f"Preparing tasks for {args.num_shards} processes...")
    
    for split in splits_to_process:
        for i in range(args.num_shards):
            task_args = {
                "ds_name": args.dataset,
                "split_name": split,
                "out_root": out_root,
                "do_gray3": do_gray3,
                "do_rgb": do_rgb,
                "weights": w,
                "streaming": bool(args.streaming),
                "shard_id": i,               # 0 ~ num_shards-1
                "num_shards": int(args.num_shards),
                "max_images": int(args.max_images),
                "compress_level": int(args.compress_level),
            }
            tasks.append(task_args)

    print(f"Total tasks: {len(tasks)}. Starting multiprocessing pool...")

    # 2. 멀티 프로세싱 실행
    # processes 개수는 num_shards와 동일하게 설정 (각 샤드마다 1 프로세스)
    with multiprocessing.Pool(processes=args.num_shards) as pool:
        results = pool.map(worker_wrapper, tasks)

    # 3. 결과 저장
    manifest = {
        "dataset": args.dataset,
        "out_root": str(out_root.resolve()),
        "gray3": do_gray3,
        "rgb": do_rgb,
        "weights_normalized": w.tolist(),
        "streaming": bool(args.streaming),
        "num_shards": int(args.num_shards),
        "results": results,
    }
    
    with (out_root / "_export_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\nAll Done.")

if __name__ == "__main__":
    # Windows/Mac 등에서 안전한 멀티프로세싱을 위해 필수
    multiprocessing.freeze_support() 
    main()