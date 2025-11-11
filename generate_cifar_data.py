#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CIFAR-10 -> PNG (RGB & Gray-3ch via strictly linear transform) + stats
- 비선형 과정 제거: 감마 보정/클리핑/비선형 색공간 변환 없음
- Grayscale = w^T * [R,G,B] (가중합, 선형), 이후 3채널 복제(gray3)
- 통계는 변환 직후의 float[0,1] 선형 값으로 계산
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import CIFAR10


# ------------------------ Stats ------------------------

class StatsAccumulator:
    """채널별 통계 누적 (mean, std, min, max) for [0,1] scale."""
    def __init__(self, num_channels: int):
        self.C = int(num_channels)
        self.count = 0
        self.sum = np.zeros(self.C, dtype=np.float64)
        self.sumsq = np.zeros(self.C, dtype=np.float64)
        self.minv = np.full(self.C, np.inf, dtype=np.float64)
        self.maxv = np.full(self.C, -np.inf, dtype=np.float64)

    def update(self, arr_float01: np.ndarray):
        """
        arr_float01: shape (H, W, C) or (H, W) in [0,1] float
        (비선형 처리 없이 그대로 사용)
        """
        if arr_float01.ndim == 2:
            arr = arr_float01[..., None]
        elif arr_float01.ndim == 3:
            arr = arr_float01
        else:
            raise ValueError(f"Expected 2D/3D array, got shape {arr_float01.shape}")

        if arr.shape[-1] != self.C:
            raise ValueError(f"StatsAccumulator C={self.C}, but got C={arr.shape[-1]}")

        flat = arr.reshape(-1, self.C).astype(np.float64, copy=False)
        self.count += flat.shape[0]
        self.sum += flat.sum(axis=0)
        self.sumsq += (flat ** 2).sum(axis=0)

        # min/max (선형 값 그대로)
        self.minv = np.minimum(self.minv, flat.min(axis=0))
        self.maxv = np.maximum(self.maxv, flat.max(axis=0))

    def to_dict(self) -> Dict:
        eps = 1e-12
        denom = max(self.count, 1)
        mean = self.sum / denom
        var = self.sumsq / denom - mean ** 2
        var = np.maximum(var, 0.0)
        std = np.sqrt(var + eps)
        return {
            "num_channels": int(self.C),
            "count_pixels": int(self.count),
            "mean_01": mean.tolist(),
            "std_01": std.tolist(),
            "min_01": self.minv.tolist(),
            "max_01": self.maxv.tolist(),
            "mean_255": (mean * 255.0).tolist(),
            "std_255": (std * 255.0).tolist(),
            "min_255": (self.minv * 255.0).tolist(),
            "max_255": (self.maxv * 255.0).tolist(),
        }


def save_stats(folder: Path, acc: StatsAccumulator):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "_stats.json", "w", encoding="utf-8") as f:
        json.dump(acc.to_dict(), f, ensure_ascii=False, indent=2)


# ------------------------ Linear transform ------------------------

def rgb_to_gray_linear(rgb01: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    RGB(0..1, HxWx3) -> Gray(0..1, HxW) 선형변환
    - weights: shape (3,), w>=0, sum(w)=1
    - 비선형 연산(감마/클리핑 등) 없음
    """
    assert rgb01.ndim == 3 and rgb01.shape[-1] == 3, f"expected (H,W,3), got {rgb01.shape}"
    # 가중합 (선형)
    H, W, _ = rgb01.shape
    gray = (rgb01.reshape(-1, 3) @ weights.reshape(3, 1)).reshape(H, W)
    # 가중치가 비음수이고 합=1이면 convex combination → 값은 자동으로 [0,1] 범위
    return gray


# ------------------------ IO helpers ------------------------

def ensure_dirs(out_root: Path, modality: str, split: str, classes: List[str], create_classes: bool):
    base = out_root / modality / split
    base.mkdir(parents=True, exist_ok=True)
    if create_classes:
        for c in classes:
            (base / c).mkdir(parents=True, exist_ok=True)


def save_png_from_float01(arr01: np.ndarray, path: Path):
    """
    float[0,1] 배열을 PNG로 저장.
    (양자화는 저장 단계에서 불가피하지만, 변환/통계는 이미 끝난 상태)
    """
    img = Image.fromarray((arr01 * 255.0).astype(np.uint8))
    img.save(path, format="PNG", optimize=True, compress_level=4)


# ------------------------ Main pipeline ------------------------

def process_split(dataset: CIFAR10, out_root: Path, split: str,
                  do_rgb: bool, do_gray3: bool, overwrite: bool, weights: np.ndarray):
    classes = dataset.classes
    ensure_dirs(out_root, "rgb", split, classes, create_classes=do_rgb)
    ensure_dirs(out_root, "gray3", split, classes, create_classes=do_gray3)

    rgb_split_acc = StatsAccumulator(3) if do_rgb else None
    gray3_split_acc = StatsAccumulator(3) if do_gray3 else None
    rgb_class_acc = {c: StatsAccumulator(3) for c in classes} if do_rgb else {}
    gray3_class_acc = {c: StatsAccumulator(3) for c in classes} if do_gray3 else {}

    for idx in tqdm(range(len(dataset)), desc=f"[{split}] Saving & Stats"):
        arr_u8 = dataset.data[idx]                 # (32,32,3) uint8 (sRGB 값 범위 가정)
        label = int(dataset.targets[idx])
        cname = classes[label]
        fname = f"{idx:06d}.png"

        # 선형 공간 값 (0..1)
        rgb01 = arr_u8.astype(np.float32) / 255.0  # 스케일 변경만 (선형)

        # 1) RGB 저장/통계 (선형 값 기준으로 통계)
        if do_rgb:
            rgb_path = out_root / "rgb" / split / cname / fname
            if overwrite or (not rgb_path.exists()):
                save_png_from_float01(rgb01, rgb_path)
            rgb_split_acc.update(rgb01)
            rgb_class_acc[cname].update(rgb01)

        # 2) Gray-3ch 저장/통계 (선형 변환 + 채널복제만)
        if do_gray3:
            gray01 = rgb_to_gray_linear(rgb01, weights)          # (H,W)
            gray3_01 = np.repeat(gray01[..., None], 3, axis=-1)  # (H,W,3)
            gray3_path = out_root / "gray3" / split / cname / fname
            if overwrite or (not gray3_path.exists()):
                save_png_from_float01(gray3_01, gray3_path)
            gray3_split_acc.update(gray3_01)
            gray3_class_acc[cname].update(gray3_01)

    # split 통계 저장
    if do_rgb:
        save_stats(out_root / "rgb" / split, rgb_split_acc)
    if do_gray3:
        save_stats(out_root / "gray3" / split, gray3_split_acc)

    # 클래스별 통계 저장
    if do_rgb:
        for c in classes:
            save_stats(out_root / "rgb" / split / c, rgb_class_acc[c])
    if do_gray3:
        for c in classes:
            save_stats(out_root / "gray3" / split / c, gray3_class_acc[c])


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 -> PNG (RGB & Gray-3ch via linear transform) + stats (no nonlinear ops)")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--out_root", type=str, default="./cifar10_png_linear_only")
    parser.add_argument("--no_rgb", action="store_true", help="RGB 저장/통계 생략")
    parser.add_argument("--no_gray3", action="store_true", help="Gray-3ch 저장/통계 생략")
    parser.add_argument("--overwrite", action="store_true", help="기존 PNG 덮어쓰기")
    parser.add_argument("--weights", type=float, nargs=3, default=[0.2989, 0.5870, 0.1140],
                        help="grayscale 선형 가중치 wR wG wB (w>=0, sum=1 권장)")
    parser.add_argument("--mirror", type=str, default=None, help="torchvision>=0.18에서 지원")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 가중치 검증 (선형/볼록 결합)
    w = np.array(args.weights, dtype=np.float32)
    if np.any(w < 0):
        raise ValueError("weights는 비음수여야 합니다. 예: --weights 0.2989 0.5870 0.1140")
    s = w.sum()
    if s <= 0:
        raise ValueError("weights 합이 0보다 커야 합니다.")
    w = w / s  # 합=1로 정규화 (선형성 유지)

    # 다운로드
    common_kwargs = {}
    if args.mirror is not None:
        try:
            common_kwargs["mirror"] = args.mirror
        except TypeError:
            print("[WARN] 현재 torchvision 버전은 --mirror를 지원하지 않습니다.")

    trainset = CIFAR10(root=str(args.data_root), train=True,  download=True, transform=None, **common_kwargs)
    testset  = CIFAR10(root=str(args.data_root), train=False, download=True, transform=None, **common_kwargs)

    do_rgb = not args.no_rgb
    do_gray3 = not args.no_gray3

    (out_root / "rgb").mkdir(parents=True, exist_ok=True)
    (out_root / "gray3").mkdir(parents=True, exist_ok=True)

    process_split(trainset, out_root, "train", do_rgb, do_gray3, args.overwrite, w)
    process_split(testset,  out_root, "test",  do_rgb, do_gray3, args.overwrite, w)

    print("\n✅ Done! 예시 구조:")
    print(out_root.resolve())
    print("└── rgb/train|test/<class>/{000000.png,...,_stats.json}, _stats.json")
    print("└── gray3/train|test/<class>/{000000.png,...,_stats.json}, _stats.json")
    print(f"\n[Info] Linear weights (normalized): {w.tolist()}")
    print("[Note] 변환/통계는 순수 선형 연산만 사용했습니다. (PNG 저장 단계의 양자화는 예외)")
    

if __name__ == "__main__":
    main()
