#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

@torch.no_grad()
def compute_mean_std(train_dir: str, batch_size: int, workers: int, input_size: int = 299):
    # 학습과 동일하게 resize/crop을 적용할지 여부는 선택사항.
    # "실제로 모델에 들어가는 분포" 기준이면 아래처럼 Resize+CenterCrop을 맞추는 게 일관적입니다.
    tf = T.Compose([
        T.Resize(input_size + 32, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(input_size),
        T.ToTensor(),  # -> [0,1]
    ])

    ds = ImageFolder(train_dir, transform=tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    n_pixels = 0
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sumsq = torch.zeros(3, dtype=torch.float64)

    for x, _ in loader:
        # x: (B,3,H,W) in [0,1]
        b, c, h, w = x.shape
        pixels = b * h * w
        n_pixels += pixels
        channel_sum += x.double().sum(dim=(0, 2, 3))
        channel_sumsq += (x.double() ** 2).sum(dim=(0, 2, 3))

    mean = channel_sum / n_pixels
    
    var = channel_sumsq / n_pixels - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=0.0) + 1e-12)
    return mean.tolist(), std.tolist(), len(ds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, default="./imagenet1k_gray3_export/gray3/train")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--input_size", type=int, default=299)
    args = ap.parse_args()

    mean, std, n = compute_mean_std(args.train_dir, args.batch_size, args.workers, args.input_size)
    print(f"num_images={n}")
    print(f"mean={mean}")
    print(f"std ={std}")
    # gray3라면 mean/std가 3채널 거의 동일하게 나올 겁니다.

if __name__ == "__main__":
    main()
