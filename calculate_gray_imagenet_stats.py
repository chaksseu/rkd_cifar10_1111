#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def setup_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("mean_std")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


@torch.no_grad()
def compute_mean_std(
    train_dir: str,
    batch_size: int,
    workers: int,
    input_size: int,
    logger: logging.Logger,
):
    tf = T.Compose([
        T.Resize(input_size + 32, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(input_size),
        T.ToTensor(),  # [0,1]
    ])

    ds = ImageFolder(train_dir, transform=tf)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    logger.info(f"Dataset loaded: {len(ds)} images")
    logger.info(f"Batch size: {batch_size}, Workers: {workers}")
    logger.info(f"Input size: {input_size}x{input_size}")

    n_pixels = 0
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sumsq = torch.zeros(3, dtype=torch.float64)

    pbar = tqdm(loader, desc="Computing mean/std", unit="batch")
    for i, (x, _) in enumerate(pbar):
        # x: (B,3,H,W)
        b, c, h, w = x.shape
        pixels = b * h * w

        n_pixels += pixels
        channel_sum += x.double().sum(dim=(0, 2, 3))
        channel_sumsq += (x.double() ** 2).sum(dim=(0, 2, 3))

        if (i + 1) % 50 == 0:
            logger.info(
                f"Processed {min((i + 1) * batch_size, len(ds))}/{len(ds)} images "
                f"({n_pixels:,} pixels)"
            )

    mean = channel_sum / n_pixels
    var = channel_sumsq / n_pixels - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=0.0) + 1e-12)

    return mean.tolist(), std.tolist(), len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, default="./imagenet1k_export/gray3/train")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--input_size", type=int, default=299)
    ap.add_argument("--log_file", type=str, default="0102_gray_imagenets_compute_mean_std.log")
    args = ap.parse_args()

    logger = setup_logger(Path(args.log_file))

    logger.info("===== Mean / Std computation started =====")
    logger.info(f"Train dir: {args.train_dir}")

    mean, std, n = compute_mean_std(
        args.train_dir,
        args.batch_size,
        args.workers,
        args.input_size,
        logger,
    )

    logger.info("===== Computation finished =====")
    logger.info(f"num_images = {n}")
    logger.info(f"mean = {mean}")
    logger.info(f"std  = {std}")

    print(f"num_images={n}")
    print(f"mean={mean}")
    print(f"std ={std}")


if __name__ == "__main__":
    main()
