#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train Inception-v3 from scratch on exported ImageNet-1k gray3 ImageFolder + wandb logging,
with separate console log interval vs wandb log interval.

Expected folder layout:
  <data_root>/gray3/train/<class_name>/*.png
  <data_root>/gray3/val/<class_name>/*.png

Example:
  python train_inception_v3_gray3_scratch_wandb.py \
    --train_dir ./imagenet1k_gray3_export/gray3/train \
    --val_dir   ./imagenet1k_gray3_export/gray3/val \
    --output_dir ./inceptionv3_gray3_scratch \
    --device cuda:0 --epochs 90 --batch_size 256 --lr 0.4 \
    --aux_logits --amp \
    --wandb --wandb_project imagenet-gray3 --wandb_name incv3-scratch-gray3 \
    --log_interval 50 --wandb_log_interval 200
"""

import os
import json
import time
import math
import argparse
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


# -------------------------
# Utils
# -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def resolve_device(device_str: str) -> torch.device:
    try:
        dev = torch.device(device_str)
    except Exception:
        return torch.device("cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev

def set_seed(seed: int):
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_json(path: Path, obj: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)) -> Dict[str, float]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # (N,maxk)
    pred = pred.t()  # (maxk, N)
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    out = {}
    N = targets.size(0)
    for k in topk:
        c = correct[:k].reshape(-1).float().sum().item()
        out[f"top{k}"] = 100.0 * c / max(N, 1)
    return out

def unpack_inception_outputs(outputs):
    """
    torchvision inception_v3 can return:
      - train mode with aux_logits=True: InceptionOutputs(logits, aux_logits)
      - eval mode: logits tensor
    """
    if hasattr(outputs, "logits"):
        logits = outputs.logits
        aux = getattr(outputs, "aux_logits", None)
        return logits, aux
    if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        return outputs[0], outputs[1]
    return outputs, None

def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = float(lr)

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


# -------------------------
# LR schedule (Warmup + Cosine)
# -------------------------

def lr_at_epoch(base_lr: float, epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(warmup_epochs)
    t = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    t = min(max(t, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


# -------------------------
# Train / Eval
# -------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
    aux_weight: float,
    label_smoothing: float,
    max_grad_norm: float,
    log_interval: int,
    wandb_log_interval: int,
    wandb_run=None,
):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing)).to(device)

    t0 = time.time()
    n_seen = 0
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0

    for it, (x, y) in enumerate(loader, start=1):
        global_step += 1

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        else:
            autocast_ctx = torch.autocast(device_type="cpu", enabled=False)

        with autocast_ctx:
            outputs = model(x)
            logits, aux = unpack_inception_outputs(outputs)
            loss = criterion(logits, y)
            if aux is not None:
                loss = loss + float(aux_weight) * criterion(aux, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        bs = x.size(0)
        n_seen += bs
        loss_sum += float(loss.detach().item()) * bs

        acc = accuracy_topk(logits.detach(), y.detach(), topk=(1, 5))
        top1_sum += acc["top1"] * bs
        top5_sum += acc["top5"] * bs

        # Console logging
        if log_interval > 0 and (it % log_interval == 0):
            dt = time.time() - t0
            cur_loss = loss_sum / max(1, n_seen)
            cur_top1 = top1_sum / max(1, n_seen)
            cur_top5 = top5_sum / max(1, n_seen)
            cur_lr = get_lr(optimizer)

            print(
                f"[Train][E{epoch:03d}][{it:05d}/{len(loader):05d}] "
                f"step={global_step} lr={cur_lr:.6g} "
                f"loss={cur_loss:.4f} top1={cur_top1:.2f} top5={cur_top5:.2f} "
                f"({dt:.1f}s)",
                flush=True,
            )

        # wandb step logging (separate interval to reduce I/O)
        if wandb_run is not None and wandb_log_interval > 0 and (global_step % wandb_log_interval == 0):
            cur_loss = loss_sum / max(1, n_seen)
            cur_top1 = top1_sum / max(1, n_seen)
            cur_top5 = top5_sum / max(1, n_seen)
            wandb_run.log(
                {
                    "train/iter_loss_avg": cur_loss,
                    "train/iter_top1_avg": cur_top1,
                    "train/iter_top5_avg": cur_top5,
                    "train/lr": get_lr(optimizer),
                    "train/epoch": epoch,
                },
                step=global_step,
            )

    return {
        "loss": loss_sum / max(1, n_seen),
        "top1": top1_sum / max(1, n_seen),
        "top5": top5_sum / max(1, n_seen),
    }, global_step


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, label_smoothing: float):
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing)).to(device)

    n_seen = 0
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        outputs = model(x)
        logits, _ = unpack_inception_outputs(outputs)
        loss = criterion(logits, y)

        bs = x.size(0)
        n_seen += bs
        loss_sum += float(loss.detach().item()) * bs

        acc = accuracy_topk(logits.detach(), y.detach(), topk=(1, 5))
        top1_sum += acc["top1"] * bs
        top5_sum += acc["top5"] * bs

    return {
        "loss": loss_sum / max(1, n_seen),
        "top1": top1_sum / max(1, n_seen),
        "top5": top5_sum / max(1, n_seen),
    }


# -------------------------
# Main
# -------------------------

def build_argparser():
    p = argparse.ArgumentParser("Train Inception-v3 scratch on gray3 ImageNet-1k (ImageFolder) + wandb")

    # ---- paths (no required; all defaults) ----
    p.add_argument("--train_dir", type=str, default="./imagenet1k_gray3_export/gray3/train")
    p.add_argument("--val_dir", type=str, default="./imagenet1k_gray3_export/gray3/val")
    p.add_argument("--output_dir", type=str, default="./inceptionv3_gray3_scratch")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--input_size", type=int, default=299)
    p.add_argument("--no_aug", action="store_true")

    p.add_argument("--mean", type=float, nargs=3, default=[0.5, 0.5, 0.5])
    p.add_argument("--std", type=float, nargs=3, default=[0.5, 0.5, 0.5])

    p.add_argument("--lr", type=float, default=0.4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=0.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)

    p.add_argument("--warmup_epochs", type=int, default=5)

    p.add_argument("--aux_logits", action="store_true")
    p.add_argument("--aux_weight", type=float, default=0.4)

    p.add_argument("--amp", action="store_true")

    # Separate intervals
    p.add_argument("--log_interval", type=int, default=50, help="Console print interval in iterations.")
    p.add_argument("--wandb_log_interval", type=int, default=200, help="wandb step log interval in global steps (0 disables).")

    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save_best_only", action="store_true")

    # ---- wandb args ----
    p.add_argument("--wandb", action="store_false", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb_project", type=str, default="imagenet-gray3")
    p.add_argument("--wandb_name", type=str, default="")
    p.add_argument("--wandb_entity", type=str, default="", help="Optional: wandb entity/team.")
    p.add_argument("--wandb_tags", type=str, nargs="*", default=[], help="Optional: wandb tags.")
    p.add_argument("--wandb_offline", action="store_true", help="Set WANDB_MODE=offline.")
    p.add_argument("--wandb_dir", type=str, default="", help="Optional wandb dir (defaults to output_dir).")

    return p


def main():
    args = build_argparser().parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    cudnn.benchmark = True

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "ckpts")

    # ---- wandb init (optional) ----
    wandb_run = None
    if args.wandb:
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"

        try:
            import wandb
            run_name = args.wandb_name if args.wandb_name else f"inceptionv3-gray3-scratch-bs{args.batch_size}-lr{args.lr}"
            wb_kwargs = {
                "project": args.wandb_project,
                "name": run_name,
                "config": vars(args),
                "dir": (args.wandb_dir if args.wandb_dir else str(out_dir)),
                "resume": "allow",
            }
            if args.wandb_entity:
                wb_kwargs["entity"] = args.wandb_entity

            wandb_run = wandb.init(**wb_kwargs)
            if args.wandb_tags:
                wandb_run.tags = tuple(args.wandb_tags)

        except Exception as e:
            print(f"[Warn] wandb init failed. Proceeding without wandb. ({e})", flush=True)
            wandb_run = None

    # Transforms
    mean = tuple(float(x) for x in args.mean)
    std = tuple(float(x) for x in args.std)
    normalize = T.Normalize(mean=mean, std=std)

    if args.no_aug:
        train_tf = T.Compose([
            T.Resize(args.input_size + 32, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(args.input_size),
            T.ToTensor(),
            normalize,
        ])
    else:
        train_tf = T.Compose([
            T.RandomResizedCrop(
                args.input_size,
                scale=(0.08, 1.0),
                ratio=(3/4, 4/3),
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalize,
        ])

    val_tf = T.Compose([
        T.Resize(args.input_size + 32, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(args.input_size),
        T.ToTensor(),
        normalize,
    ])

    # Datasets
    train_ds = ImageFolder(args.train_dir, transform=train_tf)
    val_ds   = ImageFolder(args.val_dir, transform=val_tf)
    num_classes = len(train_ds.classes)

    print(f"[Info] train={len(train_ds)} val={len(val_ds)} classes={num_classes}", flush=True)
    print(f"[Info] device={device} amp={args.amp and device.type=='cuda'} aux_logits={args.aux_logits}", flush=True)
    print(f"[Info] normalize mean={mean} std={std}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.workers > 0),
    )

    # Model (scratch)
    model = torchvision.models.inception_v3(
        weights=None,
        aux_logits=args.aux_logits,
        transform_input=False,
        init_weights=True,
        num_classes=num_classes,
    ).to(device)

    # Optim
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(args.lr),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        nesterov=True,
    )

    # AMP
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Resume
    start_epoch = 0
    best_top1 = -1.0
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optim"])
        if "scaler" in ckpt and ckpt["scaler"] is not None and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_top1 = float(ckpt.get("best_top1", -1.0))
        global_step = int(ckpt.get("global_step", 0))
        print(f"[Resume] start_epoch={start_epoch} best_top1={best_top1:.2f} global_step={global_step}", flush=True)

    # Save config
    save_json(out_dir / "config.json", vars(args))

    # wandb meta
    if wandb_run is not None:
        wandb_run.log(
            {
                "meta/num_classes": num_classes,
                "meta/train_size": len(train_ds),
                "meta/val_size": len(val_ds),
                "meta/params": count_parameters(model),
                "meta/aux_logits": bool(args.aux_logits),
                "meta/input_size": int(args.input_size),
            },
            step=global_step,
        )

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        lr = lr_at_epoch(float(args.lr), epoch, args.epochs, args.warmup_epochs)
        set_lr(optimizer, lr)

        tr, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            global_step=global_step,
            scaler=scaler if use_amp else None,
            use_amp=use_amp,
            aux_weight=float(args.aux_weight),
            label_smoothing=float(args.label_smoothing),
            max_grad_norm=float(args.max_grad_norm),
            log_interval=int(args.log_interval),
            wandb_log_interval=int(args.wandb_log_interval),
            wandb_run=wandb_run,
        )

        va = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            label_smoothing=float(args.label_smoothing),
        )

        is_best = va["top1"] > best_top1
        if is_best:
            best_top1 = va["top1"]

        print(
            f"[Epoch {epoch:03d}] step={global_step} lr={get_lr(optimizer):.6g} "
            f"train: loss={tr['loss']:.4f} top1={tr['top1']:.2f} top5={tr['top5']:.2f} | "
            f"val: loss={va['loss']:.4f} top1={va['top1']:.2f} top5={va['top5']:.2f} | "
            f"best_top1={best_top1:.2f}",
            flush=True,
        )

        # JSONL log
        log_line = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": get_lr(optimizer),
            "train": tr,
            "val": va,
            "best_top1": best_top1,
        }
        with (out_dir / "train_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_line, ensure_ascii=False) + "\n")

        # wandb epoch-level log (always once per epoch)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch/train_loss": tr["loss"],
                    "epoch/train_top1": tr["top1"],
                    "epoch/train_top5": tr["top5"],
                    "epoch/val_loss": va["loss"],
                    "epoch/val_top1": va["top1"],
                    "epoch/val_top5": va["top5"],
                    "epoch/best_top1": best_top1,
                    "train/lr": get_lr(optimizer),
                    "train/epoch": epoch,
                },
                step=global_step,
            )

        # Save checkpoints
        ckpt_obj = {
            "epoch": epoch,
            "global_step": global_step,
            "best_top1": best_top1,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "classes": train_ds.classes,
            "normalize_mean": mean,
            "normalize_std": std,
            "aux_logits": bool(args.aux_logits),
        }

        torch.save(ckpt_obj, out_dir / "ckpts" / "last.pt")

        if args.save_best_only:
            if is_best:
                torch.save(ckpt_obj, out_dir / "ckpts" / "best.pt")
        else:
            if args.save_every > 0 and ((epoch + 1) % args.save_every == 0 or is_best):
                name = "best.pt" if is_best else f"epoch{epoch:03d}.pt"
                torch.save(ckpt_obj, out_dir / "ckpts" / name)

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

    print(f"[Done] best_top1={best_top1:.2f}", flush=True)


if __name__ == "__main__":
    main()
