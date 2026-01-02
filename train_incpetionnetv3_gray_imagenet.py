#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train Inception-v3 on ImageNet-1k style ImageFolder (single GPU) with:
- AMP (torch.amp) => no deprecation warnings
- Custom Warmup+Cosine LR scheduler => no "scheduler.step before optimizer.step" warning
- Safe EMA
- RandAugment
- Gray->RGB(3ch) force option
- Weights & Biases logging (step-based, offline/resume supported)

Aligned defaults to your run log:
- device=cuda:6
- amp=True, tf32=True, ema=True, randaug=True, force_gray3=True
- optimizer=sgd, bs=256 => lr=0.1, wd=1e-4
- warmup_epochs=5
"""

import os
import json
import time
import math
import argparse
import random
import copy
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


# -------------------------
# Logging
# -------------------------

def setup_logger(log_path: Optional[Path]) -> logging.Logger:
    logger = logging.getLogger("train_incv3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# -------------------------
# Utils
# -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, obj: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def resolve_device(device_str: str) -> torch.device:
    s = (device_str or "cuda").strip().lower()
    if s == "cpu":
        return torch.device("cpu")
    if s == "cuda":
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if s.startswith("cuda:"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            idx = int(s.split(":")[1])
        except Exception:
            idx = 0
        n = torch.cuda.device_count()
        if idx < 0 or idx >= n:
            idx = 0
        return torch.device(f"cuda:{idx}")
    try:
        dev = torch.device(s)
        if dev.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return dev
    except Exception:
        return torch.device("cpu")

@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)) -> Dict[str, float]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # [N,maxk]
    pred = pred.t()  # [maxk,N]
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    out: Dict[str, float] = {}
    N = targets.size(0)
    for k in topk:
        c = correct[:k].reshape(-1).float().sum().item()
        out[f"top{k}"] = 100.0 * c / max(N, 1)
    return out

def unpack_inception_outputs(outputs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if hasattr(outputs, "logits"):
        return outputs.logits, getattr(outputs, "aux_logits", None)
    if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        return outputs[0], outputs[1]
    return outputs, None


# -------------------------
# Grayscale safety transform
# -------------------------

class ForceGray3:
    """Convert any PIL image -> grayscale (L) -> RGB (3ch)."""
    def __call__(self, img):
        img = img.convert("L")
        img = img.convert("RGB")
        return img


# -------------------------
# Safe EMA
# -------------------------

class ModelEma:
    """
    Safe EMA:
      - EMA only for floating-point parameters
      - buffers (BN running stats, num_batches_tracked, etc.) are copied
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[torch.device] = None):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = float(decay)
        self.device = device
        if self.device is not None:
            self.module.to(self.device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        ema_params = dict(self.module.named_parameters())
        model_params = dict(model.named_parameters())

        for name, p in model_params.items():
            if name not in ema_params:
                continue
            src = p.data
            dst = ema_params[name].data
            if self.device is not None:
                src = src.to(self.device)
            if torch.is_floating_point(dst) and torch.is_floating_point(src):
                dst.mul_(self.decay).add_(src, alpha=(1.0 - self.decay))
            else:
                dst.copy_(src)

        ema_bufs = dict(self.module.named_buffers())
        model_bufs = dict(model.named_buffers())
        for name, b in model_bufs.items():
            if name in ema_bufs:
                ema_bufs[name].data.copy_(b.data.to(device=ema_bufs[name].device))


# -------------------------
# Custom Warmup + Cosine LR Scheduler (no PyTorch warning)
# -------------------------

class WarmupCosineLR:
    """
    Per-optimizer-update LR schedule.
    - Sets LR for update #0 at initialization (warmup start factor).
    - After each optimizer update, call step() to set LR for the next update.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lrs: Optional[list] = None,
        total_updates: int = 1000,
        warmup_updates: int = 0,
        min_lr: float = 1e-6,
        warmup_start_factor: float = 0.01,
    ):
        self.optimizer = optimizer
        self.total_updates = int(max(1, total_updates))
        self.warmup_updates = int(max(0, min(warmup_updates, self.total_updates - 1)))
        self.min_lr = float(min_lr)
        self.warmup_start_factor = float(warmup_start_factor)

        if base_lrs is None:
            base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.base_lrs = [float(x) for x in base_lrs]

        self.update_idx = 0
        self._set_lr(self._lr_for_update(self.update_idx))

    def _lr_for_update(self, u: int) -> list:
        if self.warmup_updates > 0 and u < self.warmup_updates:
            t = u / float(self.warmup_updates)
            factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * t
            return [max(self.min_lr, lr * factor) for lr in self.base_lrs]

        t = (u - self.warmup_updates) / float(max(1, self.total_updates - self.warmup_updates))
        t = min(max(t, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return [self.min_lr + (lr - self.min_lr) * cos for lr in self.base_lrs]

    def _set_lr(self, lrs: list):
        for pg, lr in zip(self.optimizer.param_groups, lrs):
            pg["lr"] = float(lr)

    def step(self):
        self.update_idx += 1
        u = min(self.update_idx, self.total_updates - 1)
        self._set_lr(self._lr_for_update(u))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "total_updates": self.total_updates,
            "warmup_updates": self.warmup_updates,
            "min_lr": self.min_lr,
            "warmup_start_factor": self.warmup_start_factor,
            "base_lrs": self.base_lrs,
            "update_idx": self.update_idx,
        }

    def load_state_dict(self, sd: Dict[str, Any]):
        self.total_updates = int(sd["total_updates"])
        self.warmup_updates = int(sd["warmup_updates"])
        self.min_lr = float(sd["min_lr"])
        self.warmup_start_factor = float(sd.get("warmup_start_factor", 0.01))
        self.base_lrs = [float(x) for x in sd["base_lrs"]]
        self.update_idx = int(sd["update_idx"])

        u = min(self.update_idx, self.total_updates - 1)
        self._set_lr(self._lr_for_update(u))


# -------------------------
# Optimizer defaults
# -------------------------

def infer_lr_wd(optimizer_name: str, batch_size: int, lr: float, wd: float) -> Tuple[float, float]:
    opt = optimizer_name.lower()
    bs = int(batch_size)

    if wd < 0:
        if opt == "sgd":
            wd_eff = 1e-4
        elif opt == "rmsprop":
            wd_eff = 4e-5
        else:  # adamw
            wd_eff = 1e-3
    else:
        wd_eff = wd

    if lr < 0:
        if opt == "sgd":
            lr_eff = 0.1 * (bs / 256.0)
        elif opt == "rmsprop":
            lr_eff = 0.045 * (bs / 256.0)
        else:
            lr_eff = 5e-4
    else:
        lr_eff = lr

    return float(lr_eff), float(wd_eff)

def build_optimizer(args, model: nn.Module, lr_eff: float, wd_eff: float) -> torch.optim.Optimizer:
    opt = args.optimizer.lower()
    if opt == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr_eff,
            momentum=args.momentum,
            nesterov=args.nesterov,
            weight_decay=wd_eff,
        )
    if opt == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=lr_eff,
            alpha=args.rms_alpha,
            momentum=args.momentum,
            eps=args.eps,
            weight_decay=wd_eff,
        )
    if opt == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr_eff,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=wd_eff,
        )
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


# -------------------------
# W&B
# -------------------------

def init_wandb(args, logger: logging.Logger):
    """
    W&B is optional. Returns wandb_run or None.
    Supports offline and resuming with --wandb_id.
    """
    if not args.wandb:
        return None

    try:
        import wandb  # type: ignore

        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "offline"

        init_kwargs = dict(
            project=args.wandb_project,
            name=(args.wandb_name or None),
            config=vars(args),
        )

        # resume support
        if args.wandb_id:
            init_kwargs.update(dict(id=args.wandb_id, resume="allow"))

        run = wandb.init(**init_kwargs)

        # define default x-axis metric for nicer charts
        try:
            run.define_metric("global_update")
            run.define_metric("train/*", step_metric="global_update")
            run.define_metric("val/*", step_metric="global_update")
            run.define_metric("lr", step_metric="global_update")
        except Exception:
            pass

        logger.info(f"W&B enabled: project={args.wandb_project} name={run.name}")
        return run

    except Exception as e:
        logger.info(f"[Warning] W&B init failed: {e}")
        return None


# -------------------------
# Train / Eval
# -------------------------

def train_one_epoch(
    model: nn.Module,
    model_ema: Optional[ModelEma],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[WarmupCosineLR],
    device: torch.device,
    epoch: int,
    global_update: int,
    scaler: Optional[torch.amp.GradScaler],
    use_amp: bool,
    criterion: nn.Module,
    aux_weight: float,
    grad_accum: int,
    clip_grad_norm: float,
    ema_update_interval: int,
    log_interval_updates: int,
    wandb_log_interval: int,
    logger: logging.Logger,
    wandb_run=None,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    t0 = time.time()
    n_seen = 0
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0

    dev_type = "cuda" if device.type == "cuda" else "cpu"

    # track last-step values for W&B
    last_step_loss = None
    last_step_top1 = None
    last_step_top5 = None

    for it, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(dev_type, enabled=(use_amp and dev_type == "cuda")):
            outputs = model(x)
            logits, aux = unpack_inception_outputs(outputs)
            loss = criterion(logits, y)
            if aux is not None and aux_weight > 0:
                loss = loss + aux_weight * criterion(aux, y)

            loss_scaled = loss / max(1, grad_accum)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        # stats
        with torch.no_grad():
            bs = x.size(0)
            n_seen += bs
            loss_sum += float(loss.item()) * bs

            acc = accuracy_topk(logits.detach(), y.detach(), topk=(1, 5))
            top1_sum += acc["top1"] * bs
            top5_sum += acc["top5"] * bs

            last_step_loss = float(loss.item())
            last_step_top1 = float(acc["top1"])
            last_step_top5 = float(acc["top5"])

        do_update = (it % grad_accum == 0) or (it == len(loader))
        if do_update:
            # optimizer step first
            if scaler is not None and scaler.is_enabled():
                if clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            global_update += 1

            # scheduler sets lr for NEXT update
            if scheduler is not None:
                scheduler.step()

            # EMA update
            if model_ema is not None and ema_update_interval > 0 and (global_update % ema_update_interval == 0):
                model_ema.update(model)

            cur_lr = optimizer.param_groups[0]["lr"]

            # console/file log
            if log_interval_updates > 0 and (global_update % log_interval_updates == 0):
                dt = time.time() - t0
                imgs_per_s = n_seen / max(1e-9, dt)
                logger.info(
                    f"[Train][E{epoch:03d}][upd {global_update:07d}] "
                    f"lr={cur_lr:.6g} loss={loss_sum/max(1,n_seen):.4f} "
                    f"top1={top1_sum/max(1,n_seen):.2f} top5={top5_sum/max(1,n_seen):.2f} "
                    f"imgs/s={imgs_per_s:.1f}"
                )

            # W&B log (update-based)
            if wandb_run is not None and wandb_log_interval > 0 and (global_update % wandb_log_interval == 0):
                payload = {
                    "global_update": global_update,
                    "epoch": epoch + it / max(1, len(loader)),
                    "lr": cur_lr,
                    # instantaneous (last update)
                    "train/step_loss": last_step_loss,
                    "train/step_top1": last_step_top1,
                    "train/step_top5": last_step_top5,
                    # running avg
                    "train/avg_loss": loss_sum / max(1, n_seen),
                    "train/avg_top1": top1_sum / max(1, n_seen),
                    "train/avg_top5": top5_sum / max(1, n_seen),
                }

                # optional: GPU mem stats
                if device.type == "cuda":
                    try:
                        payload["sys/gpu_mem_alloc_gb"] = torch.cuda.memory_allocated(device) / (1024**3)
                        payload["sys/gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(device) / (1024**3)
                    except Exception:
                        pass

                wandb_run.log(payload, step=global_update)

    return {
        "loss": loss_sum / max(1, n_seen),
        "top1": top1_sum / max(1, n_seen),
        "top5": top5_sum / max(1, n_seen),
    }, global_update


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
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
        loss_sum += float(loss.item()) * bs

        acc = accuracy_topk(logits, y, topk=(1, 5))
        top1_sum += acc["top1"] * bs
        top5_sum += acc["top5"] * bs

    return {
        "loss": loss_sum / max(1, n_seen),
        "top1": top1_sum / max(1, n_seen),
        "top5": top5_sum / max(1, n_seen),
    }


# -------------------------
# Argparse
# -------------------------

def build_argparser():
    p = argparse.ArgumentParser("Inception-v3 gray ImageNet training (single GPU)")

    # data / io
    p.add_argument("--train_dir", type=str, default="./imagenet1k_export/gray3/train")
    p.add_argument("--val_dir", type=str, default="./imagenet1k_export/gray3/val")
    p.add_argument("--output_dir", type=str, default="./0102_inceptionv3_sgd_gray3")
    p.add_argument("--log_file", type=str, default="train.log")

    # runtime
    p.add_argument("--device", type=str, default="cuda:6")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)

    # training
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--input_size", type=int, default=299)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--clip_grad_norm", type=float, default=1.0)

    # toggles (match your log defaults)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--randaug", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force_gray3", action=argparse.BooleanOptionalAction, default=True)

    # inception options
    p.add_argument("--aux_logits", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--aux_weight", type=float, default=0.4)

    # norm (your computed stats)
    p.add_argument("--mean", type=float, nargs=3, default=[0.45798322587856827] * 3)
    p.add_argument("--std", type=float, nargs=3, default=[0.2623006911570552] * 3)

    # augment
    p.add_argument("--randaug_num_ops", type=int, default=2)
    p.add_argument("--randaug_magnitude", type=int, default=9)

    # optimizer
    p.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw", "rmsprop"])
    p.add_argument("--lr", type=float, default=0.05,
                   help="Base LR. If <0, uses default depending on optimizer & batch size.")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="Weight decay. If <0, uses optimizer-appropriate default.")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--rms_alpha", type=float, default=0.9)
    p.add_argument("--eps", type=float, default=1e-8)

    # scheduler
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_start_factor", type=float, default=0.01)

    # ema
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--ema_update_interval", type=int, default=1)

    # loss
    p.add_argument("--label_smoothing", type=float, default=0.1)

    # checkpoint
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--save_best_only", action=argparse.BooleanOptionalAction, default=False)

    # W&B
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb_project", type=str, default="0102-imagenet-gray3")
    p.add_argument("--wandb_name", type=str, default="")
    p.add_argument("--wandb_offline", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--wandb_id", type=str, default="", help="If set, resume W&B run with this id.")
    p.add_argument("--wandb_log_interval", type=int, default=50)

    return p


# -------------------------
# Main
# -------------------------

def main():
    args = build_argparser().parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed, deterministic=args.deterministic)

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir / "ckpts")
    logger = setup_logger(out_dir / args.log_file)

    # TF32 + cudnn
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        if not args.deterministic:
            cudnn.benchmark = True

    # W&B init
    wandb_run = init_wandb(args, logger)

    # transforms
    mean = tuple(args.mean)
    std = tuple(args.std)
    normalize = T.Normalize(mean=mean, std=std)

    train_tf_list = []
    if args.force_gray3:
        train_tf_list.append(ForceGray3())
    train_tf_list.extend([
        T.RandomResizedCrop(args.input_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.RandomHorizontalFlip(),
    ])
    if args.randaug:
        train_tf_list.append(T.RandAugment(num_ops=args.randaug_num_ops, magnitude=args.randaug_magnitude))
    train_tf_list.extend([T.ToTensor(), normalize])
    train_tf = T.Compose(train_tf_list)

    val_tf_list = []
    if args.force_gray3:
        val_tf_list.append(ForceGray3())
    val_tf_list.extend([
        T.Resize(int(args.input_size * 1.14), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(args.input_size),
        T.ToTensor(),
        normalize,
    ])
    val_tf = T.Compose(val_tf_list)

    # datasets
    train_ds = ImageFolder(args.train_dir, transform=train_tf)
    val_ds = ImageFolder(args.val_dir, transform=val_tf)
    num_classes = len(train_ds.classes)

    pin = (device.type == "cuda")
    dl_kwargs = dict(
        num_workers=args.workers,
        pin_memory=pin,
        persistent_workers=(args.workers > 0),
        drop_last=True,
    )
    if args.workers > 0:
        dl_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
        persistent_workers=(args.workers > 0),
        drop_last=False,
        prefetch_factor=4 if args.workers > 0 else None,
    )

    logger.info(f"Dataset: {len(train_ds)} train, {len(val_ds)} val | classes={num_classes}")
    logger.info(f"Device: {device} | amp={args.amp} | tf32={args.tf32} | ema={args.ema} | randaug={args.randaug} | force_gray3={args.force_gray3}")
    logger.info(f"Norm: mean={mean}, std={std}")

    # model
    model = torchvision.models.inception_v3(
        weights=None,
        aux_logits=bool(args.aux_logits),
        transform_input=False,
        num_classes=num_classes,
        init_weights=True,
    ).to(device)

    # optimizer
    lr_eff, wd_eff = infer_lr_wd(args.optimizer, args.batch_size, args.lr, args.weight_decay)
    optimizer = build_optimizer(args, model, lr_eff=lr_eff, wd_eff=wd_eff)

    # updates/epoch (drop_last=True) + schedule
    steps_per_epoch = len(train_loader)
    updates_per_epoch = (steps_per_epoch + max(1, args.grad_accum) - 1) // max(1, args.grad_accum)
    total_updates = int(args.epochs * updates_per_epoch)
    warmup_updates = int(args.warmup_epochs * updates_per_epoch)

    scheduler = WarmupCosineLR(
        optimizer=optimizer,
        base_lrs=[lr_eff for _ in optimizer.param_groups],
        total_updates=total_updates,
        warmup_updates=warmup_updates,
        min_lr=args.min_lr,
        warmup_start_factor=args.warmup_start_factor,
    )

    logger.info(f"Optimizer={args.optimizer} | lr={lr_eff:g} | wd={wd_eff:g} | updates/epoch={updates_per_epoch} total_updates={total_updates} warmup_updates={warmup_updates}")

    # loss
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    # AMP scaler (new API)
    use_amp = bool(args.amp) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    # EMA
    model_ema = ModelEma(model, decay=args.ema_decay, device=device) if args.ema else None

    # resume
    start_epoch = 0
    global_update = 0
    best_top1 = 0.0

    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=True)
            if model_ema is not None and ckpt.get("model_ema") is not None:
                model_ema.module.load_state_dict(ckpt["model_ema"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            if ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            if ckpt.get("scaler") is not None and scaler is not None:
                scaler.load_state_dict(ckpt["scaler"])

            start_epoch = int(ckpt.get("epoch", -1)) + 1
            global_update = int(ckpt.get("global_update", 0))
            best_top1 = float(ckpt.get("best_top1", 0.0))
            logger.info(f"Resumed from {ckpt_path} | start_epoch={start_epoch} global_update={global_update} best_top1={best_top1:.2f}")
        else:
            logger.info(f"[Warning] resume checkpoint not found: {ckpt_path}")

    # save config
    save_json(out_dir / "config.json", vars(args))

    # training loop
    for epoch in range(start_epoch, args.epochs):
        log_interval_updates = max(1, updates_per_epoch // 10)

        tr_stats, global_update = train_one_epoch(
            model=model,
            model_ema=model_ema,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            global_update=global_update,
            scaler=scaler,
            use_amp=use_amp,
            criterion=criterion,
            aux_weight=(args.aux_weight if args.aux_logits else 0.0),
            grad_accum=max(1, args.grad_accum),
            clip_grad_norm=float(args.clip_grad_norm),
            ema_update_interval=int(args.ema_update_interval),
            log_interval_updates=log_interval_updates,
            wandb_log_interval=int(args.wandb_log_interval),
            logger=logger,
            wandb_run=wandb_run,
        )

        eval_model = model_ema.module if model_ema is not None else model
        val_stats = evaluate(eval_model, val_loader, device, criterion)

        is_best = val_stats["top1"] > best_top1
        if is_best:
            best_top1 = val_stats["top1"]

        logger.info(
            f"[Epoch {epoch:03d}] "
            f"TrainLoss={tr_stats['loss']:.4f} "
            f"ValLoss={val_stats['loss']:.4f} ValTop1={val_stats['top1']:.2f}% ValTop5={val_stats['top5']:.2f}% "
            f"(BestTop1={best_top1:.2f}%)"
        )

        # W&B epoch log
        if wandb_run is not None:
            wandb_run.log(
                {
                    "global_update": global_update,
                    "val/loss": val_stats["loss"],
                    "val/top1": val_stats["top1"],
                    "val/top5": val_stats["top5"],
                    "val/best_top1": best_top1,
                    "epoch_int": epoch,
                },
                step=global_update,
            )

        ckpt = {
            "epoch": epoch,
            "global_update": global_update,
            "best_top1": best_top1,
            "model": model.state_dict(),
            "model_ema": (model_ema.module.state_dict() if model_ema is not None else None),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "classes": train_ds.classes,
            "args": vars(args),
        }

        if not args.save_best_only:
            torch.save(ckpt, out_dir / "ckpts" / "last.pt")
        if is_best:
            torch.save(ckpt, out_dir / "ckpts" / "best.pt")

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
