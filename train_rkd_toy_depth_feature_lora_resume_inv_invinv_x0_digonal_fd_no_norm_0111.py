#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RGB -> DEPTH (paired eval) with teacher inversion + student generation.

Train:
- Student is trained on DEPTH-only (unpaired) dataset with RKD/INV/INVINV/FD/SAME losses.
- Teacher is an RGB diffusion model (UNet2DModel), frozen.
- Both share the same noise z for generation (teacher provides relational reference).

Eval (paired RGB-DEPTH) [FIXED]:
- Invert teacher on x0_rgb to get zT (DDIMInverseScheduler)
- Run student from zT to generate depth (x0_pred)
- Convert student depth (3ch) -> 1ch (channel-mean in [0,1]) => "m"
- Load GT depth:
    priority:  depth.png.raw.npy + depth.png.mask.npy  (recommended)
    fallback:  depth.png.npy (normalized [0,1]) or depth.png (L)
- Apply affine-invariant alignment (least squares) per-image:
    a = m * s + t  (fit to GT on valid pixels)
- Compute AbsRel and δ accuracies on aligned depth, masked.

Important bugfixes vs your original:
- Teacher DDIM pass NEVER switches to train() (dropout etc. off)
- Inversion is done under torch.no_grad() so it doesn't build graphs and leak memory
- Eval metrics support masks + affine alignment, consistent with Marigold-style protocol.

Deps:
  pip install diffusers transformers torch torchvision peft wandb
"""

import os
import re
import math
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.utils as vutils
import torchvision.models as models
from PIL import Image

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DDIMInverseScheduler
from peft import LoraConfig, get_peft_model, PeftModel

from transformers import AutoModel, AutoImageProcessor, CLIPModel


# ------------------------- Feature Extraction Utils -------------------------

class FeatureEmbedder(nn.Module):
    """
    Feature space for RKD/INV/INVINV/FD.
    Modes: pixel / inception / clip / dinov3
    Input assumed: (N,3,H,W) in [-1,1]
    """
    def __init__(
        self,
        mode: str,
        device: torch.device,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        dino_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        hf_local_only: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.device = device
        self.hf_local_only = bool(hf_local_only)

        if mode == "pixel":
            self.net = None
            return

        if mode == "inception":
            print("[Embedder] Loading InceptionV3...", flush=True)
            weights = models.Inception_V3_Weights.DEFAULT
            self.net = models.inception_v3(weights=weights).to(device)
            self.net.fc = nn.Identity()
            self.net.dropout = nn.Identity()
            for p in self.net.parameters():
                p.requires_grad = False
            self.net.eval()
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            self.std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            self.target_size = (299, 299)
            return

        if mode == "clip":
            print(f"[Embedder] Loading CLIP ({clip_model_name})...", flush=True)
            self.net = CLIPModel.from_pretrained(
                clip_model_name, local_files_only=self.hf_local_only
            ).vision_model.to(device)
            for p in self.net.parameters():
                p.requires_grad = False
            self.net.eval()
            self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
            self.std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
            self.target_size = (224, 224)
            return

        if mode == "dinov3":
            print(f"[Embedder] Loading DINOv3 ({dino_model_name})...", flush=True)
            try:
                proc = AutoImageProcessor.from_pretrained(
                    dino_model_name, local_files_only=self.hf_local_only
                )
            except Exception as e:
                print(f"[Warn] AutoImageProcessor load failed ({e}). Using ImageNet mean/std and 224.", flush=True)
                proc = None

            self.net = AutoModel.from_pretrained(
                dino_model_name, local_files_only=self.hf_local_only
            ).to(device)

            for p in self.net.parameters():
                p.requires_grad = False
            self.net.eval()

            if proc is not None and hasattr(proc, "image_mean") and hasattr(proc, "image_std"):
                mean = proc.image_mean
                std = proc.image_std
            else:
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]

            self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
            self.std  = torch.tensor(std,  device=device).view(1, 3, 1, 1)

            target = 224
            if proc is not None and hasattr(proc, "size"):
                sz = proc.size
                if isinstance(sz, dict):
                    if "height" in sz and "width" in sz:
                        self.target_size = (int(sz["height"]), int(sz["width"]))
                    elif "shortest_edge" in sz:
                        target = int(sz["shortest_edge"])
                        self.target_size = (target, target)
                    else:
                        self.target_size = (target, target)
                elif isinstance(sz, int):
                    self.target_size = (int(sz), int(sz))
                else:
                    self.target_size = (target, target)
            else:
                self.target_size = (target, target)
            return

        raise ValueError(f"Unknown rkd_metric: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "pixel":
            return x.reshape(x.shape[0], -1)

        x01 = (x.clamp(-1, 1) + 1.0) * 0.5
        x_up = F.interpolate(x01, size=self.target_size, mode="bilinear", align_corners=False, antialias=True)
        x_norm = (x_up - self.mean) / self.std

        if self.mode == "inception":
            return self.net(x_norm)
        if self.mode == "clip":
            out = self.net(pixel_values=x_norm)
            return out.pooler_output
        if self.mode == "dinov3":
            out = self.net(pixel_values=x_norm)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                return out.pooler_output
            return out.last_hidden_state[:, 0, :]
        return x.reshape(x.shape[0], -1)


# ------------------------- Utils -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())

def set_seed(seed: int):
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

def resolve_device(device_str: str) -> torch.device:
    try:
        dev = torch.device(device_str)
    except Exception:
        print(f"[Warn] Invalid --device '{device_str}', fallback cpu", flush=True)
        return torch.device("cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev

def to_grid(images: torch.Tensor, nrow: int = 4) -> Image.Image:
    imgs = (images.clamp(-1, 1) + 1) / 2.0
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2)
    grid = (grid * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(grid)

def pdist_vec(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.pdist(x, p=2).clamp_min(eps)

def cdist_vec(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.cdist(x, y, p=2).reshape(-1).clamp_min(eps)

def mean_from_vectors(vectors: List[torch.Tensor], device: torch.device, eps: float = 1e-12) -> torch.Tensor:
    if len(vectors) == 0:
        return torch.tensor(1.0, device=device)
    s = torch.zeros((), device=device, dtype=torch.float64)
    c = torch.zeros((), device=device, dtype=torch.float64)
    for vv in vectors:
        s = s + vv.sum().to(torch.float64)
        c = c + torch.tensor(float(vv.numel()), device=device, dtype=torch.float64)
    m = (s / c.clamp_min(1.0)).to(torch.float32)
    return m.clamp_min(eps)


# ------------------------- Train-time FD (diag Gaussian) -------------------------

def frechet_distance_diag(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    X = X.float()
    Y = Y.float()
    mu_x = X.mean(dim=0)
    mu_y = Y.mean(dim=0)
    vx = X.var(dim=0, unbiased=False) + eps
    vy = Y.var(dim=0, unbiased=False) + eps
    mean_term = (mu_x - mu_y).pow(2).sum()
    trace_term = (vx + vy - 2.0 * torch.sqrt(vx * vy)).sum()
    return (mean_term + trace_term).clamp_min(0.0)


# ------------------------- Datasets -------------------------

class DepthOnlyFolderDataset(Dataset):
    """
    Train dataset: depth-only GT images under root recursively.
    - If use_npy=True and <file>.png.npy exists, load it (float32 in [0,1]) to avoid PNG quantization.
    - Otherwise load depth image (png/jpg), read as L, ToTensor -> [0,1].

    Output:
      x: (3,H,W) in [-1,1] (gray3)
    """
    def __init__(
        self,
        root: str,
        image_size: int = 32,
        center_crop: bool = False,
        horizontal_flip: bool = True,
        use_npy: bool = False,
    ):
        self.root = Path(root)
        exts = {".png", ".jpg", ".jpeg"}
        self.files = [p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
        if len(self.files) == 0:
            raise FileNotFoundError(f"No depth images under {self.root}")

        self.image_size = int(image_size)
        self.center_crop = bool(center_crop)
        self.horizontal_flip = bool(horizontal_flip)
        self.use_npy = bool(use_npy)

        tfms = [T.Resize(self.image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True)]
        if self.center_crop:
            tfms.append(T.CenterCrop(self.image_size))
        tfms.append(T.ToTensor())
        self.tf = T.Compose(tfms)

    def __len__(self):
        return len(self.files)

    def _maybe_flip(self, x: torch.Tensor) -> torch.Tensor:
        if self.horizontal_flip and (torch.rand(()) < 0.5):
            return torch.flip(x, dims=[2])  # W-dim for (C,H,W)
        return x

    def __getitem__(self, idx: int):
        p = self.files[idx]

        if self.use_npy:
            npy = Path(p.as_posix() + ".npy")  # expects .png.npy
            if npy.exists():
                arr = np.load(npy.as_posix()).astype(np.float32)  # (H,W) in [0,1]
                x = torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)
                if x.shape[-2:] != (self.image_size, self.image_size):
                    x = F.interpolate(
                        x.unsqueeze(0),
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                x = self._maybe_flip(x)
                x = x.repeat(3, 1, 1)
                return x * 2.0 - 1.0

        with Image.open(p) as img:
            img = img.convert("L")
            x01 = self.tf(img)  # (1,H,W) in [0,1]
        x01 = self._maybe_flip(x01)
        x01 = x01.repeat(3, 1, 1)
        return x01 * 2.0 - 1.0


class PairedRGBDepthDataset(Dataset):
    """
    Eval dataset: paired RGB and depth images by relative path.
    Assumes:
      rgb_root/.../xxx.png
      depth_root/.../xxx.png  (same rel path)

    GT loading priority:
      (1) depth.png.raw.npy + depth.png.mask.npy   (recommended for "proper" masking)
      (2) depth.png.npy                            (normalized float [0,1])
      (3) depth.png                                (L -> [0,1])

    Returns:
      x_rgb:     (3,H,W) in [-1,1]
      gt_depth:  (1,H,W) float32 (units arbitrary)
      gt_mask:   (1,H,W) float32 in {0,1}
      rel_path:  str
    """
    def __init__(
        self,
        rgb_root: str,
        depth_root: str,
        image_size: int = 32,
        use_depth_raw_npy: bool = True,
        use_depth_npy: bool = True,
    ):
        self.rgb_root = Path(rgb_root)
        self.depth_root = Path(depth_root)
        exts = {".png", ".jpg", ".jpeg"}

        rgb_files = sorted([p for p in self.rgb_root.rglob("*") if p.is_file() and p.suffix.lower() in exts])
        pairs = []
        for rp in rgb_files:
            rel = rp.relative_to(self.rgb_root)
            dp = self.depth_root / rel
            if dp.exists():
                pairs.append((rp, dp, str(rel).replace("\\", "/")))
        if len(pairs) == 0:
            raise FileNotFoundError(f"No paired files found. rgb_root={self.rgb_root}, depth_root={self.depth_root}")
        self.pairs = pairs

        self.image_size = int(image_size)
        self.use_depth_raw_npy = bool(use_depth_raw_npy)
        self.use_depth_npy = bool(use_depth_npy)

        self.tf_rgb = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
        ])
        self.tf_dep = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),  # (1,H,W) in [0,1] for L
        ])

    def __len__(self):
        return len(self.pairs)

    def _resize_tensor(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        # x: (1,H,W)
        if x.shape[-2:] == (self.image_size, self.image_size):
            return x
        x2 = F.interpolate(
            x.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode=mode,
            align_corners=False if mode in ["bilinear", "bicubic"] else None,
        ).squeeze(0)
        return x2

    def __getitem__(self, idx: int):
        rp, dp, rel = self.pairs[idx]

        with Image.open(rp) as im_rgb:
            im_rgb = im_rgb.convert("RGB")
            x_rgb01 = self.tf_rgb(im_rgb)       # (3,H,W) in [0,1]
            x_rgb = x_rgb01 * 2.0 - 1.0         # [-1,1]

        # (1) raw+mask npy
        if self.use_depth_raw_npy:
            rawp = Path(dp.as_posix() + ".raw.npy")
            maskp = Path(dp.as_posix() + ".mask.npy")
            if rawp.exists() and maskp.exists():
                raw = np.load(rawp.as_posix()).astype(np.float32)   # (H,W)
                msk = np.load(maskp.as_posix())                     # (H,W), bool/uint8
                raw_t = torch.from_numpy(raw).unsqueeze(0)          # (1,H,W)
                msk_t = torch.from_numpy(msk.astype(np.float32)).unsqueeze(0)  # (1,H,W)

                raw_t = self._resize_tensor(raw_t, mode="bilinear")
                msk_t = self._resize_tensor(msk_t, mode="nearest")
                msk_t = (msk_t > 0.5).to(torch.float32)

                # also drop invalid / non-positive for safety
                msk_t = msk_t * (raw_t > 0.0).to(torch.float32)
                return x_rgb, raw_t, msk_t, rel

        # (2) normalized depth.png.npy
        if self.use_depth_npy:
            npy = Path(dp.as_posix() + ".npy")  # .png.npy
            if npy.exists():
                arr = np.load(npy.as_posix()).astype(np.float32)  # (H,W) in [0,1]
                x_d = torch.from_numpy(arr).unsqueeze(0)          # (1,H,W)
                x_d = self._resize_tensor(x_d, mode="bilinear").clamp(0.0, 1.0)
                msk = (x_d > 0.0).to(torch.float32)
                return x_rgb, x_d, msk, rel

        # (3) PNG
        with Image.open(dp) as im_d:
            im_d = im_d.convert("L")
            x_d01 = self.tf_dep(im_d)  # (1,H,W) in [0,1]
        msk = (x_d01 > 0.0).to(torch.float32)
        return x_rgb, x_d01, msk, rel


# ------------------------- Schedulers -------------------------

def load_teacher_scheduler_or_fallback(teacher_dir: Path, train_timesteps: int, beta_schedule: str) -> DDPMScheduler:
    try:
        return DDPMScheduler.from_pretrained(teacher_dir.as_posix())
    except Exception:
        return DDPMScheduler(
            num_train_timesteps=train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type="epsilon",
        )

def make_ddim(ddpm: DDPMScheduler, prediction_type: str) -> DDIMScheduler:
    ddim = DDIMScheduler.from_config(ddpm.config)
    ddim.config.clip_sample = False
    ddim.config.prediction_type = prediction_type
    return ddim


# ------------------------- DDIM forward / inversion -------------------------

def predx0_seq_from_xt(
    model,
    ddim: DDIMScheduler,
    x_init: torch.Tensor,
    steps: int,
    eta: float,
    device: torch.device,
    with_grad: bool,
) -> List[torch.Tensor]:
    """
    Returns list of pred_original_sample per step.
    If with_grad=False: runs under torch.no_grad and model.eval().
    If with_grad=True: runs with grad and model.train().
    """
    local = DDIMScheduler.from_config(ddim.config)
    local.set_timesteps(int(steps), device=device)
    x = x_init.to(device)

    preds: List[torch.Tensor] = []

    if with_grad:
        # model.train()
        for t in local.timesteps:
            x_in = local.scale_model_input(x, t)
            eps = model(x_in, t).sample
            out = local.step(model_output=eps, timestep=t, sample=x, eta=float(eta))
            x = out.prev_sample
            preds.append(out.pred_original_sample)
    else:
        model.eval()
        with torch.no_grad():
            for t in local.timesteps:
                x_in = local.scale_model_input(x, t)
                eps = model(x_in, t).sample
                out = local.step(model_output=eps, timestep=t, sample=x, eta=float(eta))
                x = out.prev_sample
                preds.append(out.pred_original_sample)

    return preds

def teacher_predx0_seq(teacher, ddim_T, z, steps, eta, device) -> List[torch.Tensor]:
    # IMPORTANT: teacher must be eval/no_grad
    teacher.eval()
    return predx0_seq_from_xt(teacher, ddim_T, z, steps, eta, device, with_grad=True)

def student_predx0_seq_with_grad(student, ddim_S, z, steps, eta, device) -> List[torch.Tensor]:
    student.train()
    return predx0_seq_from_xt(student, ddim_S, z, steps, eta, device, with_grad=True)

def invert_x0_to_zT_ddim_inverse_epspred(
    model,
    ddim: DDIMScheduler,
    x0: torch.Tensor,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Deterministic DDIM inversion using DDIMInverseScheduler.
    Runs in no_grad for BOTH train/eval usage (prevents graph buildup).
    """
    inv = DDIMInverseScheduler.from_config(ddim.config)
    inv.set_timesteps(int(steps), device=device)

    # model.eval()
    xt = x0.to(device)
    for t in inv.timesteps:
        x_in = inv.scale_model_input(xt, t)
        eps = model(x_in, t).sample
        xt = inv.step(eps, t, xt).prev_sample
    return xt


# ------------------------- Depth metrics (Affine-invariant) -------------------------

def pred_depth_1ch_from_student_x0(x0_pred_3ch: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    student x0: (B,3,H,W) in [-1,1]
    -> (B,1,H,W) in (0,1] by channel-mean in [0,1]
    """
    x01 = (x0_pred_3ch.clamp(-1, 1) + 1.0) * 0.5
    d1 = x01.mean(dim=1, keepdim=True)
    return d1.clamp_min(eps)

def solve_scale_shift_lstsq(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12):
    """
    Per-image least squares fit:
      minimize || (s * pred + t) - gt ||^2 over mask==1 pixels

    pred, gt, mask: (B,1,H,W)
    Returns: s, t as (B,1,1,1)
    """
    B = pred.shape[0]
    p = pred.view(B, -1).to(torch.float64)
    d = gt.view(B, -1).to(torch.float64)
    m = (mask.view(B, -1) > 0.5)

    s_out = torch.zeros((B, 1, 1, 1), device=pred.device, dtype=torch.float64)
    t_out = torch.zeros((B, 1, 1, 1), device=pred.device, dtype=torch.float64)

    for i in range(B):
        mi = m[i]
        n = int(mi.sum().item())
        if n < 2:
            # fallback: no valid pixels
            s_out[i] = 0.0
            t_out[i] = 0.0
            continue

        pv = p[i, mi]
        dv = d[i, mi]

        sum_p2 = (pv * pv).sum()
        sum_p  = pv.sum()
        sum_d  = dv.sum()
        sum_pd = (pv * dv).sum()
        n_t = torch.tensor(float(n), device=pred.device, dtype=torch.float64)

        denom = (sum_p2 * n_t - sum_p * sum_p)
        if torch.abs(denom) < eps:
            # pred is (almost) constant -> best is shifting to mean(gt)
            s = torch.tensor(0.0, device=pred.device, dtype=torch.float64)
            t = (sum_d / n_t)
        else:
            s = (n_t * sum_pd - sum_p * sum_d) / denom
            t = (sum_p2 * sum_d - sum_p * sum_pd) / denom

        s_out[i, 0, 0, 0] = s
        t_out[i, 0, 0, 0] = t

    return s_out.to(torch.float32), t_out.to(torch.float32)

@torch.no_grad()
def compute_absrel_delta(
    pred_aligned: torch.Tensor,  # (B,1,H,W)
    gt: torch.Tensor,            # (B,1,H,W)
    mask: torch.Tensor,          # (B,1,H,W) {0,1}
    delta: float = 1.25,
    depth_eps: float = 1e-6,
    reduce: str = "image",       # "image" (default) or "pixel"
):
    """
    Returns dict with:
      absrel, d1, d2, d3, valid_pixels, valid_images
    reduce:
      - "image": mean over pixels per-image, then mean over images (standard)
      - "pixel": sum over all valid pixels / total valid pixels
    """
    pred = pred_aligned.clamp_min(depth_eps)
    gt = gt.clamp_min(depth_eps)
    m = (mask > 0.5)

    B = pred.shape[0]

    if reduce == "pixel":
        valid = m
        vp = valid.sum().item()
        if vp <= 0:
            return None
        absrel = ((pred - gt).abs() / gt)[valid].mean().item()

        ratio = torch.maximum(pred / gt, gt / pred)
        d1 = (ratio < (delta))[valid].float().mean().item()
        d2 = (ratio < (delta ** 2))[valid].float().mean().item()
        d3 = (ratio < (delta ** 3))[valid].float().mean().item()
        return {
            "absrel": absrel, "d1": d1, "d2": d2, "d3": d3,
            "valid_pixels": float(vp), "valid_images": float(B),
        }

    # image-reduce
    absrel_list, d1_list, d2_list, d3_list = [], [], [], []
    valid_imgs = 0
    valid_pixels = 0.0

    ratio = torch.maximum(pred / gt, gt / pred)

    for i in range(B):
        vi = m[i]
        vp = vi.sum().item()
        if vp <= 0:
            continue
        valid_imgs += 1
        valid_pixels += float(vp)

        absrel_i = ((pred[i] - gt[i]).abs() / gt[i])[vi].mean().item()
        d1_i = (ratio[i] < (delta))[vi].float().mean().item()
        d2_i = (ratio[i] < (delta ** 2))[vi].float().mean().item()
        d3_i = (ratio[i] < (delta ** 3))[vi].float().mean().item()

        absrel_list.append(absrel_i)
        d1_list.append(d1_i)
        d2_list.append(d2_i)
        d3_list.append(d3_i)

    if valid_imgs == 0:
        return None

    return {
        "absrel": float(np.mean(absrel_list)),
        "d1": float(np.mean(d1_list)),
        "d2": float(np.mean(d2_list)),
        "d3": float(np.mean(d3_list)),
        "valid_pixels": float(valid_pixels),
        "valid_images": float(valid_imgs),
    }

def depth_to_vis_3ch_m11(depth_1ch: torch.Tensor, mask_1ch: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    depth_1ch: (B,1,H,W) arbitrary positive
    mask_1ch:  (B,1,H,W) {0,1}
    returns:   (B,3,H,W) in [-1,1] for visualization (per-image minmax over valid).
    """
    B = depth_1ch.shape[0]
    d = depth_1ch.clone()
    m = (mask_1ch > 0.5)

    out = torch.zeros((B, 3, d.shape[-2], d.shape[-1]), device=d.device, dtype=d.dtype)

    for i in range(B):
        vi = m[i, 0]
        if vi.sum() <= 0:
            di01 = torch.zeros_like(d[i, 0])
        else:
            dv = d[i, 0][vi]
            mn = dv.min()
            mx = dv.max()
            if (mx - mn) < eps:
                di01 = torch.zeros_like(d[i, 0])
            else:
                di01 = (d[i, 0] - mn) / (mx - mn)
                di01 = di01.clamp(0.0, 1.0)
        vis = di01.unsqueeze(0).repeat(3, 1, 1) * 2.0 - 1.0
        out[i] = vis
    return out.clamp(-1, 1)


# ------------------------- Losses (train) -------------------------

def compute_losses(
    preds_T: List[torch.Tensor],   # teacher RGB x0 preds
    preds_S: List[torch.Tensor],   # student DEPTH x0 preds
    x0_real: torch.Tensor,         # real DEPTH GT (3ch gray3) in [-1,1]
    x0_inv_T: torch.Tensor,        # teacher output when fed zT_real
    embedder: FeatureEmbedder,
    args,
):
    eps = 1e-12
    device = x0_real.device

    T_last_img = preds_T[-1]
    S_last_img = preds_S[-1]

    def get_feats(img):
        return embedder(img)

    # RKD
    rkd_s_list, rkd_t_list = [], []
    if args.w_rkd != 0.0:
        stride = max(1, int(args.rkd_stride))
        if args.rkd_teacher_ref == "last":
            feats_T_last = get_feats(T_last_img)
            T_ref_pdist = pdist_vec(feats_T_last, eps=eps)
            for k in range(0, len(preds_S), stride):
                feats_S = get_feats(preds_S[k])
                rkd_s_list.append(pdist_vec(feats_S, eps=eps))
                rkd_t_list.append(T_ref_pdist)
        else:
            for k in range(0, len(preds_S), stride):
                feats_S = get_feats(preds_S[k])
                feats_T = get_feats(preds_T[k])
                rkd_s_list.append(pdist_vec(feats_S, eps=eps))
                rkd_t_list.append(pdist_vec(feats_T, eps=eps))

    # INV / INVINV
    inv_s = inv_t = None
    if args.w_inv != 0.0:
        f_S_last = get_feats(S_last_img)
        f_real   = get_feats(x0_real)
        f_T_last = get_feats(T_last_img)
        f_inv    = get_feats(x0_inv_T)
        inv_s = cdist_vec(f_S_last, f_real, eps=eps)
        inv_t = cdist_vec(f_T_last, f_inv, eps=eps)

    invinv_s = invinv_t = None
    if args.w_invinv != 0.0:
        f_real2 = get_feats(x0_real) if (args.w_inv == 0.0) else f_real
        f_inv2  = get_feats(x0_inv_T) if (args.w_inv == 0.0) else f_inv
        invinv_s = pdist_vec(f_real2, eps=eps)
        invinv_t = pdist_vec(f_inv2, eps=eps)

    # mean normalization
    student_parts: List[torch.Tensor] = []
    teacher_parts: List[torch.Tensor] = []
    if args.w_rkd != 0.0 and len(rkd_s_list) > 0:
        student_parts.append(torch.cat(rkd_s_list, dim=0))
        teacher_parts.append(torch.cat([d for d in rkd_t_list], dim=0))
    if args.w_inv != 0.0 and inv_s is not None:
        student_parts.append(inv_s); teacher_parts.append(inv_t)
    if args.w_invinv != 0.0 and invinv_s is not None:
        student_parts.append(invinv_s); teacher_parts.append(invinv_t)

    if len(student_parts) > 0:
        student_mean = mean_from_vectors(student_parts, device=device, eps=eps)
        teacher_mean = mean_from_vectors(teacher_parts, device=device, eps=eps)
    else:
        student_mean = torch.tensor(1.0, device=device)
        teacher_mean = torch.tensor(1.0, device=device)

    if args.w_rkd != 0.0:
        rkd_s_list = [d / student_mean for d in rkd_s_list]
        rkd_t_list = [d / teacher_mean for d in rkd_t_list]
    if args.w_inv != 0.0 and inv_s is not None:
        inv_s = inv_s / student_mean
        inv_t = inv_t / teacher_mean
    if args.w_invinv != 0.0 and invinv_s is not None:
        invinv_s = invinv_s / student_mean
        invinv_t = invinv_t / teacher_mean

    # scalar losses
    loss_rkd = torch.tensor(0.0, device=device)
    if args.w_rkd != 0.0 and len(rkd_s_list) > 0:
        acc = 0.0
        for ds, dt in zip(rkd_s_list, rkd_t_list):
            acc = acc + F.mse_loss(ds, dt, reduction="mean")
        loss_rkd = acc / max(1, len(rkd_s_list))

    loss_inv = torch.tensor(0.0, device=device)
    if args.w_inv != 0.0 and inv_s is not None:
        loss_inv = F.mse_loss(inv_s, inv_t, reduction="mean")

    loss_invinv = torch.tensor(0.0, device=device)
    if args.w_invinv != 0.0 and invinv_s is not None:
        loss_invinv = F.mse_loss(invinv_s, invinv_t, reduction="mean")

    # FD (diag)
    loss_fd = torch.tensor(0.0, device=device)
    fd_s = fd_t = torch.tensor(0.0, device=device)
    if args.w_fd != 0.0:
        S_f = embedder(S_last_img).float()
        R_f = embedder(x0_real).float()
        T_f = embedder(T_last_img).float()
        I_f = embedder(x0_inv_T).float()
        fd_s = frechet_distance_diag(S_f, R_f, eps=args.fd_eps)
        fd_t = frechet_distance_diag(T_f, I_f, eps=args.fd_eps)
        loss_fd = fd_s + fd_t

    # SAME
    loss_same = torch.tensor(0.0, device=device)
    if args.w_same != 0.0:
        xs = torch.stack(preds_S, dim=0)  # [K,B,3,H,W]
        if args.same_mode == "mean":
            mu = xs.mean(dim=0, keepdim=True)
            loss_same = F.mse_loss(xs, mu.expand_as(xs), reduction="mean")
        else:
            ref = xs[-1].detach()
            loss_same = F.mse_loss(xs[:-1], ref.unsqueeze(0).expand_as(xs[:-1]), reduction="mean")

    total = (
        args.w_rkd * loss_rkd +
        args.w_inv * loss_inv +
        args.w_invinv * loss_invinv +
        args.w_fd * loss_fd +
        args.w_same * loss_same
    )

    stats = {
        "loss_rkd": loss_rkd.detach(),
        "loss_inv": loss_inv.detach(),
        "loss_invinv": loss_invinv.detach(),
        "loss_fd": loss_fd.detach(),
        "fd_s": fd_s.detach(),
        "fd_t": fd_t.detach(),
        "loss_same": loss_same.detach(),
        "student_mean_dist": student_mean.detach(),
        "teacher_mean_dist": teacher_mean.detach(),
    }
    return total, stats

def build_loss_logs(total_loss: torch.Tensor, stats: dict, args) -> dict:
    def w_and_raw(raw_val: float, w: float):
        w = float(w)
        weighted = raw_val * w
        raw = (weighted / w) if (w != 0.0) else raw_val
        return weighted, raw

    rkd_raw    = float(stats["loss_rkd"].item())
    inv_raw    = float(stats["loss_inv"].item())
    invinv_raw = float(stats["loss_invinv"].item())
    fd_raw     = float(stats["loss_fd"].item())
    fd_s_raw   = float(stats["fd_s"].item())
    fd_t_raw   = float(stats["fd_t"].item())
    same_raw   = float(stats["loss_same"].item())
    total      = float(total_loss.detach().item())

    rkd_w, rkd_raw2       = w_and_raw(rkd_raw, args.w_rkd)
    inv_w, inv_raw2       = w_and_raw(inv_raw, args.w_inv)
    invinv_w, invinv_raw2 = w_and_raw(invinv_raw, args.w_invinv)
    fd_w, fd_raw2         = w_and_raw(fd_raw, args.w_fd)
    fd_s_w, fd_s_raw2     = w_and_raw(fd_s_raw, args.w_fd)
    fd_t_w, fd_t_raw2     = w_and_raw(fd_t_raw, args.w_fd)
    same_w, same_raw2     = w_and_raw(same_raw, args.w_same)

    return {
        "loss/total": total,
        "loss/rkd": rkd_w,
        "loss/inv": inv_w,
        "loss/invinv": invinv_w,
        "loss/fd": fd_w,
        "loss/fd_s": fd_s_w,
        "loss/fd_t": fd_t_w,
        "loss/same": same_w,
        "loss_raw/rkd": rkd_raw2,
        "loss_raw/inv": inv_raw2,
        "loss_raw/invinv": invinv_raw2,
        "loss_raw/fd": fd_raw2,
        "loss_raw/same": same_raw2,
    }


# ------------------------- Eval: paired RGB->Depth (Affine-invariant) -------------------------

@torch.no_grad()
def eval_rgb2depth_affine_absrel_delta(
    teacher: UNet2DModel,
    student: UNet2DModel,
    ddim_T: DDIMScheduler,
    ddim_S: DDIMScheduler,
    args,
    device: torch.device,
    global_step: int,
    out_dir: Path,
    summary_path: Path,
    use_amp: bool,
    amp_dtype: Optional[torch.dtype],
    wandb_run,
    wandb,
    eval_loader: DataLoader,
):
    teacher.eval()
    student.eval()

    # Autocast for student generation only (NOT inversion) if enabled
    gen_autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
        if (use_amp and device.type == "cuda" and args.eval_gen_amp)
        else nullcontext()
    )

    # accumulators
    # We'll accumulate per-batch image-reduced metrics then average (robust with masks)
    absrel_list, d1_list, d2_list, d3_list = [], [], [], []
    absrel_list_pix, d1_list_pix, d2_list_pix, d3_list_pix = [], [], [], []  # optional pixel-weighted log

    vis_done = False

    for bi, (x_rgb, gt_depth, gt_mask, rels) in enumerate(eval_loader, start=1):
        if args.eval_num_batches > 0 and bi > args.eval_num_batches:
            break

        x_rgb = x_rgb.to(device, non_blocking=True)          # (B,3,H,W) in [-1,1]
        gt_depth = gt_depth.to(device, non_blocking=True)    # (B,1,H,W) >0
        gt_mask = gt_mask.to(device, non_blocking=True)      # (B,1,H,W) {0,1}

        # extra safety: ignore non-positive GT
        gt_mask = gt_mask * (gt_depth > args.depth_eps).to(torch.float32)

        # 1) teacher inversion on RGB -> zT (NO autocast, NO grad)
        zT = invert_x0_to_zT_ddim_inverse_epspred(
            model=teacher,
            ddim=ddim_T,
            x0=x_rgb,
            steps=args.eval_steps,
            device=device,
        )

        # 2) student forward from zT -> depth pred (autocast optional)
        with gen_autocast_ctx:
            preds_S = predx0_seq_from_xt(
                model=student,
                ddim=ddim_S,
                x_init=zT,
                steps=args.eval_steps,
                eta=args.eval_eta,
                device=device,
                with_grad=False,
            )
            x0_depth_pred_3ch = preds_S[-1]  # (B,3,H,W) in [-1,1]

        pred_m = pred_depth_1ch_from_student_x0(x0_depth_pred_3ch, eps=args.depth_eps)  # (B,1,H,W) in (0,1]

        if args.eval_affine_align:
            s, t = solve_scale_shift_lstsq(pred_m, gt_depth, gt_mask, eps=1e-12)

            print("s stats:", s.min().item(), s.mean().item(), s.max().item())
            print("t stats:", t.min().item(), t.mean().item(), t.max().item())
            print("gt stats:", gt_depth[gt_mask>0.5].min().item(),
                            gt_depth[gt_mask>0.5].mean().item(),
                            gt_depth[gt_mask>0.5].max().item())
            print("pred_m stats:", pred_m[gt_mask>0.5].min().item(),
                                pred_m[gt_mask>0.5].mean().item(),
                                pred_m[gt_mask>0.5].max().item())


            pred_a = (pred_m * s + t).clamp_min(args.depth_eps)
        else:
            pred_a = pred_m

        # compute metrics (image mean + pixel mean)
        met_img = compute_absrel_delta(
            pred_aligned=pred_a, gt=gt_depth, mask=gt_mask,
            delta=args.delta, depth_eps=args.depth_eps, reduce="image"
        )
        met_pix = compute_absrel_delta(
            pred_aligned=pred_a, gt=gt_depth, mask=gt_mask,
            delta=args.delta, depth_eps=args.depth_eps, reduce="pixel"
        )

        if met_img is not None:
            absrel_list.append(met_img["absrel"])
            d1_list.append(met_img["d1"])
            d2_list.append(met_img["d2"])
            d3_list.append(met_img["d3"])

        if met_pix is not None:
            absrel_list_pix.append(met_pix["absrel"])
            d1_list_pix.append(met_pix["d1"])
            d2_list_pix.append(met_pix["d2"])
            d3_list_pix.append(met_pix["d3"])

        # visualization: RGB / pred(m) / pred(aligned) / GT (first batch only)
        if (not vis_done) and args.eval_save_vis:
            vis_done = True
            ensure_dir(out_dir / "eval_vis")

            predm_vis = depth_to_vis_3ch_m11(pred_m, gt_mask, eps=args.depth_eps)
            preda_vis = depth_to_vis_3ch_m11(pred_a, gt_mask, eps=args.depth_eps)
            gt_vis    = depth_to_vis_3ch_m11(gt_depth.clamp_min(args.depth_eps), gt_mask, eps=args.depth_eps)

            cat = torch.cat([x_rgb, predm_vis, preda_vis, gt_vis], dim=0)
            n = x_rgb.shape[0]
            grid = to_grid(cat, nrow=n)
            vis_path = out_dir / "eval_vis" / f"rgb_predm_preda_gt_step{global_step:08d}.png"
            grid.save(vis_path)

            with summary_path.open("a", encoding="utf-8") as f:
                f.write(f"[EVAL-VIS] step={global_step:08d} saved={vis_path.as_posix()}\n")

            if wandb_run is not None and wandb is not None:
                try:
                    wandb.log({"eval/rgb_predm_preda_gt": wandb.Image(grid)}, step=global_step)
                except Exception:
                    pass

    if len(absrel_list) == 0:
        return

    # dataset averages (image-mean)
    absrel_m = float(np.mean(absrel_list))
    d1_m = float(np.mean(d1_list))
    d2_m = float(np.mean(d2_list))
    d3_m = float(np.mean(d3_list))

    # dataset averages (pixel-weighted) for reference
    absrel_p = float(np.mean(absrel_list_pix)) if len(absrel_list_pix) else float("nan")
    d1_p = float(np.mean(d1_list_pix)) if len(d1_list_pix) else float("nan")
    d2_p = float(np.mean(d2_list_pix)) if len(d2_list_pix) else float("nan")
    d3_p = float(np.mean(d3_list_pix)) if len(d3_list_pix) else float("nan")

    line = (
        f"[EVAL] step={global_step:08d} "
        f"(image-mean) AbsRel={absrel_m:.6f} d1={d1_m:.6f} d2={d2_m:.6f} d3={d3_m:.6f} | "
        f"(pixel-mean) AbsRel={absrel_p:.6f} d1={d1_p:.6f} d2={d2_p:.6f} d3={d3_p:.6f} "
        f"(affine_align={int(args.eval_affine_align)}, steps={args.eval_steps}, eta={args.eval_eta}, "
        f"batches_limit={args.eval_num_batches})"
    )
    print(line, flush=True)
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

    if wandb_run is not None and wandb is not None:
        try:
            wandb.log({
                "eval/absrel": absrel_m,
                "eval/delta1": d1_m,
                "eval/delta2": d2_m,
                "eval/delta3": d3_m,
                "eval_pix/absrel": absrel_p,
                "eval_pix/delta1": d1_p,
                "eval_pix/delta2": d2_p,
                "eval_pix/delta3": d3_p,
                "eval/eval_steps": int(args.eval_steps),
                "eval/eval_eta": float(args.eval_eta),
                "eval/eval_num_batches": int(args.eval_num_batches),
                "eval/use_depth_raw_npy": int(args.eval_use_depth_raw_npy),
                "eval/use_depth_npy": int(args.eval_use_depth_npy),
                "eval/affine_align": int(args.eval_affine_align),
                "eval/gen_amp": int(args.eval_gen_amp),
            }, step=global_step)
        except Exception:
            pass


# ------------------------- Sampling (optional) -------------------------

@torch.no_grad()
def sample_student_depth_grid(
    student,
    ddim_S: DDIMScheduler,
    args,
    device: torch.device,
    global_step: int,
    out_dir: Path,
    use_amp: bool,
    amp_dtype: Optional[torch.dtype],
    wandb_run,
    wandb,
):
    if args.sample_n <= 0:
        return
    ensure_dir(out_dir / "samples")

    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + int(global_step))

    local = DDIMScheduler.from_config(ddim_S.config)
    local.set_timesteps(int(args.sample_steps), device=device)

    x = torch.randn((args.sample_n, 3, args.image_size, args.image_size), device=device, generator=gen)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )

    with autocast_ctx:
        for t in local.timesteps:
            x_in = local.scale_model_input(x, t)
            eps = student(x_in, t).sample
            x = local.step(model_output=eps, timestep=t, sample=x, eta=float(args.sample_eta), generator=gen).prev_sample

    depth1 = pred_depth_1ch_from_student_x0(x, eps=args.depth_eps)  # (N,1,H,W) in (0,1]
    mask = torch.ones_like(depth1)
    depth_vis = depth_to_vis_3ch_m11(depth1, mask)

    nrow = int(math.isqrt(args.sample_n))
    if nrow * nrow != args.sample_n:
        nrow = min(args.sample_n, 8)

    grid = to_grid(depth_vis, nrow=nrow)
    grid_path = out_dir / "samples" / f"depth_samples_step{global_step:08d}.png"
    grid.save(grid_path)

    if wandb_run is not None and wandb is not None:
        try:
            wandb.log({"samples/depth": wandb.Image(grid)}, step=global_step)
        except Exception:
            pass


# ------------------------- Sampling (optional) -------------------------

@torch.no_grad()
def sample_rgb_gt_est_grid(
    teacher,
    student,
    ddim_T: DDIMScheduler,
    ddim_S: DDIMScheduler,
    args,
    device: torch.device,
    global_step: int,
    out_dir: Path,
    use_amp: bool,
    amp_dtype: Optional[torch.dtype],
    wandb_run,
    wandb,
    eval_loader: DataLoader,
):
    """
    Make a grid: [RGB row] / [GT depth row] / [EST depth row]
    using one batch from eval_loader.
    """
    if eval_loader is None:
        return

    ensure_dir(out_dir / "samples")

    # take one batch (deterministic: always first batch)
    batch = next(iter(eval_loader))
    x_rgb, gt_depth, gt_mask, rels = batch

    x_rgb = x_rgb.to(device, non_blocking=True)         # (B,3,H,W) in [-1,1]
    gt_depth = gt_depth.to(device, non_blocking=True)   # (B,1,H,W) >0 or [0,1]
    gt_mask = gt_mask.to(device, non_blocking=True)     # (B,1,H,W) {0,1}

    # safety
    gt_mask = gt_mask * (gt_depth > args.depth_eps).to(torch.float32)

    # optionally limit how many columns to show (use sample_n)
    n = min(int(args.sample_n), x_rgb.shape[0])

    # preserve mode
    was_training = student.training
    teacher.eval()
    student.eval()

    # inversion (NO autocast)
    zT = invert_x0_to_zT_ddim_inverse_epspred(
        model=teacher,
        ddim=ddim_T,
        x0=x_rgb[:n],
        steps=int(args.sample_steps),     # use sample_steps for “sampling”
        device=device,
    )

    # student generation (autocast optional)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )

    with autocast_ctx:
        preds_S = predx0_seq_from_xt(
            model=student,
            ddim=ddim_S,
            x_init=zT,
            steps=int(args.sample_steps),
            eta=float(args.sample_eta),
            device=device,
            with_grad=False,
        )
        x0_depth_pred_3ch = preds_S[-1]  # (n,3,H,W) in [-1,1]

    pred_m = pred_depth_1ch_from_student_x0(x0_depth_pred_3ch, eps=args.depth_eps)  # (n,1,H,W)

    # (optional) affine align using same flag as eval
    if args.eval_affine_align:
        s, t = solve_scale_shift_lstsq(pred_m, gt_depth[:n], gt_mask[:n], eps=1e-12)
        pred_a = (pred_m * s + t).clamp_min(args.depth_eps)
    else:
        pred_a = pred_m

    # visualize GT/EST with per-image minmax over valid pixels
    gt_vis  = depth_to_vis_3ch_m11(gt_depth[:n].clamp_min(args.depth_eps), gt_mask[:n], eps=args.depth_eps)
    est_vis = depth_to_vis_3ch_m11(pred_a, gt_mask[:n], eps=args.depth_eps)

    # grid layout: 3 rows (RGB / GT / EST), n columns
    cat = torch.cat([x_rgb[:n], gt_vis, est_vis], dim=0)
    grid = to_grid(cat, nrow=n)

    grid_path = out_dir / "samples" / f"rgb_gt_est_step{global_step:08d}.png"
    grid.save(grid_path)

    if wandb_run is not None and wandb is not None:
        try:
            wandb.log({"samples/rgb_gt_est": wandb.Image(grid)}, step=global_step)
        except Exception:
            pass

    # restore mode
    if was_training:
        student.train()



# ------------------------- Train -------------------------

def _parse_step_from_path(p: str) -> int:
    # supports ckpt_step000123, ckpt_step123, ...step123...
    m = re.search(r"step(\d+)", Path(p).name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0

def train(args):
    torch.backends.cudnn.benchmark = True
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index)

    # embedder
    print(f"[Info] Embedder mode: {args.rkd_metric}", flush=True)
    embedder = FeatureEmbedder(
        mode=args.rkd_metric,
        device=device,
        clip_model_name=args.clip_model_name,
        dino_model_name=args.dino_model_name,
        hf_local_only=args.hf_local_only,
    )

    # wandb
    wandb_run = None
    wandb = None
    try:
        import wandb as _wandb
        wandb = _wandb
        wandb_run = wandb.init(
            project=args.project,
            name=args.run_name,
            config=vars(args),
            resume="allow",
            dir=args.output_dir,
        )
    except Exception as e:
        print(f"[Warn] wandb init failed. ({e})", flush=True)

    # AMP
    use_amp = (device.type == "cuda") and (args.mixed_precision != "no")
    amp_dtype = None
    scaler = None
    if use_amp:
        if args.mixed_precision == "fp16":
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        elif args.mixed_precision == "bf16":
            amp_dtype = torch.bfloat16
            scaler = None
        else:
            use_amp = False

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "samples")
    ensure_dir(out_dir / "ckpts")
    summary_path = out_dir / "summary.txt"

    # train data: depth-only
    train_ds = DepthOnlyFolderDataset(
        args.student_data_dir,
        image_size=args.image_size,
        center_crop=args.center_crop,
        horizontal_flip=not args.no_hflip,
        use_npy=args.train_use_depth_npy,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.real_batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"[Info] Train depth images: {len(train_ds)} @ {args.student_data_dir} | use_npy={args.train_use_depth_npy}", flush=True)

    # eval data: paired rgb-depth
    eval_loader = None
    if args.eval_rgb_dir and args.eval_depth_dir:
        eval_ds = PairedRGBDepthDataset(
            args.eval_rgb_dir,
            args.eval_depth_dir,
            image_size=args.image_size,
            use_depth_raw_npy=args.eval_use_depth_raw_npy,
            use_depth_npy=args.eval_use_depth_npy,
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.eval_batch,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        print(
            f"[Info] Eval pairs: {len(eval_ds)} rgb-depth pairs | "
            f"use_raw_npy={args.eval_use_depth_raw_npy} use_npy={args.eval_use_depth_npy}",
            flush=True
        )
    else:
        print("[Warn] eval_rgb_dir/eval_depth_dir not set. Paired eval will be skipped.", flush=True)

    # teacher (RGB diffusion)
    teacher_dir = Path(args.teacher_dir)
    teacher = UNet2DModel.from_pretrained(teacher_dir.as_posix()).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # schedulers
    ddpm = load_teacher_scheduler_or_fallback(teacher_dir, args.train_timesteps, args.beta_schedule)
    ddim_T = make_ddim(ddpm, prediction_type="epsilon")
    ddim_S = make_ddim(ddpm, prediction_type="epsilon")

    # student init: teacher-init + LoRA (resume supported)
    print("[Info] Initializing student from teacher weights + LoRA", flush=True)
    base_student = UNet2DModel.from_pretrained(teacher_dir.as_posix()).to(device)
    base_student.requires_grad_(False)

    global_step = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"[Info] Resuming LoRA from: {args.resume_checkpoint}", flush=True)
        student = PeftModel.from_pretrained(base_student, args.resume_checkpoint, is_trainable=True)
        global_step = _parse_step_from_path(args.resume_checkpoint)
        if global_step > 0:
            print(f"[Info] Parsed global_step={global_step}", flush=True)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            init_lora_weights=args.lora_init,
        )
        student = get_peft_model(base_student, lora_cfg)

    student.print_trainable_parameters()
    student.train()

    print(f"[Info] Teacher params: {count_parameters(teacher):,}", flush=True)
    print(f"[Info] Student params: {count_parameters(student):,}", flush=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        student.train()
        print(f"[Epoch {epoch}] start (global_step={global_step})", flush=True)

        for it, x0_real_depth in enumerate(train_loader, start=1):
            # random train DDIM steps
            if args.ddim_steps_min == args.ddim_steps_max:
                ddim_steps_train = int(args.ddim_steps_min)
            else:
                ddim_steps_train = int(torch.randint(args.ddim_steps_min, args.ddim_steps_max + 1, (1,)).item())

            # shared noise
            z = torch.randn((args.noise_batch, 3, args.image_size, args.image_size), device=device)
            x0_real_depth = x0_real_depth.to(device, non_blocking=True)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                if (use_amp and device.type == "cuda")
                else nullcontext()
            )

            # teacher forward on z + student forward on z
            with autocast_ctx:
                with torch.no_grad():
                    preds_T = teacher_predx0_seq(
                        teacher, ddim_T, z, steps=ddim_steps_train, eta=args.ddim_eta, device=device
                    )
                preds_S = student_predx0_seq_with_grad(
                    student, ddim_S, z, steps=ddim_steps_train, eta=args.ddim_eta, device=device
                )

            # inversion branch
            zT_real = invert_x0_to_zT_ddim_inverse_epspred(
                model=student, ddim=ddim_S, x0=x0_real_depth, steps=ddim_steps_train, device=device
            )
            preds_T_inv = teacher_predx0_seq(
                teacher, ddim_T, zT_real, steps=ddim_steps_train, eta=args.ddim_eta, device=device
            )
            x0_inv_T = preds_T_inv[-1]

            with autocast_ctx:
                total_loss, stats = compute_losses(
                    preds_T=preds_T,
                    preds_S=preds_S,
                    x0_real=x0_real_depth,
                    x0_inv_T=x0_inv_T,
                    embedder=embedder,
                    args=args,
                )

            if scaler is not None:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if args.max_grad_norm and args.max_grad_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if global_step % args.log_interval == 0:
                line = (
                    f"[Epoch {epoch:03d}] step={global_step:08d} "
                    f"total={float(total_loss.detach().item()):.6f} "
                    f"rkd={float(stats['loss_rkd'].item()):.6f} "
                    f"inv={float(stats['loss_inv'].item()):.6f} "
                    f"invinv={float(stats['loss_invinv'].item()):.6f} "
                    f"fd={float(stats['loss_fd'].item()):.6f} "
                    f"same={float(stats['loss_same'].item()):.6f}"
                )
                print(line, flush=True)
                with summary_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")

                if wandb_run is not None and wandb is not None:
                    try:
                        wandb.log({
                            **build_loss_logs(total_loss, stats, args),
                            "train/epoch": int(epoch),
                            "train/step": int(global_step),
                            "train/lr": float(args.lr),
                            "train/ddim_steps": int(ddim_steps_train),
                        }, step=global_step)
                    except Exception:
                        pass

            # optional sampling
            if args.sample_interval > 0 and (global_step % args.sample_interval == 0):
                if eval_loader is not None:
                    sample_rgb_gt_est_grid(
                        teacher=teacher,
                        student=student,
                        ddim_T=ddim_T,
                        ddim_S=ddim_S,
                        args=args,
                        device=device,
                        global_step=global_step,
                        out_dir=out_dir,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        wandb_run=wandb_run,
                        wandb=wandb,
                        eval_loader=eval_loader,
                    )
                else:
                    sample_student_depth_grid(
                        student=student,
                        ddim_S=ddim_S,
                        args=args,
                        device=device,
                        global_step=global_step,
                        out_dir=out_dir,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        wandb_run=wandb_run,
                        wandb=wandb,
                    )


            # paired eval (affine-invariant)
            if args.eval_interval > 0 and (global_step % args.eval_interval == 0) and (eval_loader is not None):
                eval_rgb2depth_affine_absrel_delta(
                    teacher=teacher,
                    student=student,
                    ddim_T=ddim_T,
                    ddim_S=ddim_S,
                    args=args,
                    device=device,
                    global_step=global_step,
                    out_dir=out_dir,
                    summary_path=summary_path,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    wandb_run=wandb_run,
                    wandb=wandb,
                    eval_loader=eval_loader,
                )

            # save
            if args.save_interval > 0 and (global_step % args.save_interval == 0):
                save_dir = out_dir / "ckpts" / f"ckpt_step{global_step:06d}"
                ensure_dir(save_dir)
                student.save_pretrained(save_dir.as_posix())
                ddpm.save_pretrained(save_dir.as_posix())
                print(f"[CKPT] Saved -> {save_dir}", flush=True)

        # epoch end save
        last_dir = out_dir / "last"
        ensure_dir(last_dir)
        student.save_pretrained(last_dir.as_posix())
        ddpm.save_pretrained(last_dir.as_posix())
        print(f"[Epoch {epoch}] saved last -> {last_dir}", flush=True)

    if wandb_run is not None and wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass


# ------------------------- Args -------------------------

def build_argparser():
    p = argparse.ArgumentParser("RKD training on depth + paired RGB->Depth eval (affine-invariant)")

    DATE = "0114"
    CUDA_NUM = 0
    BATCH_SIZE = 8
    RKD_METRIC = "dinov3"  # choices=["pixel", "inception", "clip", "dinov3"]
    RKD_W = 1.0
    INV_W= 1.0
    INVINV_W = 1.0
    FD_W = 1.0
    SAME_W = 1.0

    DEFAULT_RGB_ROOT = "/workspace/rkd_cifar10_1111/cifar10_png_linear_only/rgb"
    DEFAULT_DEPTH_ROOT = "/workspace/rkd_cifar10_1111/cifar10_png_linear_only/depth"
    DEFAULT_TEACHER_DIR = "/workspace/rkd_cifar10_1111/ddpm_cifar10_rgb_T400_DDIM50/ckpt_step150000"

    # paths
    # p.add_argument("--resume_checkpoint", type=str, default="0111_out_rgb2depth_rkd_affine_eval/pixel_RKD1.0_INV1.0_INVINV1.0_FD_0.01_SAME_1.0/last")
    # p.add_argument("--resume_checkpoint", type=str, default="0111_out_rgb2depth_rkd_affine_eval/pixel_RKD1.0_INV1.0_INVINV1.0_FD_0.01_SAME_1.0/ckpts/ckpt_step002000")
    p.add_argument("--resume_checkpoint", type=str, default="")

    
    p.add_argument("--student_data_dir", type=str, default=f"{DEFAULT_DEPTH_ROOT}/train")
    p.add_argument("--teacher_dir", type=str, default=DEFAULT_TEACHER_DIR)
    p.add_argument(
        "--output_dir",
        type=str,
        default=f"/workspace/rkd_cifar10_1111/{DATE}_out_rgb2depth_rkd_affine_eval/{RKD_METRIC}_RKD{RKD_W}_INV{INV_W}_INVINV{INVINV_W}_FD_{FD_W}_SAME_{SAME_W}",
    )

    # paired eval roots
    p.add_argument("--eval_rgb_dir", type=str, default=f"{DEFAULT_RGB_ROOT}/test")
    p.add_argument("--eval_depth_dir", type=str, default=f"{DEFAULT_DEPTH_ROOT}/test")

    # device & amp
    p.add_argument("--device", type=str, default=f"cuda:{CUDA_NUM}")
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--wandb_offline", action="store_true")

    # hf loading
    p.add_argument("--hf_local_only", action="store_true", help="HuggingFace from_pretrained(local_files_only=True)")

    # data
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--center_crop", action="store_true")
    p.add_argument("--no_hflip", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)

    # train depth load
    p.add_argument("--train_use_depth_npy", action="store_true", default=True)

    # eval GT load
    p.add_argument("--eval_use_depth_raw_npy", action="store_true", default=True,
                   help="Prefer depth.png.raw.npy + depth.png.mask.npy when available.")
    p.add_argument("--eval_use_depth_npy", action="store_true", default=True,
                   help="Fallback to depth.png.npy (normalized) if raw not present.")

    # train
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--real_batch", type=int, default=BATCH_SIZE)
    p.add_argument("--noise_batch", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # diffusion
    p.add_argument("--train_timesteps", type=int, default=400)
    p.add_argument("--beta_schedule", type=str, default="linear")
    p.add_argument("--ddim_steps_min", type=int, default=40)
    p.add_argument("--ddim_steps_max", type=int, default=60)
    p.add_argument("--ddim_eta", type=float, default=0.0)

    # LoRA
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_init", type=str, default="gaussian")
    p.add_argument("--lora_target_modules", type=str, nargs="+", default=["to_q", "to_k", "to_v", "to_out.0"])

    # losses
    p.add_argument("--w_rkd", type=float, default=RKD_W)
    p.add_argument("--w_inv", type=float, default=INV_W)
    p.add_argument("--w_invinv", type=float, default=INVINV_W)
    p.add_argument("--w_fd", type=float, default=FD_W)
    p.add_argument("--w_same", type=float, default=SAME_W)

    p.add_argument("--rkd_stride", type=int, default=1)
    p.add_argument("--rkd_teacher_ref", type=str, default="last", choices=["last", "matched"])
    p.add_argument("--same_mode", type=str, default="mean", choices=["mean", "last"])
    p.add_argument("--fd_eps", type=float, default=1e-8)

    # embedder
    p.add_argument("--rkd_metric", type=str, default=RKD_METRIC, choices=["pixel", "inception", "clip", "dinov3"])
    p.add_argument("--dino_model_name", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    p.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")

    # logging / save
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=2000)

    # optional sampling
    p.add_argument("--sample_interval", type=int, default=2000, help="0 disables sampling")
    p.add_argument("--sample_n", type=int, default=36)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--sample_eta", type=float, default=0.0)

    # eval params (paired)
    p.add_argument("--eval_interval", type=int, default=2000)
    p.add_argument("--eval_batch", type=int, default=128)
    p.add_argument("--eval_num_batches", type=int, default=0, help="0이면 전체 eval set 사용")
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--eval_eta", type=float, default=0.0)
    p.add_argument("--eval_gen_amp", action="store_true", help="Enable autocast for student generation (not inversion).")

    # affine-invariant eval
    p.add_argument("--eval_affine_align", action="store_true", default=True,
                   help="Align pred to GT with least squares scale+shift before metrics.")

    p.add_argument("--delta", type=float, default=1.25)
    p.add_argument("--depth_eps", type=float, default=1e-6)

    # NOTE: this is a disable flag (default True)
    p.add_argument("--eval_save_vis", action="store_false",
                   help="Disable saving one RGB/pred(m)/pred(aligned)/GT grid per eval.")

    # wandb names
    p.add_argument("--project", type=str, default=f"{DATE}_rkd-rgb2depth-cifar10")
    p.add_argument("--run_name", type=str, default=f"student-lora-depth-{RKD_METRIC}-RKD{RKD_W}_INV{INV_W}_INVINV{INVINV_W}_FD_{FD_W}_SAME_{SAME_W}")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    ensure_dir(Path(args.output_dir))
    train(args)
