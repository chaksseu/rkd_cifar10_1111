#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Distillation using ONLY:
  - x0 preds (pred_original_sample sequence)  -> flatten for RKD/INV/INVINV/FD
  - diffusion intermediate RAW features (hook) -> flatten for RKD/INV/INVINV/FD

No pretrained embedder (no CLIP/Inception/DINO).
Student diffusion features are NOT detached.

Losses:
  - RKD_x0, RKD_diff
  - INV_x0, INV_diff
  - INVINV_x0, INVINV_diff
  - FD_x0, FD_diff   (diagonal Fréchet distance; safe for huge D)
  - SAME (x0 trajectory regularization)

Deps:
  pip install diffusers torch torchvision peft wandb pytorch-fid
"""

import os
import math
import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DDIMInverseScheduler
from peft import LoraConfig, get_peft_model, PeftModel


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
    torch.cuda.manual_seed_all(seed)

def resolve_device(device_str: str) -> torch.device:
    try:
        dev = torch.device(device_str)
    except Exception:
        print(f"[Warn] Invalid --device '{device_str}'. Falling back to 'cpu'.", flush=True)
        return torch.device("cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev

def to_grid(images: torch.Tensor, nrow: int = 4) -> Image.Image:
    imgs = (images.clamp(-1, 1) + 1) / 2.0
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2)
    grid = (grid * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(grid)

def save_tensor_batch_to_dir(x: torch.Tensor, out_dir: Path, start_idx: int):
    ensure_dir(out_dir)
    x01 = (x.clamp(-1, 1) + 1) / 2.0
    x255 = (x01 * 255.0).clamp(0, 255).byte().cpu()
    for i in range(x255.shape[0]):
        arr = x255[i].permute(1, 2, 0).numpy()
        Image.fromarray(arr).save(out_dir / f"gen_{start_idx + i:06d}.png")

def collect_image_paths_recursive(root: Path, exts={".png", ".jpg", ".jpeg"}) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def flatten_real_cache(test_dir: Path, cache_dir: Path, use_symlink: bool = True) -> int:
    ensure_dir(cache_dir)
    existing = list(cache_dir.glob("*"))
    if len(existing) > 0:
        return len(existing)
    paths = collect_image_paths_recursive(test_dir)
    print(f"[FID] Flattening test set ({len(paths)} imgs) to {cache_dir} ...", flush=True)
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

def pdist_vec(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # x: (B, D)
    if x is None or x.numel() == 0 or x.shape[0] < 2:
        dev = x.device if x is not None else "cpu"
        return torch.zeros((0,), device=dev)
    return torch.pdist(x, p=2).clamp_min(eps)

def cdist_vec(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # x: (B, D), y: (B, D) -> (B*B,)
    if x is None or y is None or x.numel() == 0 or y.numel() == 0:
        dev = x.device if x is not None else "cpu"
        return torch.zeros((0,), device=dev)
    return torch.cdist(x, y, p=2).reshape(-1).clamp_min(eps)

def mean_from_vectors(vectors: List[torch.Tensor], device: torch.device, eps: float = 1e-12) -> torch.Tensor:
    if len(vectors) == 0:
        return torch.tensor(1.0, device=device)
    s = torch.zeros((), device=device, dtype=torch.float64)
    c = torch.zeros((), device=device, dtype=torch.float64)
    for vv in vectors:
        if vv is None or vv.numel() == 0:
            continue
        s = s + vv.sum().to(torch.float64)
        c = c + torch.tensor(float(vv.numel()), device=device, dtype=torch.float64)
    if float(c.item()) <= 0:
        return torch.tensor(1.0, device=device)
    m = (s / c.clamp_min(1.0)).to(torch.float32)
    return m.clamp_min(eps)

def frechet_distance_diag(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Diagonal-cov Fréchet distance (safe for large D).
    X, Y: (B, D)
    """
    X = X.float()
    Y = Y.float()
    mu_x = X.mean(dim=0)
    mu_y = Y.mean(dim=0)
    vx = X.var(dim=0, unbiased=False) + eps
    vy = Y.var(dim=0, unbiased=False) + eps
    mean_term = (mu_x - mu_y).pow(2).sum()
    trace_term = (vx + vy - 2.0 * torch.sqrt(vx * vy)).sum()
    return (mean_term + trace_term).clamp_min(0.0)

def _mean_and_cov(X: torch.Tensor, eps: float = 1e-6):
    X = X.to(torch.float64)
    N, D = X.shape
    if N == 0:
        mu = torch.zeros(D, dtype=X.dtype, device=X.device)
        C  = torch.eye(D, dtype=X.dtype, device=X.device)
        return mu, C
    mu = X.mean(dim=0, keepdim=True)
    xc = X - mu
    denom = (N - 1) if N > 1 else 1
    C = (xc.t() @ xc) / denom
    I = torch.eye(D, dtype=X.dtype, device=X.device)
    C = 0.5 * (C + C.t()) + eps * I
    return mu.squeeze(0), C

def _sqrtm_psd(A: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    A = 0.5 * (A + A.t())
    evals, vecs = torch.linalg.eigh(A)
    evals = (evals + eps).clamp_min(0)
    return (vecs * evals.sqrt().unsqueeze(0)) @ vecs.t()

def fid_gaussian_torch(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert X.dim() == 2 and Y.dim() == 2 and X.size(1) == Y.size(1)
    out_dtype = X.dtype

    mx, Cx = _mean_and_cov(X, eps)
    my, Cy = _mean_and_cov(Y, eps)

    mean_term = ((mx - my) ** 2).sum()

    Cy_sqrt = _sqrtm_psd(Cy, eps=eps)
    B = Cy_sqrt @ Cx @ Cy_sqrt
    B_sqrt = _sqrtm_psd(B, eps=eps)

    trace_term = torch.trace(Cx + Cy - 2.0 * B_sqrt)
    return (mean_term + trace_term).clamp_min(0.0).to(out_dtype)

# ------------------------- Dataset -------------------------

class StudentImageFolderDataset(Dataset):
    def __init__(self, root: str, image_size: int = 32, center_crop: bool = False, horizontal_flip: bool = True):
        self.root = Path(root)
        exts = {".png", ".jpg", ".jpeg"}
        self.files = [p for p in self.root.rglob("*") if p.suffix.lower() in exts and p.is_file()]
        if len(self.files) == 0:
            raise FileNotFoundError(f"No images found under {self.root}!")

        tfms = [T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True)]
        if center_crop:
            tfms.append(T.CenterCrop(image_size))
        if horizontal_flip:
            tfms.append(T.RandomHorizontalFlip(p=0.5))
        tfms.append(T.ToTensor())
        self.to_tensor = T.Compose(tfms)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            x01 = self.to_tensor(img)
        return x01 * 2.0 - 1.0


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


# ------------------------- UNet RAW feature hooks (NO pooling; flatten at loss time) -------------------------

def _unwrap_unet_for_hooks(m: nn.Module) -> nn.Module:
    # PeftModel(LoRA) wrapper -> base UNet
    if isinstance(m, PeftModel):
        if hasattr(m, "get_base_model"):
            return m.get_base_model()
        if hasattr(m, "base_model"):
            bm = m.base_model
            if hasattr(bm, "model"):
                return bm.model
            return bm
    return m

def _pick_feature_tensor(out: Any) -> torch.Tensor:
    """
    diffusers blocks may return tensor or tuple/list.
    We choose the first tensor that looks like an activation (prefer 4D).
    """
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        # prefer 4D
        for v in out:
            if torch.is_tensor(v) and v.dim() == 4:
                return v
        for v in out:
            if torch.is_tensor(v):
                return v
    if isinstance(out, dict):
        for v in out.values():
            if torch.is_tensor(v) and v.dim() == 4:
                return v
        for v in out.values():
            if torch.is_tensor(v):
                return v
    raise RuntimeError("Could not pick a feature tensor from hook output")

class RawFeatureCollector:
    """
    Collect RAW intermediate activations from specified blocks for each forward call.
    For each timestep: we store a dict(spec->tensor).
    Later: we flatten each tensor and concat: (B,C,H,W)->(B, C*H*W), NO pooling.
    Student tensors are NOT detached.
    """
    def __init__(self, unet: nn.Module, specs: List[str], cast: str = "none"):
        self.unet = _unwrap_unet_for_hooks(unet)
        self.specs = [str(s) for s in specs]
        self.cast = cast  # none|fp16|fp32
        self._buf: Dict[str, torch.Tensor] = {}
        self._handles = []

        mods = self._resolve_modules(self.unet, self.specs)
        for name, mod in mods:
            self._handles.append(mod.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def fn(module, inp, out):
            t = _pick_feature_tensor(out)
            if self.cast == "fp16":
                t = t.to(torch.float16)
            elif self.cast == "fp32":
                t = t.to(torch.float32)
            # IMPORTANT: NO detach
            self._buf[name] = t
        return fn

    def reset(self):
        self._buf = {}

    def flatten_and_concat(self) -> torch.Tensor:
        missing = [s for s in self.specs if s not in self._buf]
        if len(missing) > 0:
            raise RuntimeError(f"[RawFeatureCollector] Missing specs: {missing}")
        flats = []
        for s in self.specs:
            t = self._buf[s]
            flats.append(t.reshape(t.shape[0], -1))  # flatten ONLY
        return torch.cat(flats, dim=1)

    def close(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    @staticmethod
    def _resolve_modules(unet: nn.Module, specs: List[str]) -> List[Tuple[str, nn.Module]]:
        modules = []
        down_blocks = getattr(unet, "down_blocks", None)
        up_blocks = getattr(unet, "up_blocks", None)
        mid_block = getattr(unet, "mid_block", None)

        if down_blocks is None or up_blocks is None or mid_block is None:
            raise RuntimeError("UNet2DModel does not have expected blocks (down_blocks/up_blocks/mid_block).")

        def add(name, mod):
            if mod is None:
                raise RuntimeError(f"Module for spec '{name}' is None")
            modules.append((name, mod))

        for s in specs:
            if s == "mid":
                add("mid", mid_block); continue
            if s == "down_last":
                add("down_last", down_blocks[-1]); continue
            if s == "up_last":
                add("up_last", up_blocks[-1]); continue
            if s.startswith("down."):
                idx = int(s.split(".")[1])
                add(s, down_blocks[idx]); continue
            if s.startswith("up."):
                idx = int(s.split(".")[1])
                add(s, up_blocks[idx]); continue
            raise ValueError(f"Unknown diff spec '{s}'. Use mid|down.i|up.i|down_last|up_last")
        return modules


# ------------------------- Rollout / Inversion (collect x0 + diff features) -------------------------

def teacher_rollout(
    teacher: UNet2DModel,
    ddim_T: DDIMScheduler,
    z: torch.Tensor,
    steps: int,
    eta: float,
    device: torch.device,
    feat_col: Optional[RawFeatureCollector],
    store_stride: int = 1,
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
    local = DDIMScheduler.from_config(ddim_T.config)
    local.set_timesteps(steps, device=device)
    x = z.to(device)

    teacher.eval()
    preds_x0: List[torch.Tensor] = []
    feats: List[torch.Tensor] = []

    for i, t in enumerate(local.timesteps):
        if feat_col is not None:
            feat_col.reset()

        x_in = local.scale_model_input(x, t)
        eps = teacher(x_in, t).sample
        out = local.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x = out.prev_sample
        preds_x0.append(out.pred_original_sample)

        if feat_col is not None and (i % max(1, store_stride) == 0):
            feats.append(feat_col.flatten_and_concat())  # (B, D) already flattened

    return preds_x0, (feats if feat_col is not None else None) # preds_x0, (feats if feat_col is not None else None)

def student_rollout_with_grad(
    student,
    ddim_S: DDIMScheduler,
    z: torch.Tensor,
    steps: int,
    eta: float,
    device: torch.device,
    feat_col: Optional[RawFeatureCollector],
    store_stride: int = 1,
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
    local = DDIMScheduler.from_config(ddim_S.config)
    local.set_timesteps(steps, device=device)
    x = z.to(device)

    student.train()
    preds_x0: List[torch.Tensor] = []
    feats: List[torch.Tensor] = []

    for i, t in enumerate(local.timesteps):
        if feat_col is not None:
            feat_col.reset()

        x_in = local.scale_model_input(x, t)
        eps = student(x_in, t).sample
        out = local.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x = out.prev_sample
        preds_x0.append(out.pred_original_sample)

        if feat_col is not None and (i % max(1, store_stride) == 0):
            feats.append(feat_col.flatten_and_concat())  # NO detach

    return preds_x0, (feats if feat_col is not None else None)

def invert_x0_to_zT_with_grad_and_feats(
    student,
    ddim_S: DDIMScheduler,
    x0: torch.Tensor,
    steps: int,
    device: torch.device,
    feat_col: Optional[RawFeatureCollector],
    store_stride: int = 1,
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    inv = DDIMInverseScheduler.from_config(ddim_S.config)
    inv.set_timesteps(steps, device=device)

    student.train()
    xt = x0
    feats: List[torch.Tensor] = []

    for i, t in enumerate(inv.timesteps):
        if feat_col is not None:
            feat_col.reset()

        x_in = inv.scale_model_input(xt, t)
        eps = student(x_in, t).sample
        xt = inv.step(eps, t, xt).prev_sample

        if feat_col is not None and (i % max(1, store_stride) == 0):
            feats.append(feat_col.flatten_and_concat())  # NO detach

    return xt, (feats if feat_col is not None else None)

@torch.no_grad()
def sample_images_ddim(
    student,
    ddim_S: DDIMScheduler,
    num_images: int,
    image_size: int,
    device: torch.device,
    steps: int,
    eta: float,
    generator: Optional[torch.Generator] = None,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    was_training = student.training
    student.eval()

    local = DDIMScheduler.from_config(ddim_S.config)
    local.set_timesteps(steps, device=device)

    dtype = next(student.parameters()).dtype
    x = torch.randn((num_images, 3, image_size, image_size), device=device, dtype=dtype, generator=generator)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )

    with autocast_ctx:
        for t in local.timesteps:
            x_in = local.scale_model_input(x, t)
            eps = student(x_in, t).sample
            x = local.step(model_output=eps, timestep=t, sample=x, eta=eta, generator=generator).prev_sample

    if was_training:
        student.train()
    return x


# ------------------------- Losses (x0 + diff feature 모두 flatten 기반) -------------------------

def compute_losses(
    preds_T: List[torch.Tensor],
    preds_S: List[torch.Tensor],
    feats_gen_T: Optional[List[torch.Tensor]],
    feats_gen_S: Optional[List[torch.Tensor]],
    x0_real: torch.Tensor,
    x0_inv_T: torch.Tensor,
    feats_inv_T: Optional[List[torch.Tensor]],
    feats_inv_S: Optional[List[torch.Tensor]],
    args,
):
    eps = 1e-12
    device = x0_real.device

    # x0 flatten
    def fx0(x):  # x: (B,3,H,W)
        return x.reshape(x.shape[0], -1)

    # get last diff feature (already flattened (B,D))
    def last_feat(fs: Optional[List[torch.Tensor]]) -> Optional[torch.Tensor]:
        if fs is None or len(fs) == 0:
            return None
        return fs[-1]

    # ---------------- RKD x0 ----------------
    rkd_x0_s_list, rkd_x0_t_list = [], []
    if args.w_rkd_x0 != 0.0:
        if args.rkd_x0_teacher_ref == "last":
            T_ref = fx0(preds_T[-1].float())
            T_ref_pdist = pdist_vec(T_ref, eps=eps)
            for k in range(0, len(preds_S), max(1, args.rkd_x0_stride)):
                S_k = fx0(preds_S[k].float())
                rkd_x0_s_list.append(pdist_vec(S_k, eps=eps))
                rkd_x0_t_list.append(T_ref_pdist)
        else:  # matched
            for k in range(0, len(preds_S), max(1, args.rkd_x0_stride)):
                S_k = fx0(preds_S[k].float())
                T_k = fx0(preds_T[k].float())
                rkd_x0_s_list.append(pdist_vec(S_k, eps=eps))
                rkd_x0_t_list.append(pdist_vec(T_k, eps=eps))

    # ---------------- RKD diff (flatten feature) ----------------
    rkd_df_s_list, rkd_df_t_list = [], []
    if args.w_rkd_diff != 0.0:
        assert feats_gen_T is not None and feats_gen_S is not None, "w_rkd_diff requires diffusion features"
        assert len(feats_gen_T) == len(feats_gen_S), f"gen feats len mismatch: T={len(feats_gen_T)} S={len(feats_gen_S)}"

        if args.rkd_diff_teacher_ref == "last":
            T_ref = feats_gen_T[-1]
            T_ref_pdist = pdist_vec(T_ref, eps=eps)
            for k in range(0, len(feats_gen_S), max(1, args.rkd_diff_stride)):
                rkd_df_s_list.append(pdist_vec(feats_gen_S[k], eps=eps))
                rkd_df_t_list.append(T_ref_pdist)
        else:  # matched
            for k in range(0, len(feats_gen_S), max(1, args.rkd_diff_stride)):
                rkd_df_s_list.append(pdist_vec(feats_gen_S[k], eps=eps))
                rkd_df_t_list.append(pdist_vec(feats_gen_T[k], eps=eps))

    # ---------------- INV x0 ----------------
    inv_x0_s = inv_x0_t = None
    if args.w_inv_x0 != 0.0:
        inv_x0_s = cdist_vec(fx0(preds_S[-1].float()), fx0(x0_real.float()), eps=eps)
        inv_x0_t = cdist_vec(fx0(preds_T[-1].float()), fx0(x0_inv_T.float()), eps=eps)

    # ---------------- INV diff (flatten feature) ----------------
    inv_df_s = inv_df_t = None
    if args.w_inv_diff != 0.0:
        S_gen_last = last_feat(feats_gen_S)
        T_gen_last = last_feat(feats_gen_T)
        S_inv_last = last_feat(feats_inv_S)
        T_inv_last = last_feat(feats_inv_T)
        assert S_gen_last is not None and T_gen_last is not None and S_inv_last is not None and T_inv_last is not None, \
            "w_inv_diff requires gen & inv diffusion features for both teacher/student"
        # Student: dist(gen_feat_last, inv_feat_last), Teacher: dist(gen_feat_last, inv_feat_last)
        inv_df_s = cdist_vec(S_gen_last, S_inv_last, eps=eps)
        inv_df_t = cdist_vec(T_gen_last, T_inv_last, eps=eps)

    # ---------------- INVINV x0 ----------------
    invinv_x0_s = invinv_x0_t = None
    if args.w_invinv_x0 != 0.0:
        invinv_x0_s = pdist_vec(fx0(x0_real.float()), eps=eps)
        invinv_x0_t = pdist_vec(fx0(x0_inv_T.float()), eps=eps)

    # ---------------- INVINV diff (flatten feature) ----------------
    invinv_df_s = invinv_df_t = None
    if args.w_invinv_diff != 0.0:
        S_inv_last = last_feat(feats_inv_S)
        T_inv_last = last_feat(feats_inv_T)
        assert S_inv_last is not None and T_inv_last is not None, "w_invinv_diff requires inv diffusion features"
        invinv_df_s = pdist_vec(S_inv_last, eps=eps)
        invinv_df_t = pdist_vec(T_inv_last, eps=eps)

    # ---------------- FD x0 (diag) ----------------
    fd_x0 = torch.tensor(0.0, device=device)
    if args.w_fd_x0 != 0.0:
        fd_s = frechet_distance_diag(fx0(preds_S[-1]), fx0(x0_real), eps=args.fd_eps)
        fd_t = frechet_distance_diag(fx0(preds_T[-1]), fx0(x0_inv_T), eps=args.fd_eps)
        fd_x0 = fd_s + fd_t

    # ---------------- FD diff (diag; flattened feature) ----------------
    fd_df = torch.tensor(0.0, device=device)
    if args.w_fd_diff != 0.0:
        S_gen_last = last_feat(feats_gen_S)
        T_gen_last = last_feat(feats_gen_T)
        S_inv_last = last_feat(feats_inv_S)
        T_inv_last = last_feat(feats_inv_T)
        assert S_gen_last is not None and T_gen_last is not None and S_inv_last is not None and T_inv_last is not None, \
            "w_fd_diff requires gen & inv diffusion features for both teacher/student"
        fd_s = frechet_distance_diag(S_gen_last, S_inv_last, eps=args.fd_eps)
        fd_t = frechet_distance_diag(T_gen_last, T_inv_last, eps=args.fd_eps)
        fd_df = fd_s + fd_t

    # ---------------- SAME (x0 trajectory) ----------------
    loss_same = torch.tensor(0.0, device=device)
    if args.w_same != 0.0:
        xs = torch.stack(preds_S, dim=0)
        if args.same_mode == "mean":
            mu = xs.mean(dim=0, keepdim=True)
            loss_same = F.mse_loss(xs, mu.expand_as(xs), reduction="mean")
        else:
            ref = xs[-1].detach()
            loss_same = F.mse_loss(xs[:-1], ref.unsqueeze(0).expand_as(xs[:-1]), reduction="mean")

    # ---------------- Mean normalization for distance-vectors ----------------
    student_parts: List[torch.Tensor] = []
    teacher_parts: List[torch.Tensor] = []

    if args.w_rkd_x0 != 0.0 and len(rkd_x0_s_list) > 0:
        student_parts.append(torch.cat(rkd_x0_s_list, dim=0))
        teacher_parts.append(torch.cat(rkd_x0_t_list, dim=0))
    if args.w_rkd_diff != 0.0 and len(rkd_df_s_list) > 0:
        student_parts.append(torch.cat(rkd_df_s_list, dim=0))
        teacher_parts.append(torch.cat(rkd_df_t_list, dim=0))
    if args.w_inv_x0 != 0.0 and inv_x0_s is not None:
        student_parts.append(inv_x0_s); teacher_parts.append(inv_x0_t)
    if args.w_inv_diff != 0.0 and inv_df_s is not None:
        student_parts.append(inv_df_s); teacher_parts.append(inv_df_t)
    if args.w_invinv_x0 != 0.0 and invinv_x0_s is not None:
        student_parts.append(invinv_x0_s); teacher_parts.append(invinv_x0_t)
    if args.w_invinv_diff != 0.0 and invinv_df_s is not None:
        student_parts.append(invinv_df_s); teacher_parts.append(invinv_df_t)

    if len(student_parts) > 0:
        student_mean = mean_from_vectors(student_parts, device=device, eps=eps)
        teacher_mean = mean_from_vectors(teacher_parts, device=device, eps=eps)
    else:
        student_mean = torch.tensor(1.0, device=device)
        teacher_mean = torch.tensor(1.0, device=device)

    # normalize vectors
    if args.w_rkd_x0 != 0.0:
        rkd_x0_s_list = [d / student_mean for d in rkd_x0_s_list]
        rkd_x0_t_list = [d / teacher_mean for d in rkd_x0_t_list]
    if args.w_rkd_diff != 0.0:
        rkd_df_s_list = [d / student_mean for d in rkd_df_s_list]
        rkd_df_t_list = [d / teacher_mean for d in rkd_df_t_list]
    if args.w_inv_x0 != 0.0 and inv_x0_s is not None:
        inv_x0_s = inv_x0_s / student_mean
        inv_x0_t = inv_x0_t / teacher_mean
    if args.w_inv_diff != 0.0 and inv_df_s is not None:
        inv_df_s = inv_df_s / student_mean
        inv_df_t = inv_df_t / teacher_mean
    if args.w_invinv_x0 != 0.0 and invinv_x0_s is not None:
        invinv_x0_s = invinv_x0_s / student_mean
        invinv_x0_t = invinv_x0_t / teacher_mean
    if args.w_invinv_diff != 0.0 and invinv_df_s is not None:
        invinv_df_s = invinv_df_s / student_mean
        invinv_df_t = invinv_df_t / teacher_mean

    # scalar losses
    loss_rkd_x0 = torch.tensor(0.0, device=device)
    if args.w_rkd_x0 != 0.0 and len(rkd_x0_s_list) > 0:
        loss_rkd_x0 = sum(F.mse_loss(ds, dt, reduction="mean") for ds, dt in zip(rkd_x0_s_list, rkd_x0_t_list)) / max(1, len(rkd_x0_s_list))

    loss_rkd_diff = torch.tensor(0.0, device=device)
    if args.w_rkd_diff != 0.0 and len(rkd_df_s_list) > 0:
        loss_rkd_diff = sum(F.mse_loss(ds, dt, reduction="mean") for ds, dt in zip(rkd_df_s_list, rkd_df_t_list)) / max(1, len(rkd_df_s_list))

    loss_inv_x0 = torch.tensor(0.0, device=device)
    if args.w_inv_x0 != 0.0 and inv_x0_s is not None:
        loss_inv_x0 = F.mse_loss(inv_x0_s, inv_x0_t, reduction="mean")

    loss_inv_diff = torch.tensor(0.0, device=device)
    if args.w_inv_diff != 0.0 and inv_df_s is not None:
        loss_inv_diff = F.mse_loss(inv_df_s, inv_df_t, reduction="mean")

    loss_invinv_x0 = torch.tensor(0.0, device=device)
    if args.w_invinv_x0 != 0.0 and invinv_x0_s is not None:
        loss_invinv_x0 = F.mse_loss(invinv_x0_s, invinv_x0_t, reduction="mean")

    loss_invinv_diff = torch.tensor(0.0, device=device)
    if args.w_invinv_diff != 0.0 and invinv_df_s is not None:
        loss_invinv_diff = F.mse_loss(invinv_df_s, invinv_df_t, reduction="mean")

    total = (
        args.w_rkd_x0 * loss_rkd_x0 +
        args.w_rkd_diff * loss_rkd_diff +
        args.w_inv_x0 * loss_inv_x0 +
        args.w_inv_diff * loss_inv_diff +
        args.w_invinv_x0 * loss_invinv_x0 +
        args.w_invinv_diff * loss_invinv_diff +
        args.w_fd_x0 * fd_x0 +
        args.w_fd_diff * fd_df +
        args.w_same * loss_same
    )

    stats = {
        "loss_rkd_x0": loss_rkd_x0.detach(),
        "loss_rkd_diff": loss_rkd_diff.detach(),
        "loss_inv_x0": loss_inv_x0.detach(),
        "loss_inv_diff": loss_inv_diff.detach(),
        "loss_invinv_x0": loss_invinv_x0.detach(),
        "loss_invinv_diff": loss_invinv_diff.detach(),
        "loss_fd_x0": fd_x0.detach(),
        "loss_fd_diff": fd_df.detach(),
        "loss_same": loss_same.detach(),
        "student_mean_dist": student_mean.detach(),
        "teacher_mean_dist": teacher_mean.detach(),
    }
    return total, stats


def build_loss_logs(total_loss: torch.Tensor, stats: dict, args) -> dict:
    return {
        "loss/total": float(total_loss.detach().item()),
        "loss/rkd_x0": float(stats["loss_rkd_x0"].item() * args.w_rkd_x0),
        "loss/rkd_diff": float(stats["loss_rkd_diff"].item() * args.w_rkd_diff),
        "loss/inv_x0": float(stats["loss_inv_x0"].item() * args.w_inv_x0),
        "loss/inv_diff": float(stats["loss_inv_diff"].item() * args.w_inv_diff),
        "loss/invinv_x0": float(stats["loss_invinv_x0"].item() * args.w_invinv_x0),
        "loss/invinv_diff": float(stats["loss_invinv_diff"].item() * args.w_invinv_diff),
        "loss/fd_x0": float(stats["loss_fd_x0"].item() * args.w_fd_x0),
        "loss/fd_diff": float(stats["loss_fd_diff"].item() * args.w_fd_diff),
        "loss/same": float(stats["loss_same"].item() * args.w_same),
        "norm/student_mean_dist": float(stats["student_mean_dist"].item()),
        "norm/teacher_mean_dist": float(stats["teacher_mean_dist"].item()),
    }


# ------------------------- Eval: pytorch-fid (optional) -------------------------

def compute_fid_pytorch_fid(real_dir: Path, gen_dir: Path, device: torch.device, batch_size: int, dims: int) -> float:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    fid = calculate_fid_given_paths(
        [real_dir.as_posix(), gen_dir.as_posix()],
        batch_size=batch_size,
        device=device,
        dims=dims,
    )
    return float(fid)

@torch.no_grad()
def eval_sample_and_fid(
    student,
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
    fid_real_all_dir: Optional[Path],
    num_test_imgs_all: int,
):
    ensure_dir(out_dir / "samples")
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + int(global_step))

    imgs = sample_images_ddim(
        student=student,
        ddim_S=ddim_S,
        num_images=args.sample_n,
        image_size=args.image_size,
        device=device,
        steps=args.sample_steps,
        eta=args.sample_eta,
        generator=gen,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    nrow = int(math.isqrt(args.sample_n))
    if nrow * nrow != args.sample_n:
        nrow = min(args.sample_n, 8)

    grid = to_grid(imgs, nrow=nrow)
    grid_path = out_dir / "samples" / f"samples_step{global_step:08d}.png"
    grid.save(grid_path)

    if wandb_run is not None and wandb is not None:
        try:
            wandb.log({"eval/samples": wandb.Image(grid)}, step=global_step)
        except Exception:
            pass

    with summary_path.open("a", encoding="utf-8") as f:
        f.write(f"[EVAL] step={global_step:08d} saved_samples={grid_path.as_posix()}\n")

    if args.disable_fid:
        return
    if fid_real_all_dir is None or (not fid_real_all_dir.exists()):
        return

    fid_num = num_test_imgs_all if args.fid_num_samples <= 0 else int(min(args.fid_num_samples, num_test_imgs_all))
    gen_dir = out_dir / "fid" / f"step{global_step:08d}"
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    ensure_dir(gen_dir)

    remaining = fid_num
    cursor = 0
    while remaining > 0:
        cur = min(args.fid_gen_batch, remaining)
        xs = sample_images_ddim(
            student=student,
            ddim_S=ddim_S,
            num_images=cur,
            image_size=args.image_size,
            device=device,
            steps=args.sample_steps,
            eta=args.sample_eta,
            generator=gen,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        save_tensor_batch_to_dir(xs, gen_dir, start_idx=cursor)
        cursor += cur
        remaining -= cur

    try:
        fid_all = compute_fid_pytorch_fid(
            real_dir=fid_real_all_dir,
            gen_dir=gen_dir,
            device=device,
            batch_size=args.fid_batch_size,
            dims=args.fid_dims,
        )
    except Exception as e:
        msg = f"[FID] step={global_step} failed: {e}"
        print(msg, flush=True)
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
        if not args.fid_keep_gen:
            shutil.rmtree(gen_dir, ignore_errors=True)
        return

    print(f"[FID] step={global_step} fid_all={fid_all:.4f} (N={fid_num})", flush=True)
    with summary_path.open("a", encoding="utf-8") as f:
        f.write(f"[FID] step={global_step:08d} fid_all={fid_all:.6f} N={fid_num}\n")

    if wandb_run is not None and wandb is not None:
        try:
            wandb.log({"eval/fid_all": fid_all, "eval/fid_num_samples": int(fid_num)}, step=global_step)
        except Exception:
            pass

    if not args.fid_keep_gen:
        shutil.rmtree(gen_dir, ignore_errors=True)


# ------------------------- Train -------------------------

def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)

def train(args):
    torch.backends.cudnn.benchmark = True
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    set_seed(args.seed)
    device = resolve_device(args.device)

    # wandb (optional)
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

    use_amp = (device.type == "cuda") and (args.mixed_precision != "no")
    amp_dtype = None
    scaler = None
    if use_amp:
        if args.mixed_precision == "fp16":
            amp_dtype = torch.float16
            scaler = _make_grad_scaler(enabled=True)
        elif args.mixed_precision == "bf16":
            amp_dtype = torch.bfloat16
            scaler = None
        else:
            use_amp = False

    if device.type == "cuda":
        torch.cuda.set_device(device)

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "samples")
    ensure_dir(out_dir / "ckpts")
    ensure_dir(out_dir / "fid")
    summary_path = out_dir / "summary.txt"

    dataset = StudentImageFolderDataset(
        args.student_data_dir,
        image_size=args.image_size,
        center_crop=args.center_crop,
        horizontal_flip=not args.no_hflip,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.real_batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"[Info] Student images: {len(dataset)} @ {args.student_data_dir}", flush=True)
    print(f"[Info] Device: {device} | AMP: {args.mixed_precision if use_amp else 'no'}", flush=True)

    # eval FID real cache
    fid_real_all_dir = None
    num_test_imgs_all = 0
    if (not args.disable_fid) and args.test_dir and len(args.test_dir.strip()) > 0:
        test_dir = Path(args.test_dir)
        if test_dir.exists():
            fid_real_root = out_dir / "fid" / "real_cache"
            fid_real_all_dir = fid_real_root / "all"
            num_test_imgs_all = flatten_real_cache(test_dir, fid_real_all_dir, use_symlink=not args.fid_no_symlink)
            print(f"[FID] Real cache: N={num_test_imgs_all} @ {fid_real_all_dir}", flush=True)
        else:
            args.disable_fid = True
    else:
        args.disable_fid = True

    # teacher
    teacher_dir = Path(args.teacher_dir)
    teacher = UNet2DModel.from_pretrained(teacher_dir.as_posix()).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # schedulers
    ddpm = load_teacher_scheduler_or_fallback(teacher_dir, args.train_timesteps, args.beta_schedule)
    ddim_T = make_ddim(ddpm, prediction_type="epsilon")
    ddim_S = make_ddim(ddpm, prediction_type="epsilon")

    # student init: teacher init + LoRA
    print("[Info] Initializing Student...", flush=True)
    base_student = UNet2DModel.from_pretrained(teacher_dir.as_posix()).to(device)
    base_student.requires_grad_(False)

    global_step = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"[Info] Resume: {args.resume_checkpoint}", flush=True)
        student = PeftModel.from_pretrained(base_student, args.resume_checkpoint, is_trainable=True)
        try:
            ckpt_name = Path(args.resume_checkpoint).name
            if "step" in ckpt_name:
                global_step = int(ckpt_name.split("step")[-1])
                print(f"[Info] Resuming at global_step={global_step}", flush=True)
        except Exception:
            pass
    else:
        print("[Info] New training: init LoRA", flush=True)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            init_lora_weights=args.lora_init,
        )
        student = get_peft_model(base_student, lora_config)

    student.print_trainable_parameters()
    student.train()

    print(f"[Info] Teacher params: {count_parameters(teacher):,}", flush=True)
    print(f"[Info] Student params: {count_parameters(student):,}", flush=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad(set_to_none=True)

    # feature collectors
    feat_col_T = RawFeatureCollector(teacher, specs=args.diff_specs, cast=args.diff_cast) if args.use_diff_feats else None
    feat_col_S = RawFeatureCollector(student, specs=args.diff_specs, cast=args.diff_cast) if args.use_diff_feats else None

    for epoch in range(1, args.epochs + 1):
        student.train()
        print(f"[Epoch {epoch}] start (step={global_step})", flush=True)

        for it, x0_real in enumerate(loader, start=1):
            # sample train steps
            if args.ddim_steps_min == args.ddim_steps_max:
                ddim_steps_train = int(args.ddim_steps_min)
            else:
                ddim_steps_train = int(torch.randint(args.ddim_steps_min, args.ddim_steps_max + 1, (1,)).item())

            z = torch.randn((args.noise_batch, 3, args.image_size, args.image_size), device=device)
            x0_real = x0_real.to(device, non_blocking=True)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                if (use_amp and device.type == "cuda")
                else nullcontext()
            )

            # (1) rollout from z: get pred_x0 seq + diff feature seq (flattened)
            with autocast_ctx:
                with torch.no_grad():
                    preds_T, feats_gen_T = teacher_rollout(
                        teacher, ddim_T, z, steps=ddim_steps_train, eta=args.ddim_eta, device=device,
                        feat_col=feat_col_T,
                        store_stride=max(1, args.diff_store_stride),
                    )
                preds_S, feats_gen_S = student_rollout_with_grad(
                    student, ddim_S, z, steps=ddim_steps_train, eta=args.ddim_eta, device=device,
                    feat_col=feat_col_S,
                    store_stride=max(1, args.diff_store_stride),
                )

            # (2) inversion: x0_real -> zT_real using student (with grad + feats)
            zT_real, feats_inv_S = invert_x0_to_zT_with_grad_and_feats(
                student, ddim_S, x0_real, steps=ddim_steps_train, device=device,
                feat_col=feat_col_S,
                store_stride=max(1, args.diff_store_stride),
            )

            # (3) teacher rollout from zT_real -> x0_inv_T (no_grad + feats)
            preds_T_inv, feats_inv_T = teacher_rollout(
                teacher, ddim_T, zT_real, steps=ddim_steps_train, eta=args.ddim_eta, device=device,
                feat_col=feat_col_T,
                store_stride=max(1, args.diff_store_stride),
            )
            x0_inv_T = preds_T_inv[-1]

            # (4) losses
            with autocast_ctx:
                total_loss, stats = compute_losses(
                    preds_T=preds_T,
                    preds_S=preds_S,
                    feats_gen_T=feats_gen_T if args.use_diff_feats else None,
                    feats_gen_S=feats_gen_S if args.use_diff_feats else None,
                    x0_real=x0_real,
                    x0_inv_T=x0_inv_T,
                    feats_inv_T=feats_inv_T if args.use_diff_feats else None,
                    feats_inv_S=feats_inv_S if args.use_diff_feats else None,
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
                    f"rkd_x0={float(stats['loss_rkd_x0'].item()):.6f} "
                    f"rkd_diff={float(stats['loss_rkd_diff'].item()):.6f} "
                    f"inv_x0={float(stats['loss_inv_x0'].item()):.6f} "
                    f"inv_diff={float(stats['loss_inv_diff'].item()):.6f} "
                    f"invinv_x0={float(stats['loss_invinv_x0'].item()):.6f} "
                    f"invinv_diff={float(stats['loss_invinv_diff'].item()):.6f} "
                    f"fd_x0={float(stats['loss_fd_x0'].item()):.6f} "
                    f"fd_diff={float(stats['loss_fd_diff'].item()):.6f} "
                    f"same={float(stats['loss_same'].item()):.6f}"
                )
                print(line, flush=True)
                with summary_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")

                if wandb_run is not None and wandb is not None:
                    try:
                        wandb.log(
                            {
                                **build_loss_logs(total_loss, stats, args),
                                "train/epoch": int(epoch),
                                "train/step": int(global_step),
                                "train/lr": float(args.lr),
                            },
                            step=global_step,
                        )
                    except Exception:
                        pass

            if args.sample_interval > 0 and (global_step % args.sample_interval == 0):
                eval_sample_and_fid(
                    student=student,
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
                    fid_real_all_dir=fid_real_all_dir,
                    num_test_imgs_all=num_test_imgs_all,
                )

            if args.save_interval > 0 and (global_step % args.save_interval == 0):
                save_dir = out_dir / "ckpts" / f"ckpt_step{global_step:06d}"
                ensure_dir(save_dir)
                student.save_pretrained(save_dir.as_posix())
                ddpm.save_pretrained(save_dir.as_posix())
                print(f"[CKPT] Saved -> {save_dir}", flush=True)

        last_dir = out_dir / "last"
        ensure_dir(last_dir)
        student.save_pretrained(last_dir.as_posix())
        ddpm.save_pretrained(last_dir.as_posix())
        print(f"[Epoch {epoch}] saved last -> {last_dir}", flush=True)

    if feat_col_T is not None:
        feat_col_T.close()
    if feat_col_S is not None:
        feat_col_S.close()

    if wandb_run is not None and wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass


# ------------------------- Args -------------------------

BATCH_SIZE = 8
CUDA_NUM = 4
LR = 1e-5
DATE = "0107"
RKD_W0 = 0.0
INV_W0 = 0.0
INVINV_W0 = 0.0
FD_W0 = 0.0
SAME_W0 = 0.0

RKD_W = 1.0
INV_W = 1.0
INVINV_W = 1.0
FD_W = 0.00001
SAME_W = 0.0

def build_argparser():
    p = argparse.ArgumentParser("Distill with x0 + diffusion intermediate features (flattened for losses)")

    p.add_argument("--resume_checkpoint", type=str, default="")

    p.add_argument("--student_data_dir", type=str, default="cifar10_student_data_n10/gray3/train")
    p.add_argument("--test_dir", type=str, default="cifar10_png_linear_only/gray3/test")
    p.add_argument("--teacher_dir", type=str, default="ddpm_cifar10_rgb_T400_DDIM50/ckpt_step150000")
    p.add_argument("--output_dir", type=str, default=f"out_{DATE}_rkd/x0+diff_flatten_bs{BATCH_SIZE}_lr{LR}-RKD{RKD_W0}_{RKD_W}-INV{INV_W0}_{INV_W}-INVINV{INVINV_W0}_{INVINV_W}-FD{FD_W0}_{FD_W}-SAME{SAME_W0}_{SAME_W}")

    p.add_argument("--device", type=str, default=f"cuda:{CUDA_NUM}")
    p.add_argument("--project", type=str, default=f"x0-diff-flatten-rkd-{DATE}")
    p.add_argument("--run_name", type=str, default=f"student-lora-x0-diffflatten-bs{BATCH_SIZE}-lr{LR}-RKD{RKD_W0}_{RKD_W}-INV{INV_W0}_{INV_W}-INVINV{INVINV_W0}_{INVINV_W}-FD{FD_W0}_{FD_W}-SAME{SAME_W0}_{SAME_W}")
    p.add_argument("--wandb_offline", action="store_true")
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--center_crop", action="store_true")
    p.add_argument("--no_hflip", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--real_batch", type=int, default=BATCH_SIZE)
    p.add_argument("--noise_batch", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--train_timesteps", type=int, default=400)
    p.add_argument("--beta_schedule", type=str, default="linear")

    p.add_argument("--ddim_steps_min", type=int, default=40)
    p.add_argument("--ddim_steps_max", type=int, default=60)
    p.add_argument("--ddim_eta", type=float, default=0.0)

    # diffusion feature hooks
    p.add_argument("--use_diff_feats", action="store_false",
                   help="Enable diffusion intermediate features (hook). If off, diffusion losses ignored.")
    p.add_argument("--diff_specs", type=str, nargs="+", default=["mid"],
                   help="Hook specs: mid | down.i | up.i | down_last | up_last")
    p.add_argument("--diff_cast", type=str, default="fp16", choices=["none", "fp16", "fp32"],
                   help="Optional cast for stored activations (NO detach). fp16 reduces VRAM.")
    p.add_argument("--diff_store_stride", type=int, default=1,
                   help="Store diffusion features every N-th timestep (append stride).")

    # RKD
    p.add_argument("--w_rkd_x0", type=float, default=RKD_W0)
    p.add_argument("--w_rkd_diff", type=float, default=RKD_W)
    p.add_argument("--rkd_x0_stride", type=int, default=1)
    p.add_argument("--rkd_diff_stride", type=int, default=1)
    p.add_argument("--rkd_x0_teacher_ref", type=str, default="last", choices=["last", "matched"])
    p.add_argument("--rkd_diff_teacher_ref", type=str, default="matched", choices=["last", "matched"])

    # INV / INVINV / FD weights (x0 vs diff)
    p.add_argument("--w_inv_x0", type=float, default=INV_W0)
    p.add_argument("--w_inv_diff", type=float, default=INV_W)
    p.add_argument("--w_invinv_x0", type=float, default=INVINV_W0)
    p.add_argument("--w_invinv_diff", type=float, default=INVINV_W)
    p.add_argument("--w_fd_x0", type=float, default=FD_W0)
    p.add_argument("--w_fd_diff", type=float, default=FD_W)
    p.add_argument("--fd_eps", type=float, default=1e-8)

    # SAME
    p.add_argument("--w_same", type=float, default=SAME_W)
    p.add_argument("--same_mode", type=str, default="mean", choices=["mean", "last"])

    # logging / eval
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--sample_interval", type=int, default=1000)
    p.add_argument("--sample_n", type=int, default=64)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--sample_eta", type=float, default=0.0)

    p.add_argument("--disable_fid", action="store_true")
    p.add_argument("--fid_batch_size", type=int, default=64)
    p.add_argument("--fid_gen_batch", type=int, default=256)
    p.add_argument("--fid_dims", type=int, default=2048)
    p.add_argument("--fid_keep_gen", action="store_true")
    p.add_argument("--fid_num_samples", type=int, default=0)
    p.add_argument("--fid_no_symlink", action="store_true")

    # LoRA
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_init", type=str, default="gaussian")
    p.add_argument("--lora_target_modules", type=str, nargs="+",
                   default=["to_q", "to_k", "to_v", "to_out.0"])

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    ensure_dir(Path(args.output_dir))
    train(args)
