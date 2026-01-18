#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Student (x0 predictor) distillation with Feature-based losses (Pixel, CLIP, DINO):
  - RKD (pdist) in Feature space (Pixel or Perceptual)
  - INV (cdist) in Feature space
  - INVINV (pdist) in Feature space
  - FID loss (Gaussian) in Feature space (if perceptual metric is chosen, compute Gaussian on features)

Dependencies:
  - pip install diffusers transformers torch torchvision pytorch-fid peft
"""

import os
import math
import argparse
import shutil
from pathlib import Path
from typing import List, Optional
from contextlib import nullcontext

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.utils as vutils
import torchvision.models as models
from PIL import Image

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DDIMInverseScheduler
from peft import LoraConfig, get_peft_model, PeftModel  # PeftModel 추가됨

# HuggingFace
from transformers import AutoModel, AutoImageProcessor, CLIPModel


from collections import OrderedDict
from safetensors.torch import load_file as safe_load_file
from safetensors import safe_open


# ------------------------- Feature Extraction Utils -------------------------

class FeatureEmbedder(nn.Module):
    """
    Extracts features from images for RKD/Distance computations.
    Modes:
      - 'pixel'    : Flatten raw pixels
      - 'clip'     : CLIP vision embeddings, resize 224
      - 'dinov3'   : DINOv3 embeddings (HF AutoModel), resize based on image processor (typically 224)
    """
    def __init__(
        self,
        mode: str,
        device: torch.device,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        dino_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    ):
        super().__init__()
        self.mode = mode
        self.device = device
        self.feature_dim = -1

        if mode == "pixel":
            self.net = None
            return

        if mode == "clip":
            print(f"[Embedder] Loading CLIP ({clip_model_name})...", flush=True)
            self.net = CLIPModel.from_pretrained(clip_model_name).vision_model.to(device)
            for p in self.net.parameters():
                p.requires_grad = False
            self.net.eval()
            self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
            self.std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
            self.target_size = (224, 224)
            return

        if mode == "dinov3":
            print(f"[Embedder] Loading DINOv3 ({dino_model_name})...", flush=True)
            # Processor에서 size/mean/std를 가져오면 모델별 전처리와 최대한 일치
            try:
                proc = AutoImageProcessor.from_pretrained(dino_model_name)
            except Exception as e:
                print(f"[Warn] AutoImageProcessor load failed ({e}). Falling back to ImageNet mean/std and 224.", flush=True)
                proc = None

            self.net = AutoModel.from_pretrained(dino_model_name).to(device)
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

            # size 처리(모델/processor별로 dict/int 형태가 달라질 수 있어 방어적으로)
            target = 224
            if proc is not None and hasattr(proc, "size"):
                sz = proc.size
                if isinstance(sz, dict):
                    # 일반적으로 {"height":224,"width":224} 또는 {"shortest_edge":224}
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

        raise ValueError(f"Unknown metric mode: {mode}. Use one of ['pixel','clip','dinov3'].")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (N, 3, H, W) in range [-1, 1]
        Output: (N, D) feature vector
        """
        if self.mode == "pixel":
            return x.reshape(x.shape[0], -1)

        # # [-1,1] -> [0,1]
        # x_01 = (x + 1) * 0.5

        # bank 생성과 동일하게 clamp (pixel은 굳이 필요 없지만 해도 무방)
        x = x.clamp(-1, 1)
        # [-1,1] -> [0,1]
        x_01 = (x + 1) * 0.5

        # resize + normalize
        x_up = F.interpolate(x_01, size=self.target_size, mode="bilinear", align_corners=False, antialias=True)
        x_norm = (x_up - self.mean) / self.std

        if self.mode == "clip":
            outputs = self.net(pixel_values=x_norm)
            return outputs.pooler_output  # (N, D)

        if self.mode == "dinov3":
            outputs = self.net(pixel_values=x_norm)
            # 모델에 따라 pooler_output이 있거나 없을 수 있음
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            # 없으면 CLS 토큰 사용
            return outputs.last_hidden_state[:, 0, :]

        # fallback
        return x.reshape(x.shape[0], -1)


# ------------------------- Utils -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())

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

def pdist_vec(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.pdist(x, p=2).clamp_min(eps)

def cdist_vec(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.cdist(x, y, p=2).reshape(-1).clamp_min(eps)

def l2norm_feat(f: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # f: (N, D)
    return f / f.norm(dim=1, keepdim=True).clamp_min(eps)

def set_seed(seed: int):
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def resolve_device(device_str: str) -> torch.device:
    try:
        dev = torch.device(device_str)
    except Exception:
        print(f"[Warn] Invalid --device '{device_str}'. Falling back to 'cpu'.", flush=True)
        return torch.device("cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev

def make_abl_suffix(args) -> str:
    # 항상 같은 순서로 붙여야 비교가 쉬움
    tags = []
    if getattr(args, "rkd_no_mean_norm", False):
        tags.append("noMean")
    if getattr(args, "feat_l2norm", False):
        tags.append("featL2")
    if getattr(args, "mean_detach", False):
        tags.append("meanDetach")
    if getattr(args, "teacher_match_k", False):
        tags.append("TmatchK")

    # 아무 것도 없으면 base로 표기
    if len(tags) == 0:
        return "ABLbase"
    return "ABL-" + "-".join(tags)

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


# ------------------------- Gaussian FID (Feature or Pixel) -------------------------

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


# ------------------------- Dataset (student data) -------------------------

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


# ------------------------- Offline Teacher Bank (z + teacher_feat) -------------------------

class OfflineTeacherBankBatcher:
    """
    Safetensors bank loader (metric-wise):
      bank_root/
        clip/shards/shard_*.safetensors
        dinov3/shards/shard_*.safetensors
        pixel/shards/shard_*.safetensors

    Each shard contains:
      - "z": (N,3,H,W)
      - feats_{metric}: (N,D)
      - "steps": (N,) int16   (mixed inside shard)
    We return batches where all samples share the SAME steps (scalar),
    so downstream code can keep using one DDIM steps per batch.
    """

    METRIC2KEY = {
        "pixel":  "feats_pixel",
        "clip":   "feats_clip",
        "dinov3": "feats_dinov3",
    }

    def __init__(
        self,
        bank_root: str,
        metric: str,
        seed: int = 0,
        steps_min: int = 40,
        steps_max: int = 60,
        cache_shards: int = 2,
        index_max_unique_steps: int = 256,  # safety
    ):
        self.root = Path(bank_root)
        if not self.root.exists():
            raise FileNotFoundError(f"teacher_bank_dir not found: {self.root}")

        if metric not in self.METRIC2KEY:
            raise ValueError(f"Unsupported metric='{metric}'. Use one of {list(self.METRIC2KEY.keys())}")

        self.metric = metric
        self.feat_key = self.METRIC2KEY[metric]

        # metric-wise shards path
        self.shard_dir = self.root / metric / "shards"
        if not self.shard_dir.exists():
            raise FileNotFoundError(
                f"Expected safetensors bank layout: {self.root}/{metric}/shards/*.safetensors "
                f"(not found: {self.shard_dir})"
            )

        self.shards = sorted(self.shard_dir.glob("shard_*.safetensors"))
        if len(self.shards) == 0:
            raise FileNotFoundError(f"No shard_*.safetensors found under: {self.shard_dir}")

        self.steps_min = int(steps_min)
        self.steps_max = int(steps_max)
        if self.steps_min <= 0 or self.steps_max < self.steps_min:
            raise ValueError(f"Invalid steps range: [{self.steps_min},{self.steps_max}]")

        self.rng = random.Random(int(seed))
        self.torch_rng = torch.Generator(device="cpu")
        self.torch_rng.manual_seed(int(seed))

        # LRU cache: path -> dict of tensors (CPU)
        self.cache = OrderedDict()
        self.cache_shards = int(cache_shards)
        if self.cache_shards <= 0:
            self.cache_shards = 1

        # Build step -> list of shards that contain that step (cheap, reads only 'steps')
        self.step2shards = {s: [] for s in range(self.steps_min, self.steps_max + 1)}
        for p in self.shards:
            try:
                with safe_open(str(p), framework="pt", device="cpu") as f:
                    if "steps" not in f.keys():
                        continue
                    steps_vec = f.get_tensor("steps")  # int16 (N,)
            except Exception:
                continue

            # unique steps inside shard (cap for safety)
            uniq = torch.unique(steps_vec)
            if uniq.numel() > index_max_unique_steps:
                # extremely unlikely; still safe
                uniq = uniq[:index_max_unique_steps]

            for s in uniq.tolist():
                s = int(s)
                if self.steps_min <= s <= self.steps_max:
                    self.step2shards[s].append(p)

        # Ensure at least one step is available
        avail = [s for s, ps in self.step2shards.items() if len(ps) > 0]
        if len(avail) == 0:
            raise RuntimeError(
                f"No shards contain steps in range [{self.steps_min},{self.steps_max}]. "
                f"Check bank generation / repack."
            )

        self.available_steps = avail

    def _get_shard(self, p: Path) -> dict:
        """Load shard tensors to CPU with LRU caching."""
        key = str(p)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

        d = safe_load_file(str(p), device="cpu")  # loads all tensors in this shard
        # required keys check
        if "z" not in d:
            raise KeyError(f"Shard missing 'z': {p}")
        if "steps" not in d:
            raise KeyError(f"Shard missing 'steps': {p}")
        if self.feat_key not in d:
            raise KeyError(f"Shard missing '{self.feat_key}': {p}")

        # Ensure contiguous for faster index_select
        d["z"] = d["z"].contiguous()
        d[self.feat_key] = d[self.feat_key].contiguous()
        d["steps"] = d["steps"].contiguous()

        self.cache[key] = d
        self.cache.move_to_end(key)

        # evict old
        while len(self.cache) > self.cache_shards:
            self.cache.popitem(last=False)

        return d

    def next(self, batch_size: int, device: torch.device, z_dtype: torch.dtype):
        """
        Returns:
          z_gpu: (B,3,H,W) on GPU dtype=z_dtype
          teacher_feat_gpu: (B,D) on GPU (original dtype from file; usually fp32)
          steps: int (same for entire batch)
        """
        B = int(batch_size)
        if B <= 0:
            raise ValueError("batch_size must be positive")

        # pick a target steps uniformly over available
        steps = int(self.available_steps[self.rng.randrange(len(self.available_steps))])

        # try a few shards that contain this steps
        shard_list = self.step2shards.get(steps, [])
        if len(shard_list) == 0:
            # fallback: pick any available step again (shouldn't happen)
            steps = int(self.available_steps[self.rng.randrange(len(self.available_steps))])
            shard_list = self.step2shards[steps]

        # Attempt multiple times until we find enough samples in that shard for this step
        for _ in range(32):
            p = shard_list[self.rng.randrange(len(shard_list))]
            d = self._get_shard(p)

            steps_vec = d["steps"]  # (N,)
            idx_all = (steps_vec == steps).nonzero(as_tuple=False).squeeze(1)
            n = int(idx_all.numel())
            if n < B:
                continue

            # sample without replacement within idx_all
            perm = torch.randperm(n, generator=self.torch_rng)[:B]
            idx = idx_all.index_select(0, perm)

            z_cpu = d["z"].index_select(0, idx)
            f_cpu = d[self.feat_key].index_select(0, idx)

            z_gpu = z_cpu.to(device, non_blocking=True, dtype=z_dtype)
            f_gpu = f_cpu.to(device, non_blocking=True)  # keep fp32 by default
            return z_gpu, f_gpu, int(steps)

        raise RuntimeError(f"Failed to sample a batch for steps={steps} after many tries. Try larger shards or cache.")



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


# ------------------------- Sampling / Inversion -------------------------

# @torch.no_grad()
def teacher_predx0_seq(teacher, ddim_T, z, steps, eta, device) -> torch.Tensor:
    local = DDIMScheduler.from_config(ddim_T.config)
    local.set_timesteps(steps, device=device)
    x = z.to(device)

    teacher.eval()
    for t in local.timesteps:
        x_in = local.scale_model_input(x, t)
        eps  = teacher(x_in, t).sample
        out  = local.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x    = out.prev_sample
    return x



def student_predx0_seq_with_grad(student, ddim_S, z, steps, eta, device, rkd_step_k: int) -> torch.Tensor:
    local = DDIMScheduler.from_config(ddim_S.config)
    local.set_timesteps(steps, device=device)
    x = z.to(device)
    student.train()

    k_step = 0

    for t in local.timesteps:
        x_in = local.scale_model_input(x, t)
        eps   = student(x_in, t).sample
        out  = local.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x    = out.prev_sample

        if k_step >= rkd_step_k:
            return out.pred_original_sample
        k_step += 1

    return out.pred_original_sample


def invert_x0_to_zT_deterministic_x0pred(student, ddim_S, x0, steps, device) -> torch.Tensor:
    inv = DDIMInverseScheduler.from_config(ddim_S.config)
    inv.set_timesteps(steps, device=device)
    # student.eval()
    xt = x0
    for t in inv.timesteps:
        # t_b = torch.full((xt.shape[0],), int(t), device=device, dtype=torch.long)
        latent_in = inv.scale_model_input(xt, t)
        eps = student(latent_in, t).sample
        xt = inv.step(eps, t, xt).prev_sample
    return xt

@torch.no_grad()
def sample_images_ddim_x0pred(
    student: UNet2DModel,
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


# ------------------------- Losses -------------------------

def compute_losses(
    teacher_feat: torch.Tensor,   # (B,D) offline bank feature == T_f
    preds_S: torch.Tensor,
    x0_real: torch.Tensor,
    x0_inv_T: torch.Tensor,
    embedder: FeatureEmbedder,
    args,
):
    eps = 1e-12
    device = x0_real.device

    # ---- feature extraction ----
    with torch.no_grad():
        R_f = embedder(x0_real)

    I_f = embedder(x0_inv_T)
    S_f = embedder(preds_S)

    target_dtype = S_f.dtype
    T_f = teacher_feat.to(dtype=target_dtype)
    R_f = R_f.to(dtype=target_dtype)
    I_f = I_f.to(dtype=target_dtype)



    # ---- RKD / INV / INVINV vectors ----
    loss_rkd = torch.tensor(0.0, device=device)
    loss_inv = torch.tensor(0.0, device=device)
    loss_invinv = torch.tensor(0.0, device=device)

    rkd_s_list = rkd_t_list = None
    # RKD
    if args.w_rkd != 0.0:
        with torch.no_grad():
            rkd_t_list = pdist_vec(T_f, eps=eps)
        rkd_s_list = pdist_vec(S_f, eps=eps)

    # INV
    inv_s = inv_t = None
    if args.w_inv != 0.0:
        inv_s = cdist_vec(S_f, R_f, eps=eps)
        inv_t = cdist_vec(T_f, I_f, eps=eps)

    # INVINV
    invinv_s = invinv_t = None
    if args.w_invinv != 0.0:
        invinv_s = pdist_vec(R_f, eps=eps)
        invinv_t = pdist_vec(I_f, eps=eps)


    # mean normalization (original behavior)
    student_parts, teacher_parts = [], []

    if args.w_rkd != 0.0:
        student_parts.append(rkd_s_list)
        teacher_parts.append(rkd_t_list)
    if args.w_inv != 0.0 and inv_s is not None:
        student_parts.append(inv_s)
        teacher_parts.append(inv_t)
    if args.w_invinv != 0.0 and invinv_s is not None:
        student_parts.append(invinv_s)
        teacher_parts.append(invinv_t)

    if len(student_parts) > 0:
        student_mean = mean_from_vectors(student_parts, device=device, eps=eps)
        teacher_mean = mean_from_vectors(teacher_parts, device=device, eps=eps)
    else:
        student_mean = torch.tensor(1.0, device=device)
        teacher_mean = torch.tensor(1.0, device=device)

    # apply normalization + MSE
    if args.w_rkd != 0.0:
        loss_rkd = F.mse_loss(rkd_s_list / student_mean, rkd_t_list / teacher_mean, reduction="mean")
    if args.w_inv != 0.0 and inv_s is not None:
        loss_inv = F.mse_loss(inv_s / student_mean, inv_t / teacher_mean, reduction="mean")
    if args.w_invinv != 0.0 and invinv_s is not None:
        loss_invinv = F.mse_loss(invinv_s / student_mean, invinv_t / teacher_mean, reduction="mean")

    # ---- FD (diagonal) ----
    loss_fid = torch.tensor(0.0, device=device)
    fid_s = torch.tensor(0.0, device=device)
    fid_t = torch.tensor(0.0, device=device)
    if args.w_fid != 0.0:
        fid_s = frechet_distance_diag(S_f.float(), R_f.float(), eps=args.fid_eps)
        fid_t = frechet_distance_diag(T_f.float(), I_f.float(), eps=args.fid_eps)
        loss_fid = fid_s + fid_t

    total = args.w_rkd * loss_rkd + args.w_inv * loss_inv + args.w_invinv * loss_invinv + args.w_fid * loss_fid

    stats = {
        "loss_rkd": loss_rkd.detach(),
        "loss_inv": loss_inv.detach(),
        "loss_invinv": loss_invinv.detach(),
        "loss_fid": loss_fid.detach(),
        "fid_s": fid_s.detach(),
        "fid_t": fid_t.detach(),
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

    rkd_raw     = float(stats["loss_rkd"].item())
    inv_raw     = float(stats["loss_inv"].item())
    invinv_raw = float(stats["loss_invinv"].item())
    fid_raw     = float(stats["loss_fid"].item())
    fid_s_raw  = float(stats["fid_s"].item())
    fid_t_raw  = float(stats["fid_t"].item())
    total      = float(total_loss.detach().item())

    rkd_w,    rkd_raw2    = w_and_raw(rkd_raw,    args.w_rkd)
    inv_w,    inv_raw2    = w_and_raw(inv_raw,    args.w_inv)
    invinv_w, invinv_raw2 = w_and_raw(invinv_raw, args.w_invinv)
    fid_w,    fid_raw2    = w_and_raw(fid_raw,    args.w_fid)
    fid_s_w,  fid_s_raw2  = w_and_raw(fid_s_raw,  args.w_fid)
    fid_t_w,  fid_t_raw2  = w_and_raw(fid_t_raw,  args.w_fid)

    logs = {
        "loss/total": total,
        "loss/rkd": rkd_w,
        "loss/inv": inv_w,
        "loss/invinv": invinv_w,
        "loss/fid": fid_w,
        "loss/fid_s": fid_s_w,
        "loss/fid_t": fid_t_w,
        "loss_raw/rkd": rkd_raw2,
        "loss_raw/inv": inv_raw2,
        "loss_raw/invinv": invinv_raw2,
        "loss_raw/fid": fid_raw2,
    }
    return logs


# ------------------------- Eval: standard FID via pytorch-fid -------------------------

def compute_fid_pytorch_fid(real_dir: Path, gen_dir: Path, device: torch.device, batch_size: int, dims: int) -> float:
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except Exception as e:
        raise RuntimeError(f"pytorch-fid not available: {e}")

    fid = calculate_fid_given_paths(
        [real_dir.as_posix(), gen_dir.as_posix()],
        batch_size=batch_size,
        device=device,
        dims=dims,
    )
    return float(fid)

@torch.no_grad()
def eval_sample_and_fid(
    student: UNet2DModel,
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
    fid_real_per_class_root: Optional[Path],
    fid_class_names: List[str],
    num_test_imgs_all: int,
):
    ensure_dir(out_dir / "samples")
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + int(global_step))

    imgs = sample_images_ddim_x0pred(
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
        xs = sample_images_ddim_x0pred(
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

def train(args):
    torch.backends.cudnn.benchmark = True
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    set_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index)
        
    # Init Feature Embedder (for RKD/Losses)
    print(f"[Info] Initializing RKD Feature Embedder: {args.rkd_metric}", flush=True)
    embedder = FeatureEmbedder(
        mode=args.rkd_metric,
        device=device,
        clip_model_name=args.clip_model_name,
        dino_model_name=args.dino_model_name,
    )

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
        print(f"[Warn] wandb init failed or not installed. Proceeding without wandb. ({e})", flush=True)

    use_amp = (device.type == "cuda") and (args.mixed_precision != "no")
    amp_dtype = None
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    if use_amp:
        if args.mixed_precision == "fp16":
            amp_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler(enabled=True)
        elif args.mixed_precision == "bf16":
            amp_dtype = torch.bfloat16
            scaler = None
        else:
            use_amp = False



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
    print(f"[Info] Student data images: {len(dataset)} under {args.student_data_dir}", flush=True)
    print(f"[Info] Device: {device} | AMP: {args.mixed_precision if use_amp else 'no'}", flush=True)

    print(f"[Info] Loading offline teacher bank: {args.teacher_bank_dir}", flush=True)
    bank = OfflineTeacherBankBatcher(
        bank_root=args.teacher_bank_dir,
        metric=args.rkd_metric,
        seed=args.seed,
    )


    # ---- FID real cache (test) ----
    fid_real_all_dir = None
    fid_real_per_class_root = None
    fid_class_names: List[str] = []
    num_test_imgs_all = 0

    if (not args.disable_fid) and args.test_dir and len(args.test_dir.strip()) > 0:
        test_dir = Path(args.test_dir)
        if test_dir.exists():
            fid_real_root = out_dir / "fid" / "real_cache"
            fid_real_all_dir = fid_real_root / "all"
            num_test_imgs_all = flatten_real_cache(test_dir, fid_real_all_dir, use_symlink=not args.fid_no_symlink)
            print(f"[FID] Real cache ready: all={num_test_imgs_all} imgs @ {fid_real_all_dir}", flush=True)
        else:
            args.disable_fid = True
    else:
        args.disable_fid = True

    # ---- Teacher ----
    teacher_dir = Path(args.teacher_dir)
    teacher = UNet2DModel.from_pretrained(teacher_dir.as_posix()).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ---- Schedulers ----
    ddpm = load_teacher_scheduler_or_fallback(teacher_dir, args.train_timesteps, args.beta_schedule)
    ddim_T = make_ddim(ddpm, prediction_type="epsilon")
    ddim_S = make_ddim(ddpm, prediction_type="epsilon")

    # ---- Student Initialization (Modified for Resume) ----
    print(f"[Info] Initializing Student...", flush=True)
    # 1. Base Student를 Teacher 경로에서 로드 (Teacher와 동일한 아키텍처/가중치로 시작)
    student = UNet2DModel.from_pretrained(teacher_dir.as_posix()).to(device)
    student.requires_grad_(False)
    
    global_step = 0

    # 2. 체크포인트 재개 여부 확인
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"[Info] Resuming from checkpoint: {args.resume_checkpoint}", flush=True)
        # 기존 저장된 LoRA 어댑터 로드
        student = PeftModel.from_pretrained(student, args.resume_checkpoint, is_trainable=True)
        
        # Step 파싱 (폴더 이름에서 step 숫자 추출, 예: ckpt_step002000)
        try:
            ckpt_name = Path(args.resume_checkpoint).name
            if "step" in ckpt_name:
                global_step = int(ckpt_name.split("step")[-1])
                print(f"[Info] Resuming at global_step = {global_step}", flush=True)
            else:
                print(f"[Warn] Could not parse global_step from '{ckpt_name}'. Starting from 0.")
        except Exception as e:
            print(f"[Warn] Error parsing global_step: {e}. Starting from 0.")
            
    else:
        # 3. 새로 시작 (New LoRA Init)
        print(f"[Info] Starting new training (initializing LoRA)...", flush=True)
        lora_config = LoraConfig(
            r=32,               # LoRA Rank
            lora_alpha=32,      # Alpha
            target_modules=["to_q", "to_k", "to_v", "to_out.0"], 
            init_lora_weights="gaussian",
        )
        student = get_peft_model(student, lora_config)
    
    # 4. Train 모드 전환 및 확인
    student.print_trainable_parameters()
    student.train()

    print(f"[Info] Teacher params: {count_parameters(teacher):,}", flush=True)
    print(f"[Info] Student params: {count_parameters(student):,}", flush=True)

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    optimizer.zero_grad(set_to_none=True)

    # global_step은 위에서 설정됨 (0 또는 resume value)
    start_epoch = 1
    # 간단한 근사: epoch도 step에 비례하여 대략적으로 맞춤 (optional)
    # start_epoch = global_step // len(loader) + 1 

    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        print(f"[Epoch {epoch}] start (Global Step: {global_step})", flush=True)

        for it, x0_real in enumerate(loader, start=1):

            # rkd_step_k = ddim_steps_train - 1
            x0_real = x0_real.to(device, non_blocking=True)

            # ---- offline bank gives z + teacher_feat + steps ----
            z_dtype = next(student.parameters()).dtype
            z, teacher_feat, ddim_steps_train = bank.next(
                batch_size=args.noise_batch,
                device=device,
                z_dtype=z_dtype,
            )

            rkd_step_k = random.randrange(ddim_steps_train)

            if use_amp:
                autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
            else:
                autocast_ctx = nullcontext()

            with autocast_ctx:
                preds_S = student_predx0_seq_with_grad(
                    student, ddim_S, z, steps=ddim_steps_train, eta=args.ddim_eta, device=device, rkd_step_k=rkd_step_k
                )
                zT_real = invert_x0_to_zT_deterministic_x0pred(
                    student, ddim_S, x0_real, steps=ddim_steps_train, device=device
                )
                x0_inv_T = teacher_predx0_seq(
                    teacher, ddim_T, zT_real, steps=ddim_steps_train, eta=args.ddim_eta, device=device
                )

            total_loss, stats = compute_losses(
                teacher_feat=teacher_feat,
                preds_S=preds_S,
                x0_real=x0_real,
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
                m = {
                    "total": float(total_loss.detach().item()),
                    "rkd": float(stats["loss_rkd"].item()),
                    "inv": float(stats["loss_inv"].item()),
                    "invinv": float(stats["loss_invinv"].item()),
                    "fid": float(stats["loss_fid"].item()),
                }

                line = (
                    f"[Epoch {epoch:03d}] step={global_step:08d} "
                    f"total={m['total']:.6f} rkd={m['rkd']:.6f} inv={m['inv']:.6f} "
                    f"invinv={m['invinv']:.6f} fid={m['fid']:.6f}"
                )
                print(line, flush=True)
                with summary_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                
                if wandb_run is not None and wandb is not None:
                    try:
                        loss_logs = build_loss_logs(total_loss, stats, args)
                        wandb.log({
                            **loss_logs,
                            "train/epoch": int(epoch),
                            "train/step": int(global_step),
                            "train/lr": float(args.lr),
                        }, step=global_step)
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
                    fid_real_per_class_root=fid_real_per_class_root,
                    fid_class_names=fid_class_names,
                    num_test_imgs_all=num_test_imgs_all,
                )

            if args.save_interval > 0 and (global_step % args.save_interval == 0):
                save_dir = out_dir / "ckpts" / f"ckpt_step{global_step:06d}"
                ensure_dir(save_dir)
                student.save_pretrained(save_dir.as_posix())
                ddpm.save_pretrained(save_dir.as_posix())
                print(f"[CKPT] Saved student to {save_dir}", flush=True)

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

BATCH_SIZE = 8
CLASSN = 10
RKD_METRIC="clip" # pixel clip dinov3
CUDA_NUM = 7
LR=1e-5
DATE="0115"
BANK_DIR = "/workspace/rkd_cifar10_1111/0116_teacher_bank_sft_only_steps40to60_random"

RKD_W = 1.0
INV_W = 1.0
INVINV_W = 0.01
FD_W = 1.0

def build_argparser():
    p = argparse.ArgumentParser("Student x0 distillation with Feature-based losses")

    p.add_argument("--resume_checkpoint", type=str, default="", help="Path to checkpoint folder to resume training from (e.g. out/.../ckpt_step002000)")

    p.add_argument("--student_data_dir", type=str, default="cifar10_student_data_n10/gray3/train")
    p.add_argument("--test_dir", type=str, default="cifar10_png_linear_only/gray3/test")
    p.add_argument("--teacher_dir", type=str, default="ddpm_cifar10_rgb_T400_DDIM50/ckpt_step150000")
    p.add_argument("--output_dir", type=str, default=f"out_{DATE}_rkd/rkd_{RKD_METRIC}_lora_feature_cifar10_rgb_to_gray_single_batch{BATCH_SIZE}_N{CLASSN}_LR{LR}-EASY_FD-rkdW{RKD_W}-invW{INV_W}-invinvW{INVINV_W}-fdW{FD_W}-teacher-init-eps-bank")
    p.add_argument("--run_name", type=str, default=f"student-lora-{RKD_METRIC}-x0-rgb-to-gray-batch{BATCH_SIZE}-N{CLASSN}-LR{LR}-FD-rkdW{RKD_W}-invW{INV_W}-invinvW{INVINV_W}-fdW{FD_W}-teacher-init-eps-EASY-FD-bank")

    # Metric Selection for RKD/INV
    p.add_argument(
        "--rkd_metric",
        type=str,
        default=RKD_METRIC,
        choices=["pixel", "clip", "dinov3"],
        help="Metric space for RKD/INV losses: 'pixel', 'clip', 'dinov3'.",
    )

    p.add_argument(
        "--dino_model_name",
        type=str,
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="HuggingFace model name for DINOv3 (requires transformers>=4.56.0).",
    )

    p.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HuggingFace model name for CLIP if rkd_metric='clip'",
    )

    p.add_argument("--device", type=str, default=f"cuda:{CUDA_NUM}")
    p.add_argument("--project", type=str, default=f"rkd-feature-cifar10-rgb-to-gray-{DATE}")
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

    p.add_argument("--ddim_eta", type=float, default=0.0)

    p.add_argument("--w_rkd", type=float, default=RKD_W)
    p.add_argument("--w_inv", type=float, default=INV_W)
    p.add_argument("--w_invinv", type=float, default=INVINV_W)
    p.add_argument("--w_fid", type=float, default=FD_W)

    p.add_argument("--fid_eps", type=float, default=1e-8)

    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=2000)
    p.add_argument("--sample_interval", type=int, default=2000)
    p.add_argument("--sample_n", type=int, default=36)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--sample_eta", type=float, default=0.0)

    p.add_argument("--disable_fid", action="store_true")
    p.add_argument("--fid_batch_size", type=int, default=64)
    p.add_argument("--fid_gen_batch", type=int, default=256)
    p.add_argument("--fid_dims", type=int, default=2048)
    p.add_argument("--fid_keep_gen", action="store_true")
    p.add_argument("--fid_num_samples", type=int, default=0)
    p.add_argument("--fid_no_symlink", action="store_true")


    p.add_argument("--teacher_bank_dir", type=str, default=BANK_DIR,
                help="Offline teacher bank root (created by make_teacher_z_x0_and_feats_0116.py).")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    ensure_dir(Path(args.output_dir))
    train(args)

