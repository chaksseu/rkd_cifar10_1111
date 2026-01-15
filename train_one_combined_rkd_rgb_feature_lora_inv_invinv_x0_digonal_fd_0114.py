#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stable Diffusion 1.5 Teacher/Student (LoRA) distillation
- Feature losses computed on decoded RGB images
- Inversion: GT image -> VAE latent -> DDIMInverse (UNet + text cond) -> noise
- Dataset: (image, text). Text prompt = parent folder name (CIFAR10 class name)

Deps:
  pip install -U diffusers transformers peft torch torchvision pytorch-fid
"""

import os
import math
import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Tuple
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

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDIMInverseScheduler
from peft import LoraConfig, get_peft_model, PeftModel

from transformers import AutoModel, AutoImageProcessor, CLIPModel
from transformers import CLIPTokenizer, CLIPTextModel


# ------------------------- Feature Extraction Utils -------------------------

class FeatureEmbedder(nn.Module):
    """
    Extract features from images. Input: (N,3,H,W) in [-1,1].
    Modes: pixel / inception / clip / dinov3
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

        raise ValueError(f"Unknown metric mode: {mode}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "pixel":
            return x.reshape(x.shape[0], -1)

        x_01 = (x + 1) * 0.5
        x_up = F.interpolate(x_01, size=self.target_size, mode="bilinear", align_corners=False, antialias=True)
        x_norm = (x_up - self.mean) / self.std

        if self.mode == "inception":
            return self.net(x_norm)

        if self.mode == "clip":
            outputs = self.net(pixel_values=x_norm)
            return outputs.pooler_output

        if self.mode == "dinov3":
            outputs = self.net(pixel_values=x_norm)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            return outputs.last_hidden_state[:, 0, :]

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


# ------------------------- Dataset (image, text) -------------------------

class ImageTextFolderDataset(Dataset):
    """
    root/
      class_name_0/*.png
      class_name_1/*.png
      ...

    Returns:
      image: (3,H,W) in [-1,1]
      text:  str (class_name)
    """
    def __init__(self, root: str, image_size: int = 512, center_crop: bool = True, horizontal_flip: bool = True):
        self.root = Path(root)
        exts = {".png", ".jpg", ".jpeg"}
        self.files: List[Path] = []
        for p in self.root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                self.files.append(p)
        if len(self.files) == 0:
            raise FileNotFoundError(f"No images found under {self.root}!")

        # class names (immediate parent directory name)
        self.class_names = sorted({p.parent.name for p in self.files})

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
        prompt = path.parent.name  # folder name
        with Image.open(path) as img:
            img = img.convert("RGB")
            x01 = self.to_tensor(img)
        x = x01 * 2.0 - 1.0
        return x, prompt

def collate_image_text(batch: List[Tuple[torch.Tensor, str]]):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    texts = [b[1] for b in batch]
    return imgs, texts


# ------------------------- SD helpers: dtype / vae encode-decode / text embeddings -------------------------

def get_model_dtype(args, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if args.mixed_precision == "fp16":
        return torch.float16
    if args.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32

# @torch.no_grad()
def encode_images_to_latents(vae: AutoencoderKL, images: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    # VAE 파라미터 dtype/device에 맞추기 (fp16 로드면 images도 fp16로)
    vae_param = next(vae.parameters())
    images = images.to(device=vae_param.device, dtype=vae_param.dtype)

    # latents = vae.encode(images).latent_dist.sample()
    latents = vae.encode(images).latent_dist.mean
    return latents * scaling_factor


def decode_latents_to_images(vae: AutoencoderKL, latents: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    vae_param = next(vae.parameters())
    latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)

    imgs = vae.decode(latents / scaling_factor).sample
    return imgs


class TextCondCache:
    """
    Cache text encoder outputs per unique prompt string.
    CIFAR10은 클래스 10개라 캐시 효과가 큼.
    """
    def __init__(self):
        self.cache: Dict[Tuple[str, torch.dtype, str], torch.Tensor] = {}

    @torch.no_grad()
    def get(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        prompts: List[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Return: (B,77,768)
        out_list = []
        for p in prompts:
            key = (p, dtype, str(device))
            if key in self.cache:
                out_list.append(self.cache[key])
                continue

            tokens = tokenizer(
                p,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(device)
            attn = tokens.attention_mask.to(device)
            emb = text_encoder(input_ids=input_ids, attention_mask=attn)[0].to(dtype=dtype)
            self.cache[key] = emb  # (1,77,768)
            out_list.append(emb)

        return torch.cat(out_list, dim=0)


# ------------------------- SD Sampling / Inversion (latent space) -------------------------

@torch.no_grad()
def unet_predx0_seq_latent(
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    z_lat: torch.Tensor,
    steps: int,
    eta: float,
    device: torch.device,
    cond: torch.Tensor,
) -> torch.Tensor:
    local = DDIMScheduler.from_config(scheduler.config)
    local.set_timesteps(steps, device=device)
    x = z_lat.to(device)

    unet.eval()
    for t in local.timesteps:
        x_in = local.scale_model_input(x, t)
        eps = unet(x_in, t, encoder_hidden_states=cond).sample
        out = local.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x = out.prev_sample

    return x

def unet_predx0_seq_latent_with_grad_s(
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    z_lat: torch.Tensor,
    steps: int,
    eta: float,
    device: torch.device,
    cond: torch.Tensor,
    rkd_step_k: int,
) -> torch.Tensor:
    local = DDIMScheduler.from_config(scheduler.config)
    local.set_timesteps(steps, device=device)
    x = z_lat.to(device)

    unet.train()

    k_step = 0

    for t in local.timesteps:
        x_in = local.scale_model_input(x, t)
        eps = unet(x_in, t, encoder_hidden_states=cond).sample
        out = local.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x = out.prev_sample

        if k_step >= rkd_step_k:
            return out.pred_original_sample
        k_step += 1

    return out.pred_original_sample

def unet_predx0_seq_latent_with_grad_t(
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    z_lat: torch.Tensor,
    steps: int,
    eta: float,
    device: torch.device,
    cond: torch.Tensor,
) -> torch.Tensor:
    local = DDIMScheduler.from_config(scheduler.config)
    local.set_timesteps(steps, device=device)
    x = z_lat.to(device)

    unet.eval()

    for t in local.timesteps:
        x_in = local.scale_model_input(x, t)
        eps = unet(x_in, t, encoder_hidden_states=cond).sample
        out = local.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x = out.prev_sample

    return x

# @torch.no_grad()
def invert_x0latent_to_noise(
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    x0_latent: torch.Tensor,
    steps: int,
    device: torch.device,
    cond: torch.Tensor,
) -> torch.Tensor:
    inv = DDIMInverseScheduler.from_config(scheduler.config)
    inv.set_timesteps(steps, device=device)

    # unet.eval()
    xt = x0_latent
    for t in inv.timesteps:
        x_in = inv.scale_model_input(xt, t)
        eps = unet(x_in, t, encoder_hidden_states=cond).sample
        xt = inv.step(eps, t, xt).prev_sample
    return xt

@torch.no_grad()
def sample_images_sd_ddim(
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    vae: AutoencoderKL,
    scaling_factor: float,
    prompts: List[str],
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    text_cache: TextCondCache,
    image_size: int,
    device: torch.device,
    steps: int,
    eta: float,
    generator: Optional[torch.Generator] = None,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    unet.eval()
    dtype = next(unet.parameters()).dtype

    cond = text_cache.get(tokenizer, text_encoder, prompts, device=device, dtype=dtype)

    local = DDIMScheduler.from_config(scheduler.config)
    local.set_timesteps(steps, device=device)

    h = image_size // 8
    w = image_size // 8
    x = torch.randn((len(prompts), 4, h, w), device=device, dtype=dtype, generator=generator)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )

    with autocast_ctx:
        for t in local.timesteps:
            x_in = local.scale_model_input(x, t)
            eps = unet(x_in, t, encoder_hidden_states=cond).sample
            x = local.step(model_output=eps, timestep=t, sample=x, eta=eta, generator=generator).prev_sample
        imgs = decode_latents_to_images(vae, x, scaling_factor=scaling_factor)

    return imgs


# ------------------------- Losses -------------------------
def tri_vec(D: torch.Tensor) -> torch.Tensor:
    """Upper-triangular (excluding diagonal) flatten."""
    idx = torch.triu_indices(D.size(0), D.size(1), offset=1, device=D.device)
    return D[idx[0], idx[1]]

def compute_losses(
    preds_T_lat: torch.Tensor,
    preds_S_lat: torch.Tensor,
    x0_real_img: torch.Tensor,
    x0_inv_T_img: torch.Tensor,
    vae: AutoencoderKL,
    vae_scaling: float,
    embedder: FeatureEmbedder,
    args,
):
    """
    목표:
      A = [S; R], B = [T; I]
      loss_rel = MSE(pdist(A), pdist(B))

    - S는 grad 유지 (student로 흐르도록)
    - T/R/I는 no_grad로 계산해도 무방 (teacher/GT anchor)
    """
    eps = 1e-12
    device = x0_real_img.device

    # decode (S는 grad 유지, T는 no_grad)
    with torch.no_grad():
        T_last_img = decode_latents_to_images(vae, preds_T_lat, vae_scaling)
    S_last_img = decode_latents_to_images(vae, preds_S_lat, vae_scaling)

    # feature extraction
    with torch.no_grad():
        T_f = embedder(T_last_img)

    S_f = embedder(S_last_img)  # grad path 유지
    R_f = embedder(x0_real_img)
    I_f = embedder(x0_inv_T_img)

    # (권장) cosine 성질로 안정화하고 싶으면 L2 normalize
    # CLIP/DINO에서 특히 유리할 때가 많음
    if getattr(args, "feat_l2norm", False):
        S_f = F.normalize(S_f, dim=1)
        T_f = F.normalize(T_f, dim=1)
        R_f = F.normalize(R_f, dim=1)
        I_f = F.normalize(I_f, dim=1)

    # concat sets
    A = torch.cat([S_f, R_f], dim=0)  # (2B,D)
    B = torch.cat([T_f, I_f], dim=0)  # (2B,D)

    # "한 방" pdist
    dA = torch.pdist(A, p=2).clamp_min(eps)
    dB = torch.pdist(B, p=2).clamp_min(eps)

    # (선택) scale-invariant하게 맞추고 싶으면 평균으로 정규화
    if getattr(args, "rel_mean_norm", True):
        dA = dA / dA.mean().clamp_min(eps)
        dB = dB / dB.mean().clamp_min(eps)

    loss_rel = F.mse_loss(dA, dB, reduction="mean")

    # ---- 아래는 "진단용" (원하면 로그로 분해해서 볼 수 있게) ----
    # 전체 loss는 loss_rel만 쓰되, SS/RR/SR 블록별 MSE를 stats로 제공
    # with torch.no_grad():
    #     # 거리행렬로 블록 분해
    #     A_ng = torch.cat([S_f.detach(), R_f], dim=0).float()
    #     B_ng = torch.cat([T_f, I_f], dim=0).float()

    #     DA = torch.cdist(A_ng, A_ng, p=2)
    #     DB = torch.cdist(B_ng, B_ng, p=2)

    #     Bsz = S_f.shape[0]
    #     SS_A = tri_vec(DA[:Bsz, :Bsz]).clamp_min(eps)
    #     RR_A = tri_vec(DA[Bsz:, Bsz:]).clamp_min(eps)
    #     SR_A = DA[:Bsz, Bsz:].reshape(-1).clamp_min(eps)

    #     TT_B = tri_vec(DB[:Bsz, :Bsz]).clamp_min(eps)
    #     II_B = tri_vec(DB[Bsz:, Bsz:]).clamp_min(eps)
    #     TI_B = DB[:Bsz, Bsz:].reshape(-1).clamp_min(eps)

    #     if getattr(args, "rel_mean_norm", True):
    #         SS_A = SS_A / SS_A.mean().clamp_min(eps)
    #         TT_B = TT_B / TT_B.mean().clamp_min(eps)
    #         RR_A = RR_A / RR_A.mean().clamp_min(eps)
    #         II_B = II_B / II_B.mean().clamp_min(eps)
    #         SR_A = SR_A / SR_A.mean().clamp_min(eps)
    #         TI_B = TI_B / TI_B.mean().clamp_min(eps)

    #     loss_ss = F.mse_loss(SS_A, TT_B, reduction="mean")  # within generated
    #     loss_rr = F.mse_loss(RR_A, II_B, reduction="mean")  # within anchor (invinv 성격)
    #     loss_sr = F.mse_loss(SR_A, TI_B, reduction="mean")  # cross (inv 성격)

    # optional FID-style diag FD (원래 코드 유지)
    loss_fid = torch.tensor(0.0, device=device)
    fid_s = fid_t = torch.tensor(0.0, device=device)
    if args.w_fid != 0.0:
        # 여기서는 동일 feature를 재사용
        fid_s = frechet_distance_diag(S_f.float(), R_f.float(), eps=args.fid_eps)
        fid_t = frechet_distance_diag(T_f.float(), I_f.float(), eps=args.fid_eps)
        loss_fid = fid_s + fid_t

    # 최종 loss: 요청한 “한 방” + (옵션) FID
    total = args.w_rel * loss_rel + args.w_fid * loss_fid

    stats = {
        # 메인
        "loss_rel": loss_rel.detach(),
        # # # 진단용(원하면 프린트/로그)
        # "loss_block/SS": loss_ss.detach(),
        # "loss_block/RR": loss_rr.detach(),
        # "loss_block/SR": loss_sr.detach(),
        # fid
        "loss_fid": loss_fid.detach(),
        "fid_s": fid_s.detach(),
        "fid_t": fid_t.detach(),
    }
    return total, stats


def build_loss_logs(total_loss: torch.Tensor, stats: dict, args) -> dict:
    total = float(total_loss.detach().item())
    rel_raw = float(stats["loss_rel"].item())
    fid_raw = float(stats["loss_fid"].item())

    out = {
        "loss/total": total,
        "loss/rel": rel_raw * float(args.w_rel),
        "loss_raw/rel": rel_raw,
        "loss/fid": fid_raw * float(args.w_fid),
        "loss_raw/fid": fid_raw,
    }

    # 블록별 진단 로그(있으면)
    if "loss_block/SS" in stats:
        out["loss_block/SS"] = float(stats["loss_block/SS"].item())
        out["loss_block/RR"] = float(stats["loss_block/RR"].item())
        out["loss_block/SR"] = float(stats["loss_block/SR"].item())

    return out



# ------------------------- Eval: FID via pytorch-fid -------------------------

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
    student_unet: UNet2DConditionModel,
    ddim: DDIMScheduler,
    vae: AutoencoderKL,
    vae_scaling: float,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    text_cache: TextCondCache,
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
    class_names: List[str],
):
    ensure_dir(out_dir / "samples")
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed) + int(global_step))

    # eval prompts: cycle over class names
    prompts = []
    for i in range(args.sample_n):
        prompts.append(class_names[i % len(class_names)] if len(class_names) > 0 else args.fallback_prompt)

    imgs = sample_images_sd_ddim(
        unet=student_unet,
        scheduler=ddim,
        vae=vae,
        scaling_factor=vae_scaling,
        prompts=prompts,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        text_cache=text_cache,
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
        # random-ish prompts for fid batch
        prompts = []
        for i in range(cur):
            prompts.append(class_names[(cursor + i) % len(class_names)] if len(class_names) > 0 else args.fallback_prompt)

        xs = sample_images_sd_ddim(
            unet=student_unet,
            scheduler=ddim,
            vae=vae,
            scaling_factor=vae_scaling,
            prompts=prompts,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_cache=text_cache,
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
    model_dtype = get_model_dtype(args, device)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    # embedder
    print(f"[Info] Initializing Feature Embedder: {args.rkd_metric}", flush=True)
    embedder = FeatureEmbedder(
        mode=args.rkd_metric,
        device=device,
        clip_model_name=args.clip_model_name,
        dino_model_name=args.dino_model_name,
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

    # dataset: (img, prompt=folder)
    dataset = ImageTextFolderDataset(
        args.student_data_dir,
        image_size=args.image_size,
        center_crop=args.center_crop,
        horizontal_flip=not args.no_hflip,
    )
    class_names = dataset.class_names
    loader = DataLoader(
        dataset,
        batch_size=args.real_batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collate_image_text,
    )
    print(f"[Info] Data: {len(dataset)} imgs under {args.student_data_dir}", flush=True)
    print(f"[Info] Classes (prompts): {class_names}", flush=True)

    # FID real cache
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

    # load SD1.5
    print(f"[Info] Loading SD model: {args.sd_model_id}", flush=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_model_id, subfolder="text_encoder", torch_dtype=model_dtype).to(device)
    vae = AutoencoderKL.from_pretrained(args.sd_model_id, subfolder="vae", torch_dtype=model_dtype).to(device)

    vae_scaling = float(getattr(vae.config, "scaling_factor", args.vae_scaling_factor))

    teacher = UNet2DConditionModel.from_pretrained(args.sd_model_id, subfolder="unet", torch_dtype=model_dtype).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    ddim = DDIMScheduler.from_pretrained(args.sd_model_id, subfolder="scheduler")
    ddim.config.clip_sample = False
    ddim.config.prediction_type = "epsilon"

    # student init (LoRA)
    student = UNet2DConditionModel.from_pretrained(args.sd_model_id, subfolder="unet", torch_dtype=model_dtype).to(device)
    student.requires_grad_(False)

    global_step = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"[Info] Resuming from: {args.resume_checkpoint}", flush=True)
        student = PeftModel.from_pretrained(student, args.resume_checkpoint, is_trainable=True)
        try:
            ckpt_name = Path(args.resume_checkpoint).name
            if "step" in ckpt_name:
                global_step = int(ckpt_name.split("step")[-1])
        except Exception:
            pass
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            init_lora_weights="gaussian",
        )
        student = get_peft_model(student, lora_config)

    student.print_trainable_parameters()
    student.train()

    print(f"[Info] Teacher params: {count_parameters(teacher):,}", flush=True)
    print(f"[Info] Student params: {count_parameters(student):,}", flush=True)

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad(set_to_none=True)

    text_cache = TextCondCache()

    for epoch in range(1, args.epochs + 1):
        student.train()
        print(f"[Epoch {epoch}] start (Global Step: {global_step})", flush=True)

        for it, (x0_real_img, prompts) in enumerate(loader, start=1):
            x0_real_img = x0_real_img.to(device, non_blocking=True)
            B = x0_real_img.shape[0]

            if args.ddim_steps_min == args.ddim_steps_max:
                ddim_steps_train = int(args.ddim_steps_min)
            else:
                ddim_steps_train = int(torch.randint(args.ddim_steps_min, args.ddim_steps_max + 1, (1,)).item())

            rkd_step_k = random.randrange(ddim_steps_train)
            # rkd_step_k = ddim_steps_train - 1

            # text conditioning per-sample
            cond_dtype = next(student.parameters()).dtype
            with torch.no_grad():
                cond = text_cache.get(tokenizer, text_encoder, prompts, device=device, dtype=cond_dtype)

            # latent noise (match batch size)
            h = args.image_size // 8
            w = args.image_size // 8
            z = torch.randn((B, 4, h, w), device=device, dtype=cond_dtype)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                if (use_amp and device.type == "cuda")
                else nullcontext()
            )

            with autocast_ctx:
                # teacher preds (no grad)
                with torch.no_grad():
                    preds_T_lat = unet_predx0_seq_latent(
                        teacher, ddim, z, steps=ddim_steps_train, eta=args.ddim_eta, device=device, cond=cond
                    )

                 # student preds (grad)
                preds_S_lat = unet_predx0_seq_latent_with_grad_s(
                    student, ddim, z, steps=ddim_steps_train, eta=args.ddim_eta, device=device, cond=cond, rkd_step_k=rkd_step_k
                )

                # inversion 

                x0_real_lat = encode_images_to_latents(vae, x0_real_img, scaling_factor=vae_scaling)
                zT_real = invert_x0latent_to_noise(
                    student, ddim, x0_real_lat, steps=ddim_steps_train, device=device, cond=cond
                )
                preds_T_inv_lat = unet_predx0_seq_latent_with_grad_t(
                    teacher, ddim, zT_real, steps=ddim_steps_train, eta=args.ddim_eta, device=device, cond=cond
                )
                x0_inv_T_img = decode_latents_to_images(vae, preds_T_inv_lat, vae_scaling)

            total_loss, stats = compute_losses(
                preds_T_lat=preds_T_lat,
                preds_S_lat=preds_S_lat,
                x0_real_img=x0_real_img,
                x0_inv_T_img=x0_inv_T_img,
                vae=vae,
                vae_scaling=vae_scaling,
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
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)

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
                    f"rel={float(stats['loss_rel'].item()):.6f} "
                    # f"SS={float(stats['loss_block/SS'].item()):.6f} "
                    # f"RR={float(stats['loss_block/RR'].item()):.6f} "
                    # f"SR={float(stats['loss_block/SR'].item()):.6f} "
                    f"fid={float(stats['loss_fid'].item()):.6f}"
                )

                print(line, flush=True)
                with summary_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")

                if wandb_run is not None and wandb is not None:
                    try:
                        wandb.log({**build_loss_logs(total_loss, stats, args),
                                   "train/epoch": int(epoch),
                                   "train/step": int(global_step),
                                   "train/lr": float(args.lr)}, step=global_step)
                    except Exception:
                        pass

            if args.sample_interval > 0 and (global_step % args.sample_interval == 0):
                eval_sample_and_fid(
                    student_unet=student,
                    ddim=ddim,
                    vae=vae,
                    vae_scaling=vae_scaling,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    text_cache=text_cache,
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
                    class_names=class_names,
                )

            if args.save_interval > 0 and (global_step % args.save_interval == 0):
                save_dir = out_dir / "ckpts" / f"ckpt_step{global_step:06d}"
                ensure_dir(save_dir)
                student.save_pretrained(save_dir.as_posix())
                ddim.save_pretrained(save_dir.as_posix())
                with (save_dir / "base_model.txt").open("w", encoding="utf-8") as f:
                    f.write(args.sd_model_id + "\n")
                print(f"[CKPT] Saved student LoRA to {save_dir}", flush=True)


# ------------------------- Args -------------------------

def build_argparser():
    p = argparse.ArgumentParser("SD1.5 Teacher/Student LoRA distillation (image+text; prompt=folder name)")

    # SD
    p.add_argument("--sd_model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--vae_scaling_factor", type=float, default=0.18215)
    p.add_argument("--fallback_prompt", type=str, default="")  # if no class names found

    # resume
    p.add_argument("--resume_checkpoint", type=str, default="")

    # data
    p.add_argument("--student_data_dir", type=str, default="cifar10_student_data_n10/gray3/train")
    p.add_argument("--test_dir", type=str, default="cifar10_png_linear_only/gray3/test")
    p.add_argument("--output_dir", type=str, default="0114_rkd_sd_cifar10_graytest-one-combined")

    # metric
    p.add_argument("--rkd_metric", type=str, default="clip", choices=["pixel", "inception", "clip", "dinov3"])
    p.add_argument("--dino_model_name", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    p.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")

    # device
    p.add_argument("--device", type=str, default="cuda:6")
    p.add_argument("--project", type=str, default="0114_rkd-feature-sd15")
    p.add_argument("--run_name", type=str, default="student-lora-sd15-one-combined")
    p.add_argument("--wandb_offline", action="store_true")
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    # image
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--center_crop", action="store_true")
    p.add_argument("--no_hflip", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)

    # train
    p.add_argument("--epochs", type=int, default=1000000)
    p.add_argument("--real_batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # ddim
    p.add_argument("--ddim_steps_min", type=int, default=6)
    p.add_argument("--ddim_steps_max", type=int, default=10)
    p.add_argument("--ddim_eta", type=float, default=0.0)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)

    # loss weights
    p.add_argument("--w_rkd", type=float, default=1.0)
    p.add_argument("--w_inv", type=float, default=1.0)
    p.add_argument("--w_invinv", type=float, default=1.0)
    p.add_argument("--w_fid", type=float, default=0.01)

    p.add_argument("--w_rel", type=float, default=1.0)
    p.add_argument("--feat_l2norm", action="store_true")      # (선택) feature L2 normalize
    # p.add_argument("--rel_mean_norm", action="store_true")    # (선택) pdist 평균으로 정규화
    p.add_argument("--rel_mean_norm", action="store_true", default=True)

    # misc loss configs
    p.add_argument("--fid_eps", type=float, default=1e-8)

    # logging / eval
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--sample_interval", type=int, default=1000)
    p.add_argument("--sample_n", type=int, default=25)
    p.add_argument("--sample_steps", type=int, default=8)
    p.add_argument("--sample_eta", type=float, default=0.0)

    # fid
    p.add_argument("--disable_fid", action="store_true")
    p.add_argument("--fid_batch_size", type=int, default=16)
    p.add_argument("--fid_gen_batch", type=int, default=16)
    p.add_argument("--fid_dims", type=int, default=2048)
    p.add_argument("--fid_keep_gen", action="store_true")
    p.add_argument("--fid_num_samples", type=int, default=0)
    p.add_argument("--fid_no_symlink", action="store_true")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    ensure_dir(Path(args.output_dir))
    train(args)
