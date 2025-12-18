#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Student (x0 predictor) distillation with pixel-space losses ONLY (SINGLE GPU, NO accelerate, NO grad accumulation):
  - RKD (pdist) in pixel space
  - INV (cdist) in pixel space
  - INVINV (pdist) in pixel space
  - FID loss (full-cov Gaussian FID, sqrtm via eigh) in FULL pixel space (NO downsampling)
  - SAME loss (trajectory shrink regularizer)

+ Periodic evaluation:
  - Sample with STUDENT via DDIM (x0-predictor) every --sample_interval steps
  - Compute standard FID via pytorch-fid vs TEST set (disk images) every --sample_interval steps

Teacher: epsilon-predictor UNet2DModel from diffusers (from_pretrained).
Student: x0-predictor UNet2DModel (DDIM prediction_type="sample").

Notes:
- Device is chosen via --device (e.g., "cuda:0", "cuda:1", "cpu").
- Mixed precision is optional via --mixed_precision {no, fp16, bf16}.
- No gradient accumulation: optimizer step every iteration.
- For eval FID you need: pip install pytorch-fid
"""

import os
import math
import argparse
import shutil
from pathlib import Path
from typing import List, Optional
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DDIMInverseScheduler


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
    """
    x: [-1,1] (N,3,H,W) -> save as PNG 8-bit.
    """
    ensure_dir(out_dir)
    x01 = (x.clamp(-1, 1) + 1) / 2.0
    x255 = (x01 * 255.0).clamp(0, 255).byte().cpu()
    for i in range(x255.shape[0]):
        arr = x255[i].permute(1, 2, 0).numpy()
        Image.fromarray(arr).save(out_dir / f"gen_{start_idx + i:06d}.png")

def flatten_img(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.shape[0], -1)

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
    """
    Single-GPU mean over a list of 1D tensors (detached), used for mean-normalization.
    Returns a scalar tensor on `device`.
    """
    if len(vectors) == 0:
        return torch.tensor(1.0, device=device)

    s = torch.zeros((), device=device, dtype=torch.float64)
    c = torch.zeros((), device=device, dtype=torch.float64)

    for v in vectors:
        vv = v#.detach()
        s = s + vv.sum().to(torch.float64)
        c = c + torch.tensor(float(vv.numel()), device=device, dtype=torch.float64)

    m = (s / c.clamp_min(1.0)).to(torch.float32)
    return m.clamp_min(eps)#.detach()

def resolve_device(device_str: str) -> torch.device:
    try:
        dev = torch.device(device_str)
    except Exception:
        print(f"[Warn] Invalid --device '{device_str}'. Falling back to 'cpu'.", flush=True)
        return torch.device("cpu")

    if dev.type == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA requested but not available. Falling back to 'cpu'.", flush=True)
        return torch.device("cpu")

    if dev.type == "cuda":
        if dev.index is not None and dev.index >= torch.cuda.device_count():
            print(
                f"[Warn] CUDA device index {dev.index} out of range (count={torch.cuda.device_count()}). "
                f"Falling back to 'cuda:0'.",
                flush=True,
            )
            return torch.device("cuda:0")

    return dev

def collect_image_paths_recursive(root: Path, exts={".png", ".jpg", ".jpeg"}) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def flatten_real_cache(test_dir: Path, cache_dir: Path, use_symlink: bool = True) -> int:
    """
    Flattens (symlink or copy) all images under test_dir into cache_dir for pytorch-fid.
    If already exists, reuse.
    """
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


# ------------------------- Full-cov Gaussian FID (toy-style exact Gaussian; used as TRAIN LOSS only) -------------------------

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


# ------------------------- Dataset (student data) -------------------------

class StudentImageFolderDataset(Dataset):
    """
    Student data image folder dataset: returns x in [-1, 1], shape [3, H, W].
    """
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
    ddim.config.prediction_type = prediction_type  # "epsilon" or "sample"
    return ddim


# ------------------------- Sampling / Inversion -------------------------

@torch.no_grad()
def teacher_predx0_seq(teacher, ddim_T, z, steps, eta, device) -> List[torch.Tensor]:
    local = DDIMScheduler.from_config(ddim_T.config)
    local.set_timesteps(steps, device=device)
    x = z.to(device)

    teacher.eval()
    preds: List[torch.Tensor] = []
    for t in local.timesteps:
        x_in = local.scale_model_input(x, t)
        eps  = teacher(x_in, t).sample
        out  = local.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x    = out.prev_sample
        preds.append(out.pred_original_sample)
    return preds

def student_predx0_seq_with_grad(student, ddim_S, z, steps, eta, device) -> List[torch.Tensor]:
    local = DDIMScheduler.from_config(ddim_S.config)
    local.set_timesteps(steps, device=device)
    x = z.to(device)

    student.train()
    preds: List[torch.Tensor] = []
    for t in local.timesteps:
        x_in = local.scale_model_input(x, t)
        x0   = student(x_in, t).sample  # x0 prediction
        out  = local.step(model_output=x0, timestep=t, sample=x, eta=eta)
        x    = out.prev_sample
        preds.append(out.pred_original_sample)
    return preds

@torch.no_grad()
def invert_x0_to_zT_deterministic_x0pred(student, ddim_S, x0, steps, device) -> torch.Tensor:
    inv = DDIMInverseScheduler.from_config(ddim_S.config)
    inv.set_timesteps(steps, device=device)      # [T' - 1, ..., 0]
    student.train()

    xt = x0
    for t in inv.timesteps:
        t_b = torch.full((xt.shape[0],), int(t), device=device, dtype=torch.long)
        latent_in = inv.scale_model_input(xt, t)
        x0 = student(latent_in, t_b).sample
        xt = inv.step(x0, t, xt).prev_sample

    return x0

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
    """
    DDIM sampling for x0-predictor student (scheduler prediction_type="sample").
    Returns x in [-1,1], shape (N,3,H,W).
    """
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
            x0 = student(x_in, t).sample
            x = local.step(model_output=x0, timestep=t, sample=x, eta=eta, generator=generator).prev_sample

    if was_training:
        student.train()
    return x


# ------------------------- Losses -------------------------

def compute_losses(
    preds_T: List[torch.Tensor],
    preds_S: List[torch.Tensor],
    x0_real: torch.Tensor,
    x0_inv_T: torch.Tensor,
    args,
):
    eps = 1e-12
    device = x0_real.device

    T_last = preds_T[-1].detach()
    S_last = preds_S[-1]

    # ---- RKD vectors ----
    rkd_s_list, rkd_t_list = [], []
    if args.rkd_teacher_ref == "last":
        T_ref_pdist = pdist_vec(flatten_img(T_last).float(), eps=eps).detach()
        for k in range(0, len(preds_S), max(1, args.rkd_stride)):
            rkd_s_list.append(pdist_vec(flatten_img(preds_S[k]).float(), eps=eps))
            rkd_t_list.append(T_ref_pdist)
    else:  # matched
        for k in range(0, len(preds_S), max(1, args.rkd_stride)):
            rkd_s_list.append(pdist_vec(flatten_img(preds_S[k]).float(), eps=eps))
            rkd_t_list.append(pdist_vec(flatten_img(preds_T[k].detach()).float(), eps=eps).detach())

    # ---- INV / INVINV vectors ----
    inv_s = inv_t = None
    if args.w_inv != 0.0:
        inv_s = cdist_vec(flatten_img(S_last).float(), flatten_img(x0_real).float(), eps=eps)
        inv_t = cdist_vec(flatten_img(T_last).float(), flatten_img(x0_inv_T).float(), eps=eps)#.detach()

    invinv_s = invinv_t = None
    if args.w_invinv != 0.0:
        invinv_s = pdist_vec(flatten_img(x0_real).float(), eps=eps)
        invinv_t = pdist_vec(flatten_img(x0_inv_T).float(), eps=eps)#.detach()

    # ---- mean normalization (single GPU) ----
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

    # ---- scalar losses ----
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

    # ---- TRAIN loss FID (Gaussian over full pixels) ----
    loss_fid = torch.tensor(0.0, device=device)
    fid_s = fid_t = torch.tensor(0.0, device=device)
    if args.w_fid != 0.0:
        S_f = flatten_img(S_last).float()
        R_f = flatten_img(x0_real).float()
        T_f = flatten_img(T_last).float()
        I_f = flatten_img(x0_inv_T).float()
        fid_s = fid_gaussian_torch(S_f, R_f, eps=args.fid_eps)          # grad flows
        fid_t = fid_gaussian_torch(T_f, I_f, eps=args.fid_eps)#.detach() # detached
        loss_fid = fid_s + fid_t

    # ---- SAME ----
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
        args.w_fid * loss_fid +
        args.w_same * loss_same
    )

    stats = {
        "loss_rkd": loss_rkd.detach(),
        "loss_inv": loss_inv.detach(),
        "loss_invinv": loss_invinv.detach(),
        "loss_fid": loss_fid.detach(),
        "fid_s": fid_s.detach(),
        "fid_t": fid_t.detach(),
        "loss_same": loss_same.detach(),
        "student_mean_dist": student_mean.detach(),
        "teacher_mean_dist": teacher_mean.detach(),
    }
    return total, stats

def build_loss_logs(total_loss: torch.Tensor, stats: dict, args) -> dict:
    """
    W&B logs:
      - loss/*      : weighted (i.e., contribution used in total)
      - loss_raw/*  : raw (if weight != 0, divide weighted by weight; else keep raw)
    """
    def w_and_raw(raw_val: float, w: float):
        w = float(w)
        weighted = raw_val * w
        raw = (weighted / w) if (w != 0.0) else raw_val
        return weighted, raw

    # raw (unweighted) values from stats
    rkd_raw    = float(stats["loss_rkd"].item())
    inv_raw    = float(stats["loss_inv"].item())
    invinv_raw = float(stats["loss_invinv"].item())
    fid_raw    = float(stats["loss_fid"].item())
    fid_s_raw  = float(stats["fid_s"].item())
    fid_t_raw  = float(stats["fid_t"].item())
    same_raw   = float(stats["loss_same"].item())
    total      = float(total_loss.detach().item())

    # weighted + raw (raw defined as weighted/weight when weight!=0)
    rkd_w,    rkd_raw2    = w_and_raw(rkd_raw,    args.w_rkd)
    inv_w,    inv_raw2    = w_and_raw(inv_raw,    args.w_inv)
    invinv_w, invinv_raw2 = w_and_raw(invinv_raw, args.w_invinv)
    fid_w,    fid_raw2    = w_and_raw(fid_raw,    args.w_fid)
    fid_s_w,  fid_s_raw2  = w_and_raw(fid_s_raw,  args.w_fid)
    fid_t_w,  fid_t_raw2  = w_and_raw(fid_t_raw,  args.w_fid)
    same_w,   same_raw2   = w_and_raw(same_raw,   args.w_same)

    logs = {
        # weighted (used in total)
        "loss/total": total,
        "loss/rkd": rkd_w,
        "loss/inv": inv_w,
        "loss/invinv": invinv_w,
        "loss/fid": fid_w,
        "loss/fid_s": fid_s_w,
        "loss/fid_t": fid_t_w,
        "loss/same": same_w,

        # raw (defined as weighted/weight if weight!=0)
        # "loss_raw/total": total,
        "loss_raw/rkd": rkd_raw2,
        "loss_raw/inv": inv_raw2,
        "loss_raw/invinv": invinv_raw2,
        "loss_raw/fid": fid_raw2,
        # "loss_raw/fid_s": fid_s_raw2,
        # "loss_raw/fid_t": fid_t_raw2,
        "loss_raw/same": same_raw2,
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
    # ---- Sampling preview ----
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

    # wandb image log
    if wandb_run is not None and wandb is not None:
        try:
            wandb.log({"eval/samples": wandb.Image(grid)}, step=global_step)
        except Exception:
            pass

    with summary_path.open("a", encoding="utf-8") as f:
        f.write(f"[EVAL] step={global_step:08d} saved_samples={grid_path.as_posix()}\n")

    # ---- FID (standard, pytorch-fid) ----
    if args.disable_fid:
        return
    if fid_real_all_dir is None or (not fid_real_all_dir.exists()):
        print("fid_real_all_dir is None")
        return
    if num_test_imgs_all <= 0:
        print("num_test_imgs_all <= 0")
        return

    fid_num = num_test_imgs_all if args.fid_num_samples <= 0 else int(min(args.fid_num_samples, num_test_imgs_all))
    gen_dir = out_dir / "fid" / f"step{global_step:08d}"
    if gen_dir.exists():
        shutil.rmtree(gen_dir)
    ensure_dir(gen_dir)

    # generate fid_num samples to disk
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

    # overall fid
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

    # per-class optional
    if args.fid_per_class and (fid_real_per_class_root is not None) and fid_class_names:
        per_class = {}
        for cname in fid_class_names:
            real_c = fid_real_per_class_root / cname
            if not real_c.exists():
                continue
            try:
                fid_c = compute_fid_pytorch_fid(
                    real_dir=real_c,
                    gen_dir=gen_dir,
                    device=device,
                    batch_size=args.fid_batch_size,
                    dims=args.fid_dims,
                )
                per_class[cname] = fid_c
                if wandb_run is not None and wandb is not None:
                    try:
                        wandb.log({f"eval/fid_class/{cname}": fid_c}, step=global_step)
                    except Exception:
                        pass
            except Exception:
                continue

        if per_class:
            txt = ", ".join([f"{k}={v:.4f}" for k, v in per_class.items()])
            with summary_path.open("a", encoding="utf-8") as f:
                f.write(f"[FID] step={global_step:08d} per_class: {txt}\n")

    if not args.fid_keep_gen:
        shutil.rmtree(gen_dir, ignore_errors=True)


# ------------------------- Train -------------------------

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
        print(f"[Warn] wandb init failed or not installed. Proceeding without wandb. ({e})", flush=True)

    # AMP settings
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

    if device.type == "cuda":
        torch.cuda.set_device(device)

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "samples")
    ensure_dir(out_dir / "ckpts")
    ensure_dir(out_dir / "fid")
    summary_path = out_dir / "summary.txt"

    # ---- Train dataset ----
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

            # per-class cache (optional)
            if args.fid_per_class:
                fid_real_per_class_root = fid_real_root / "per_class"
                ensure_dir(fid_real_per_class_root)
                class_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
                for cd in sorted(class_dirs, key=lambda x: x.name):
                    cname = cd.name
                    cache_dir = fid_real_per_class_root / cname
                    flatten_real_cache(cd, cache_dir, use_symlink=not args.fid_no_symlink)
                    fid_class_names.append(cname)

            print(f"[FID] Real cache ready: all={num_test_imgs_all} imgs @ {fid_real_all_dir}", flush=True)
        else:
            print(f"[FID] test_dir does not exist: {test_dir}. Disabling FID.", flush=True)
            args.disable_fid = True
    else:
        if not args.disable_fid:
            print("[FID] --test_dir is empty; disabling FID.", flush=True)
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
    # ddim_S = make_ddim(ddpm, prediction_type="sample")  # student x0-predictor DDIM
    ddim_S = make_ddim(ddpm, prediction_type="epsilon")  # student x0-predictor DDIM

    # ---- Student ----
    if args.student_dir != "":
        student_dir = Path(args.student_dir)
        student = UNet2DModel.from_pretrained(student_dir.as_posix()).to(device)
    else:
        student = UNet2DModel(
            sample_size=args.image_size,
            in_channels=3,
            out_channels=3,
            block_out_channels=tuple(args.student_channels),
            down_block_types=("DownBlock2D",) * len(args.student_channels),
            up_block_types=("UpBlock2D",) * len(args.student_channels),
            layers_per_block=args.layers_per_block,
            norm_num_groups=args.norm_num_groups,
            attention_head_dim=None,
        ).to(device)

    student.train()
    for p in student.parameters():
        p.requires_grad = True

    print(f"[Info] Teacher params: {count_parameters(teacher):,}", flush=True)
    print(f"[Info] Student params: {count_parameters(student):,}", flush=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad(set_to_none=True)

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        student.train()
        print(f"[Epoch {epoch}] start", flush=True)

        for it, x0_real in enumerate(loader, start=1):
            # training DDIM steps (randomized range)
            if args.ddim_steps_min == args.ddim_steps_max:
                ddim_steps_train = int(args.ddim_steps_min)
            else:
                ddim_steps_train = int(torch.randint(args.ddim_steps_min, args.ddim_steps_max + 1, (1,)).item())

            z = torch.randn((args.noise_batch, 3, args.image_size, args.image_size), device=device)
            x0_real = x0_real.to(device, non_blocking=True)

            if use_amp:
                autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
            else:
                autocast_ctx = nullcontext()

            with autocast_ctx:
                with torch.no_grad():
                    preds_T = teacher_predx0_seq(
                        teacher, ddim_T, z, steps=ddim_steps_train, eta=args.ddim_eta, device=device
                    )
                preds_S = student_predx0_seq_with_grad(
                    student, ddim_S, z, steps=ddim_steps_train, eta=args.ddim_eta, device=device
                )

            # with torch.no_grad():
            zT_real = invert_x0_to_zT_deterministic_x0pred(
                student, ddim_S, x0_real, steps=ddim_steps_train, device=device
            )
            preds_T_inv = teacher_predx0_seq(
                teacher, ddim_T, zT_real, steps=ddim_steps_train, eta=args.ddim_eta, device=device
            )
            x0_inv_T = preds_T_inv[-1]#.detach()

            with autocast_ctx:
                total_loss, stats = compute_losses(
                    preds_T=preds_T,
                    preds_S=preds_S,
                    x0_real=x0_real,
                    x0_inv_T=x0_inv_T,
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

            # ---- Train log ----
            if global_step % args.log_interval == 0:
                m = {
                    "total": float(total_loss.detach().item()),
                    "rkd": float(stats["loss_rkd"].item()),
                    "inv": float(stats["loss_inv"].item()),
                    "invinv": float(stats["loss_invinv"].item()),
                    "fid": float(stats["loss_fid"].item()),
                    "fid_s": float(stats["fid_s"].item()),
                    "fid_t": float(stats["fid_t"].item()),
                    "same": float(stats["loss_same"].item()),
                    "smean": float(stats["student_mean_dist"].item()),
                    "tmean": float(stats["teacher_mean_dist"].item()),
                }

                line = (
                    f"[Epoch {epoch:03d}] step={global_step:08d} "
                    f"total={m['total']:.6f} rkd={m['rkd']:.6f} inv={m['inv']:.6f} "
                    f"invinv={m['invinv']:.6f} fid={m['fid']:.6f}(s={m['fid_s']:.6f},t={m['fid_t']:.6f}) "
                    f"same={m['same']:.6f} sMean={m['smean']:.6e} tMean={m['tmean']:.6e} "
                    f"ddim_steps_train={ddim_steps_train}"
                )
                print(line, flush=True)
                with summary_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                
                #
                if wandb_run is not None and wandb is not None:
                    try:
                        loss_logs = build_loss_logs(total_loss, stats, args)

                        wandb.log({
                            # ✅ losses under loss/ and loss_raw/
                            **loss_logs,

                            # (loss 외 메타는 유지해도 됨)
                            "train/ddim_steps_train": int(ddim_steps_train),
                            "train/epoch": int(epoch),
                            "train/step": int(global_step),
                            "train/lr": float(args.lr),

                            # (원래 찍던 dist도 loss가 아니니 train/에 두거나 다른 prefix로 두면 됨)
                            "train/student_mean_dist": float(stats["student_mean_dist"].item()),
                            "train/teacher_mean_dist": float(stats["teacher_mean_dist"].item()),
                        }, step=global_step)
                    except Exception:
                        pass

            # ---- Periodic eval: sampling + FID ----
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

            # ---- Checkpoint ----
            if args.save_interval > 0 and (global_step % args.save_interval == 0):
                save_dir = out_dir / "ckpts" / f"ckpt_step{global_step:06d}"
                ensure_dir(save_dir)
                student.save_pretrained(save_dir.as_posix())
                ddpm.save_pretrained(save_dir.as_posix())
                print(f"[CKPT] Saved student to {save_dir}", flush=True)

        # epoch-end "last"
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

BATCH_SIZE = 32
CLASSN = 100
CUDA_NUM = 3
RKD_W = 0.1
INV_W = 0.1
INVINV_W = 1.0
FD_W = 0.0000001
SAME_W = 0.0


def build_argparser():
    p = argparse.ArgumentParser("Student x0 distillation (single GPU, no accelerate, periodic sampling + FID eval)")

    # student data directory (alias: --train_dir for backward compatibility)
    p.add_argument("--student_data_dir", "--train_dir", dest="student_data_dir",
                   type=str, default="cifar10_student_data_n100/gray3/train")

    # test dir for standard FID (pytorch-fid)
    p.add_argument("--test_dir", type=str, default="cifar10_png_linear_only/gray3/test",
                   help="Folder containing test images (optionally class subdirs). Used for standard FID via pytorch-fid.")

    p.add_argument("--teacher_dir", type=str, default="ddpm_cifar10_rgb_T400_DDIM50/ckpt_step150000")
    p.add_argument("--student_dir", type=str, default="ddpm_cifar10_rgb_T400_DDIM50/ckpt_step150000")
    p.add_argument("--output_dir", type=str, default=f"out_1218_rkd_cifar10_rgb_to_gray_single_batch{BATCH_SIZE}_N{CLASSN}-FD-rkdW{RKD_W}-invW{INV_W}-invinvW{INVINV_W}-fdW{FD_W}-sameW{SAME_W}-teacher-init-eps")

    p.add_argument("--device", type=str, default=f"cuda:{CUDA_NUM}", help='e.g., "cuda:0", "cuda:1", "cpu"')

    p.add_argument("--project", type=str, default="rkd-cifar10-rgb-to-gray-1218")
    p.add_argument("--run_name", type=str, default=f"student-x0-pixel-rgb-to-gray-batch{BATCH_SIZE}-N{CLASSN}-FD-rkdW{RKD_W}-invW{INV_W}-invinvW{INVINV_W}-fdW{FD_W}-sameW{SAME_W}-teacher-init-eps")
    p.add_argument("--wandb_offline", action="store_true")
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--center_crop", action="store_true")
    p.add_argument("--no_hflip", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--real_batch", type=int, default=BATCH_SIZE)
    p.add_argument("--noise_batch", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--train_timesteps", type=int, default=400)
    p.add_argument("--beta_schedule", type=str, default="linear",
                   choices=["linear", "scaled_linear", "squaredcos_cap_v2"])

    # training DDIM steps (random range)
    p.add_argument("--ddim_steps_min", type=int, default=40)
    p.add_argument("--ddim_steps_max", type=int, default=60)
    p.add_argument("--ddim_eta", type=float, default=0.0)

    p.add_argument("--student_channels", type=int, nargs="+", default=[128, 256, 256])
    p.add_argument("--layers_per_block", type=int, default=2)
    p.add_argument("--norm_num_groups", type=int, default=32)

    # weights
    p.add_argument("--w_rkd", type=float, default=RKD_W)
    p.add_argument("--w_inv", type=float, default=INV_W)
    p.add_argument("--w_invinv", type=float, default=INVINV_W)
    p.add_argument("--w_fid", type=float, default=FD_W)
    p.add_argument("--w_same", type=float, default=SAME_W)


    # RKD options
    p.add_argument("--rkd_stride", type=int, default=1)
    p.add_argument("--rkd_teacher_ref", type=str, default="last", choices=["last", "matched"])

    # SAME options
    p.add_argument("--same_mode", type=str, default="mean", choices=["mean", "last"])

    # TRAIN loss FID eps (Gaussian full-pixel)
    p.add_argument("--fid_eps", type=float, default=1e-8)

    # logging / ckpt
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=2000)

    # periodic eval: sampling + standard FID
    p.add_argument("--sample_interval", type=int, default=2000, help="Do sampling + FID every N steps (0 disables)")
    p.add_argument("--sample_n", type=int, default=64, help="Preview sample count (prefer square number)")
    p.add_argument("--sample_steps", type=int, default=50, help="DDIM steps for evaluation sampling/FID")
    p.add_argument("--sample_eta", type=float, default=0.0, help="DDIM eta for evaluation sampling/FID")

    # standard FID (pytorch-fid) options
    p.add_argument("--disable_fid", action="store_true", help="Disable standard FID computation/logging")
    p.add_argument("--fid_batch_size", type=int, default=64, help="Batch size for Inception during FID")
    p.add_argument("--fid_gen_batch", type=int, default=256, help="Batch size for generating samples for FID")
    p.add_argument("--fid_dims", type=int, default=2048, help="Inception feature dims (default 2048)")
    p.add_argument("--fid_keep_gen", action="store_true", help="Keep generated images used for FID on disk")
    p.add_argument("--fid_num_samples", type=int, default=0, help="How many samples to generate for FID (0=use all test images)")
    p.add_argument("--fid_per_class", action="store_false", help="Also compute per-class FID if test_dir has class subdirs")
    p.add_argument("--fid_no_symlink", action="store_true", help="Do not use symlinks when building real cache")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    ensure_dir(Path(args.output_dir))
    train(args)
