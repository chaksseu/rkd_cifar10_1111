#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import json, math, random
from pathlib import Path
from typing import Dict, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from diffusers import DDPMScheduler, DDIMScheduler, DDIMInverseScheduler
from torch.utils.data import Dataset, DataLoader
import itertools
# ===================== CONFIG ===================== #

W_RKD = 0.1
W_INV = 0.1
W_INVINV = 1.0
W_FID = 0.0001
W_DIFF = 0.0
CUDA_NUM = 2
BATCH_SIZE = 1024

WANDB_NAME=f"1107_lr1e4_n32_b{BATCH_SIZE}_ddim_50_150_steps_no_init_rkdW{W_RKD}_invW{W_INV}_invinvW{W_INVINV}_fidW{W_FID}"


CONFIG = {
    # I/O
    "device": f"cuda:{CUDA_NUM}",
    "out_dir": f"runs/{WANDB_NAME}",
    # teacher / student
    "teacher_ckpt": "ckpt_teacher_T1000_step370000_1021.pt", 
    "student_init_ckpt": "",                     
    # "student_init_ckpt": "runs/1025_lr1e4_n32_b1024_ddim_50_150_steps_no_init_rkdW0.0_invW0.0_invinv_W1.0_diffW0.1/ckpt_student_step200000.pt",                     
    "resume_student_ckpt": f"",        
    "teacher_data_stats": "smile_data_n8192_scale10_rot0_trans_0_0/teacher_normalization_stats.json",
    "student_data_stats": "smile_data_n32_scale2_rot60_trans_50_-20/normalization_stats.json",

    # diffusion loss 가중치
    "W_RKD": W_RKD,
    "W_INV": W_INV,                               # ε-pred MSE 가중치
    "W_INVINV": W_INVINV,                               # ε-pred MSE 가중치
    "W_DIFF": W_DIFF,                               # ε-pred MSE 가중치
    "W_FID": W_FID,                               # ε-pred MSE 가중치

    "rkd_ddim_steps_to_t": 100,   # t_sel까지 최대 몇 번의 DDIM 전이만 사용할지

    "batch_size": BATCH_SIZE,
    "num_noises": 8192, 
    "epochs_total": 5000000,          # 총 스텝 수 (기존 epochs_per_stage 대신 사용)

    "noise_pool_file": None,        # None이면 out_dir/data/noises_pool.npy 로 저장
    "regen_noise_pool": False,      # True면 항상 새로 만듦
    
    # schedule / time
    "T": 1000,                 # total diffusion steps (timesteps = 0..T-1)

    "seed": 42,
    # RKD weights
    "W_NORM": 0.0,
    "use_mu_normalization": True,
    # noises / data dims
    "dim": 2,                 # 2D toy

    # model sizes
    "teacher_hidden": 256, "teacher_depth": 8, "teacher_time_dim": 64,
    "student_hidden": 256, "student_depth": 8, "student_time_dim": 64,
    # optim
    "lr": 1e-4, "weight_decay": 0.0, "max_grad_norm": 1.0,
    # sampling viz
    "vis_interval_epochs": 5000,
    "n_vis": 8192,       # 경로를 수집/표시할 noise 개수
    "ddim_eta": 0.0,
    # wandb
    "use_wandb": True,
    "wandb_project": "RKD-DKDM-AICA-1107",
    "wandb_run_name": WANDB_NAME,
}

CONFIG.update({
    # student 데이터 경로/형식
    "student_data_path": "smile_data_n32_scale2_rot60_trans_50_-20/train.npy",   # 혹은 .csv
    # "student_data_path": "smile_data_n8192_scale10_rot0_trans_0_0/train.npy",   # 혹은 .csv
    "student_data_format": "npy",                # "npy" | "csv"
    "student_dataset_batch_size": 32,          # 없으면 batch_size 사용
})

# ===================== UTILS ===================== #
import re

from pathlib import Path



def _mean_and_cov(X: torch.Tensor, eps: float = 1e-6):
    # X: [N, D] -> (mu[D], C[D,D])
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
    C = 0.5 * (C + C.t()) + eps * I   # 대칭화 + 지터
    return mu.squeeze(0), C

def _sqrtm_psd(A: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    # A: 대칭 PSD 가정
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







def ensure_noise_pool(cfg) -> Path:
    """노이즈 풀(.npy) 보장 생성 후 경로 반환"""
    out_dir = Path(cfg["out_dir"])
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = Path(cfg["noise_pool_file"] or (data_dir / "noises_pool.npy"))
    if cfg.get("regen_noise_pool", False) or (not path.exists()):
        N, D = cfg["num_noises"], cfg["dim"]
        z = np.random.randn(N, D).astype("float32")
        np.save(path, z)
        print(f"[NOISE] generated pool: {path}  shape={(N,D)}")
    else:
        print(f"[NOISE] using existing pool: {path}")
    return path


def make_ddim_like(train_sched, device, T: int):
    """
    train_sched.config(베타/타임스텝 등)를 복사해 DDIM 스케줄러 생성.
    timesteps = [T-1, ..., 0] 로 설정.
    """
    ddim = DDIMScheduler.from_config(train_sched.config)
    ddim.config.clip_sample = False
    ddim.config.prediction_type = "epsilon"
    ddim.set_timesteps(T, device=device)  # [T-1, ..., 0]
    return ddim

def sample_noise_batch(noise_pool: np.ndarray, B: int, device: torch.device) -> torch.Tensor:
    """노이즈 풀에서 인덱스 랜덤 샘플로 배치 구성"""
    N = noise_pool.shape[0]
    idx = np.random.randint(0, N, size=B)
    z = torch.from_numpy(noise_pool[idx]).to(device=device, dtype=torch.float32)
    return z



def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_norm_stats(json_path: str):
    json_path = Path(json_path)
    with json_path.open("r") as f:
        d = json.load(f)           # {"mean": [...], "std": [...]}
    mu = np.array(d["mean"],  dtype=np.float32)
    sigma = np.array(d["std"], dtype=np.float32)
    return mu, sigma

def load_norm_stats_torch(json_path: str, device: str = 'cpu'):
    json_path = Path(json_path)
    with json_path.open("r") as f:
        d = json.load(f)  # {"mean": [...], "std": [...]}
    
    mu_np = np.array(d["mean"], dtype=np.float32)
    sigma_np = np.array(d["std"], dtype=np.float32)

    mu_tensor = torch.from_numpy(mu_np)
    sigma_tensor = torch.from_numpy(sigma_np)

    mu_tensor = mu_tensor.to(device)
    sigma_tensor = sigma_tensor.to(device)
    
    return mu_tensor, sigma_tensor

def save_and_log_xt_pairs_for_all_t(
    left_seq: np.ndarray,   # [K,B,2] Teacher
    right_seq: np.ndarray,  # [K,B,2] Student
    ts: np.ndarray,         # [K]     정수 timestep들 (T-1→...→0)
    noise_ids: np.ndarray,  # [B]     색상 매핑용
    out_dir: Path,
    step_i: int,
    sync_axes: bool = False,
    dot_size: int = 6,
    use_wandb: bool = False,
):
    base = out_dir / "figs" / f"xt_pairs_step{step_i:06d}"
    base.mkdir(parents=True, exist_ok=True)

    logged = {}
    for k, t in enumerate(ts):
        left_xy  = left_seq[k]   # (B,2)
        right_xy = right_seq[k]  # (B,2)
        path = base / f"t{int(t):03d}.png"

        plot_pair_scatter_colored(
            left_xy=left_xy,
            right_xy=right_xy,
            noise_ids=noise_ids,
            N_total=len(noise_ids),
            left_title = fr"Teacher $x_t$ (t={int(t)})",
            right_title= fr"Student $x_t$ (t={int(t)})",
            out_path=path,
            dot_size=dot_size,
            sync_axes=sync_axes,
        )

        if use_wandb:
            logged[f"img/xt_pairs/t{int(t):03d}"] = wandb.Image(str(path))

    if use_wandb and logged:
        wandb.log(logged, step=step_i)

    return base  # 저장된 폴더 경로


def _square_limits_from(data: np.ndarray, pad_ratio: float = 0.05):
    """
    데이터의 x/y 범위를 보고, 긴 변 기준(span)으로 여백을 준 뒤
    중앙 정렬된 정사각형 xlim/ylim을 반환.
    """
    data = np.asarray(data)
    xmin, xmax = float(data[:, 0].min()), float(data[:, 0].max())
    ymin, ymax = float(data[:, 1].min()), float(data[:, 1].max())
    dx, dy = xmax - xmin, ymax - ymin

    # 긴 변 + 최소 폭 보호
    base = max(dx, dy, 1e-3)
    pad  = pad_ratio * base

    # 여백 적용
    xmin -= pad; xmax += pad
    ymin -= pad; ymax += pad

    # 중앙 정렬된 정사각형으로 확장
    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0
    span = max(xmax - xmin, ymax - ymin)
    half = span / 2.0

    return (xmid - half, xmid + half), (ymid - half, ymid + half)


def colors_from_noise_ids(ids: np.ndarray, N_total: int, alpha: float = 0.85):
    """
    같은 'global noise id'에는 항상 같은 색이 나오도록 결정적 매핑.
    - hue = (id * φ) mod 1.0  (φ ≈ 0.618…)
    - colormap: hsv (연속)
    """
    ids = np.asarray(ids, dtype=np.int64)
    phi = 0.6180339887498949
    hues = (ids * phi) % 1.0
    cmap = plt.get_cmap("hsv")
    cols = cmap(hues)
    cols[:, 3] = alpha  # alpha
    return cols

def plot_pair_scatter_colored(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    noise_ids: np.ndarray,
    N_total: int,
    left_title: str,
    right_title: str,
    out_path: Path,
    dot_size: int = 8,
    sync_axes: bool = False,
    pad_ratio: float = 0.05,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    A = np.asarray(left_xy); B = np.asarray(right_xy)
    assert A.shape == B.shape and A.shape[1] == 2, "Expect (B,2) arrays for both panels."

    colors = colors_from_noise_ids(np.asarray(noise_ids), N_total)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4),
                             sharex=sync_axes, sharey=sync_axes)

    # 공통 정사각형 축 범위 (sync_axes=True)
    if sync_axes:
        both = np.vstack([A, B])
        xlim_all, ylim_all = _square_limits_from(both, pad_ratio=pad_ratio)

    for ax, data, title in ((axes[0], A, left_title), (axes[1], B, right_title)):
        ax.scatter(data[:, 0], data[:, 1], s=dot_size, c=colors, edgecolors="none")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)

        if sync_axes:
            ax.set_xlim(*xlim_all); ax.set_ylim(*ylim_all)
        else:
            xlim, ylim = _square_limits_from(data, pad_ratio=pad_ratio)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)




@torch.no_grad()
def collect_xt_seq_ddim(
    model: nn.Module,
    ddim: DDIMScheduler,
    z: torch.Tensor,
    t_stop: int = 0,            # 이 t에서의 x_t까지 포함해서 수집하고 멈춤
    return_ts: bool = True,
    device: str = "cuda",
    sample_steps: int = 100,    # 바깥에서 바꿀 수 있게 인자로 노출
    eta: float = 0.0,           # DDIM 결정론(기본 0.0)
):
    # 스케줄러 복제 및 세팅
    local = DDIMScheduler.from_config(ddim.config)
    local.set_timesteps(sample_steps, device=device)  # descending: [T'−1, ..., 0]

    # 상태 준비
    x = z.to(device).detach().clone()
    B = x.size(0)

    xs, ts = [], []

    was_training = model.training
    model.eval()
    try:
        for t_tensor in local.timesteps:                 # ex) [999, 979, ..., 0]
            t_int = int(t_tensor)

            # 1) (현재 t의) x_t 저장
            xs.append(x.detach().cpu().numpy())
            ts.append(t_int)

            # 2) 멈춤 조건: t_stop에 도달했으면 그 x_t까지 포함하고 break
            if t_int <= t_stop:
                break

            # 3) 다음 스텝으로 진행 (local 사용)
            t_b  = torch.full((B,), t_int, device=x.device, dtype=torch.long)
            x_in = local.scale_model_input(x, t_tensor)
            eps  = model(x_in, t_b)                      # 모델이 epsilon 예측한다고 가정
            out  = local.step(model_output=eps, timestep=t_tensor, sample=x, eta=eta)
            x    = out.prev_sample                       # x_{t-1}
    finally:
        if was_training:
            model.train()

    seq = np.stack(xs, axis=0)  # [K, B, ...]
    return (seq, np.asarray(ts, dtype=int)) if return_ts else seq



def sample_ddim_student(
    model, sample_scheduler, z, device, sample_steps=None, eta=0.0, t_sel=0,
):
    x = z.to(device)
    B = x.shape[0]

    local = DDIMScheduler.from_config(sample_scheduler.config)
    local.set_timesteps(sample_steps, device=device)

    xs = []

    model.train()
    for i, t in enumerate(local.timesteps):
        t_int = int(t)

        t_b = torch.full((B,), t_int, device=device, dtype=torch.long)
        x_in = local.scale_model_input(x, t)
        eps = model(x_in, t_b)
        out = local.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x = out.prev_sample

        xs.append(x) 

        if t_int <= t_sel:
            break
    return xs


@torch.no_grad()
def sample_ddim_teacher(
    model, sample_scheduler, z, device, sample_steps=None, eta=0.0, t_sel=0,
):
    x = z.to(device)
    B = x.shape[0]

    local = DDIMScheduler.from_config(sample_scheduler.config)
    local.set_timesteps(sample_steps, device=device)

    xs = []

    model.eval()
    for t in local.timesteps:  # [T-1, ..., 0] 순서
        t_int = int(t)
        
        t_b = torch.full((B,), t_int, device=device, dtype=torch.long)
        x_in = local.scale_model_input(x, t)
        eps = model(x_in, t_b)
        out = local.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x = out.prev_sample

        xs.append(x) 

        if t_int <= t_sel:
            break
    return xs


def sample_ddim_teacher_grad(
    model, sample_scheduler, z, device, sample_steps=None, eta=0.0, t_sel=0,
):
    x = z.to(device)
    B = x.shape[0]

    local = DDIMScheduler.from_config(sample_scheduler.config)
    local.set_timesteps(sample_steps, device=device)

    xs = []

    model.eval()
    for t in local.timesteps:  # [T-1, ..., 0] 순서
        t_int = int(t)
        
        t_b = torch.full((B,), t_int, device=device, dtype=torch.long)
        x_in = local.scale_model_input(x, t)
        eps = model(x_in, t_b)
        out = local.step(model_output=eps, timestep=t, sample=x, eta=eta)
        x = out.prev_sample

        xs.append(x) 

        if t_int <= t_sel:
            break
    return xs


def sample_ddim_inv_student(
    model, sample_scheduler, x0, device, sample_steps=None, eta=0.0,
):
    """
    DDIM 인버전(eta=0 가정): x_0 -> ... -> x_{T-1}
    Teacher에 다시 주입할 z를 만들 때 사용.
    """
    x = x0.to(device)
    B = x.shape[0]

    inv = DDIMInverseScheduler.from_config(sample_scheduler.config)

    inv.set_timesteps(sample_steps, device=device)      # [T' - 1, ..., 0]

    xs = [x]
    model.train()

    for t in inv.timesteps:
        t_b = torch.full((x.shape[0],), int(t), device=device, dtype=torch.long)
        latent_in = inv.scale_model_input(x, t)
        eps = model(latent_in, t_b)
        x = inv.step(eps, t, x).prev_sample
        xs.append(x) 

    return xs  

# ===================== MODEL ===================== #

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__(); self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        t = t.float().unsqueeze(1)
        freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=t.device) * -1.0)
        angles = t * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1: emb = F.pad(emb, (0, 1))
        return emb

class MLPDenoiser(nn.Module):
    """ ε-predictor for 2D toy; used for both Teacher and Student. """
    def __init__(self, in_dim=2, time_dim=64, hidden=128, depth=3, out_dim=2):
        super().__init__()
        self.t_embed = SinusoidalTimeEmbedding(time_dim)
        layers = []
        for i in range(depth):
            layers += [nn.Linear(in_dim + time_dim if i == 0 else hidden, hidden), nn.SiLU()]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, out_dim)
    def forward(self, x, t):
        te = self.t_embed(t); h = torch.cat([x, te], dim=-1)
        return self.out(self.mlp(h))  # ε(x, t)

# ===================== SCHEDULERS ===================== #

def build_schedulers(num_train_timesteps: int):
    train_sched = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        clip_sample=False,
    )
    train_sched.config.prediction_type = "epsilon"

    sample_sched = DDIMScheduler.from_config(train_sched.config)
    sample_sched.config.clip_sample = False
    sample_sched.config.prediction_type = "epsilon"  # DDIM도 동일하게

    return train_sched, sample_sched


def denormalize_np(arr: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return arr * sigma + mu

def normalize_np(arr: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (arr - mu) / sigma

def denormalize_torch(arr: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # NumPy 버전과 연산 로직은 동일합니다.
    return arr * sigma + mu

def normalize_torch(arr: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # NumPy 버전과 연산 로직은 동일합니다.
    return (arr - mu) / sigma
# ===================== Dataset ===================== #

class StudentX0Dataset(Dataset):
    def __init__(self, path: str, fmt: str, mu: np.ndarray, sigma: np.ndarray):
        self.X = self._load(path, fmt)  # (N,2) or (N,D)
        assert self.X.ndim == 2 and self.X.shape[1] >= 2, "Expect (N,2) or (N,D)"
        self.X = self.X[:, :2].astype(np.float32)
        self.X = normalize_np(self.X, mu, sigma).astype(np.float32)  # 모델 입력 스케일로
    def _load(self, path, fmt):
        p = Path(path)
        if fmt == "npy":
            return np.load(p)
        elif fmt == "csv":
            return np.loadtxt(p, delimiter=",")
        else:
            raise ValueError(f"Unsupported format: {fmt}")
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i]  # x0 (normalized)

def build_student_dataloader(cfg, mu, sigma):
    bs = int(cfg.get("student_dataset_batch_size", cfg["batch_size"]))
    ds = StudentX0Dataset(cfg["student_data_path"], cfg["student_data_format"], mu, sigma)
    return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)

# ===================== RKD on epsilon ===================== #
def loss_rkd_xt(
    xt_s: torch.Tensor,      # [B, D]
    xt_s_2: torch.Tensor,    # [B, D]
    xt_t: torch.Tensor,      # [B, D]
    xt_t_2: torch.Tensor,    # [B, D]
    use_mu_norm: bool = True,
    w_rkd: float = 1.0,
    w_norm: float = 0.0,     # (미사용) 필요없으면 제거해도 됨
    eps: float = 1e-12,
):
    B = xt_s.size(0)
    if B < 1:  # 전체 대응은 B>=1이면 가능(대각만이라도 있음)
        z = xt_s.new_zeros(())
        return {"total": z, "rkd": z}

    # cross-distance 전체 (대각 포함)
    t_full = torch.cdist(xt_t,   xt_t_2, p=2)  # [B,B]
    s_full = torch.cdist(xt_s,   xt_s_2, p=2)  # [B,B]

    # 벡터화: 모든 (i,j) 사용
    t_d = t_full.reshape(-1).clamp_min(eps)
    s_d = s_full.reshape(-1).clamp_min(eps)

    if use_mu_norm:
        t_d = t_d / t_d.mean().clamp_min(eps)
        s_d = s_d / s_d.mean().clamp_min(eps)

    loss_rkd = w_rkd * F.mse_loss(s_d, t_d, reduction="mean")
    return {"total": loss_rkd, "rkd": loss_rkd}




def loss_rkd_xt_pdist(
    xt_s: torch.Tensor,      # [B, D]
    xt_t: torch.Tensor,      # [B, D]
    use_mu_norm: bool = True,
    w_rkd: float = 1.0,
    w_norm: float = 0.0,
    eps: float = 1e-12,
):
    B = xt_s.size(0)
    if B < 2:
        z = xt_s.new_zeros(())
        return {"total": z, "rkd": z, "norm": z, "t_norm": z, "s_norm": z}

    # pairwise distances on epsilon vectors

    s_d = torch.pdist(xt_s, p=2).clamp_min(eps)           # (U,)
    t_d = torch.pdist(xt_t, p=2).clamp_min(eps)  # (U,)

    if use_mu_norm:
        t_d = t_d / t_d.mean().clamp_min(eps)
        s_d = s_d / s_d.mean().clamp_min(eps)
    
    loss_rkd = w_rkd * F.mse_loss(s_d, t_d, reduction="mean")

    total = loss_rkd
    return {"total": total, "rkd": loss_rkd}

 

# ===================== Training ===================== #

def train_student_uniform_xt(cfg: Dict):
    """
    - 노이즈 풀에서 배치 샘플
    - t ~ Uniform{0..T-1}, 배치 내 t 동일
    - 같은 z로 Teacher/Student를 각각 DDIM 역진행
      * Teacher: eval + no_grad → x_t^T
      * Student: train + grad ON (전체 경로) → x_t^S
    - RKD(x_t^S, x_t^T)로 학습
    """
    out_dir = Path(cfg["out_dir"]); (out_dir / "figs").mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # 스케줄러/모델
    train_sched, sample_sched = build_schedulers(cfg["T"])
    ddim = make_ddim_like(train_sched, device, cfg["T"])

    teacher = MLPDenoiser(2, cfg["teacher_time_dim"], cfg["teacher_hidden"], cfg["teacher_depth"], 2).to(device)
    teacher.load_state_dict(torch.load(cfg["teacher_ckpt"], map_location=device), strict=True)
    for p in teacher.parameters(): p.requires_grad = False
    teacher.eval()

    student = MLPDenoiser(2, cfg["student_time_dim"], cfg["student_hidden"], cfg["student_depth"], 2).to(device)
    # init / resume
    if cfg.get("resume_student_ckpt"):
        p = Path(cfg["resume_student_ckpt"])
        if p.exists():
            student.load_state_dict(torch.load(p, map_location=device), strict=True)
            print("[RESUME] Loaded student:", p)
    elif cfg.get("student_init_ckpt"):
        p = Path(cfg["student_init_ckpt"])
        if p.exists():
            student.load_state_dict(torch.load(p, map_location=device), strict=True)
            print("[INIT] Loaded student init:", p)
        else:
            print("[INIT] Student from scratch")

    opt = torch.optim.AdamW(student.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # W&B
    if cfg["use_wandb"]:
        wandb.login()
        wandb.init(project=cfg["wandb_project"], name=cfg["wandb_run_name"], config=cfg)
        wandb.define_metric("step")
        wandb.define_metric("loss/*", step_metric="step")
        wandb.define_metric("loss_by_t/*", step_metric="step")
        wandb.define_metric("loss_by_tp1_t/*", step_metric="step")

    # 노이즈 풀
    noise_path = ensure_noise_pool(cfg)
    noise_pool = np.load(noise_path)  # (N,D), 작으면 통째 로드 OK

    mu_teacher, sigma_teacher = load_norm_stats(cfg["teacher_data_stats"])
    mu_student, sigma_student = load_norm_stats(cfg["student_data_stats"])

    mu_teacher_tensor, sigma_teacher_tensor = load_norm_stats_torch(cfg["teacher_data_stats"], device)
    mu_student_tensor, sigma_student_tensor = load_norm_stats_torch(cfg["student_data_stats"], device)

    # --- Student 도메인 dataloader (diffusion loss용) ---
    student_loader = build_student_dataloader(cfg, mu_student, sigma_student)
    student_iter = iter(student_loader)

    def next_student_batch():
        nonlocal student_iter
        try:
            x0 = next(student_iter)  # (B_s, 2) torch.Tensor로 들어옴
        except StopIteration:
            student_iter = iter(student_loader)
            x0 = next(student_iter)
        if not torch.is_tensor(x0):
            x0 = torch.as_tensor(x0, dtype=torch.float32)
        return x0.to(device, non_blocking=True)




    T = int(cfg["T"])
    total_steps = int(cfg.get("epochs_total", 50000))
    B = int(cfg["batch_size"])

    for step_i in range(1, total_steps + 1):
        # 1) t ~ Uniform{0..T-1} (배치 내 동일)
        # t_sel = int(np.random.randint(0, T))
        t_sel = int(0)
        # t_sel = int(T-1)
        # t_sel = torch.randint(low=0, high=T, size=(B,), device=device, dtype=torch.long)
        
        # z 샘플
        # 동일 노이즈 안에서만 사용하므로 오버피팅 역할
        z = sample_noise_batch(noise_pool, B, device)
        # # random noise
        z = torch.randn_like(z)

        # ddim_steps = int(cfg.get("rkd_ddim_steps_to_t", 5))
        # ddim_steps = 50

        # possible_steps = [200, 195, 191, 190, 187, 174, 161, 157, 148, 135]
        # ddim_steps = random.choice(possible_steps)

        ddim_steps = int(np.random.randint(50,151))


        with torch.no_grad():
            xt_T_seq = sample_ddim_teacher(
                model=teacher, sample_scheduler=ddim, z=z, 
                device=device, sample_steps=ddim_steps,
                eta=float(cfg.get("ddim_eta", 0.0)), t_sel=t_sel,
            )

        student.train()
        xt_S_seq = sample_ddim_student(
            model=student, sample_scheduler=ddim, z=z, 
            device=device, sample_steps=ddim_steps,
            eta=float(cfg.get("ddim_eta", 0.0)), t_sel=t_sel,
        )


        x0_batch = next_student_batch()  # shape (B_s, 2)
        student.train()

        # ===================== STUDENT DOMAIN DATA =====================
        # 학생 도메인 x0 배치 가져오기 (정규화 완료 상태)        
        S_inv_z_seq = sample_ddim_inv_student(
            model=student, sample_scheduler=ddim, x0=x0_batch, 
            device=device, sample_steps=ddim_steps,
            eta=float(cfg.get("ddim_eta", 0.0)),
        )
        x0_inv_T = sample_ddim_teacher_grad(
            model=teacher, sample_scheduler=ddim, z=S_inv_z_seq[-1], 
            device=device, sample_steps=ddim_steps,
            eta=float(cfg.get("ddim_eta", 0.0)), t_sel=0,
        )

        ##################  RKD LOSSES ##################

        rkd_loss = torch.tensor(0.0, device=device)
        inversion_loss = torch.tensor(0.0, device=device)
        invinv_loss = torch.tensor(0.0, device=device)
        fid_loss = torch.tensor(0.0, device=device)

        diff_loss = torch.tensor(0.0, device=device)
        

        for i, (xt_S, xt_T, xt_S_inv, xt_T_inv) in enumerate(zip(xt_S_seq, xt_T_seq, reversed(S_inv_z_seq[:-1]), x0_inv_T)):


            xt_S = denormalize_torch(xt_S, mu_student_tensor, sigma_student_tensor)
            xt_T = denormalize_torch(xt_T, mu_teacher_tensor, sigma_teacher_tensor)
            xt_S_inv = denormalize_torch(xt_S_inv, mu_student_tensor, sigma_student_tensor)
            xt_T_inv = denormalize_torch(xt_T_inv, mu_teacher_tensor, sigma_teacher_tensor)

            # RKD
            rkd_s_d = torch.pdist(xt_S, p=2).clamp_min(1e-12)           
            rkd_t_d = torch.pdist(xt_T, p=2).clamp_min(1e-12)  
            # INV
            s_full = torch.cdist(xt_S, xt_S_inv, p=2)  
            t_full = torch.cdist(xt_T, xt_T_inv, p=2)  
            inv_s_d = s_full.reshape(-1).clamp_min(1e-12)            
            inv_t_d = t_full.reshape(-1).clamp_min(1e-12)
            # INVINV
            invinv_s_d = torch.pdist(xt_S_inv, p=2).clamp_min(1e-12)           
            invinv_t_d = torch.pdist(xt_T_inv, p=2).clamp_min(1e-12)  

            # mean normalization 
            # teacher mean
            teacher_sum = rkd_t_d.sum() + inv_t_d.sum() + invinv_t_d.sum()
            teacher_cnt = rkd_t_d.numel() + inv_t_d.numel() + invinv_t_d.numel()
            teacher_mean = teacher_sum / teacher_cnt  
            # student mean
            student_sum = rkd_s_d.sum() + inv_s_d.sum() + invinv_s_d.sum()
            student_cnt = rkd_s_d.numel() + inv_s_d.numel() + invinv_s_d.numel()
            student_mean = student_sum / student_cnt  

            rkd_s_d = rkd_s_d / student_mean
            rkd_t_d = rkd_t_d / teacher_mean
            inv_s_d = inv_s_d / student_mean
            inv_t_d = inv_t_d / teacher_mean
            invinv_s_d = invinv_s_d / student_mean
            invinv_t_d = invinv_t_d / teacher_mean

            # FID
            fid_student = fid_gaussian_torch(xt_S, xt_S_inv)
            fid_teacher = fid_gaussian_torch(xt_T, xt_T_inv) 

            # loss
            rkd_loss += cfg["W_RKD"] * F.mse_loss(rkd_s_d, rkd_t_d, reduction="mean") / len(xt_S_seq)
            inversion_loss += cfg["W_INV"] * F.mse_loss(inv_s_d, inv_t_d, reduction="mean") / len(xt_S_seq)
            invinv_loss += cfg["W_INVINV"] * F.mse_loss(invinv_s_d, invinv_t_d, reduction="mean") / len(xt_S_seq)
            fid_loss += cfg["W_FID"] * (fid_student + fid_teacher) / len(xt_S_seq)



        # # ===================== NEW: diffusion ε-MSE loss =====================
        # t_b_s = torch.randint(low=0, high=T, size=(x0_batch.shape[0],), device=device, dtype=torch.long)
        # eps = torch.randn_like(x0_batch)    
        # x_t_for_diff = train_sched.add_noise(x0_batch, eps, t_b_s)  # q(x_t|x0, ε, t)
        # eps_pred = student(x_t_for_diff, t_b_s)  # prediction_type='epsilon'

        # diff_loss += cfg["W_DIFF"] * F.mse_loss(eps_pred, eps, reduction="mean")
        # # ===============================


        ################## TOTAL LOSS ##################
        loss = rkd_loss + inversion_loss + invinv_loss + fid_loss + diff_loss


        opt.zero_grad()
        loss.backward()
        if cfg.get("max_grad_norm", 0) > 0:
            nn.utils.clip_grad_norm_(student.parameters(), cfg["max_grad_norm"])
        opt.step()

        # if (step_i % max(1, total_steps // 20) == 0) or (step_i == 1):
        if (step_i % 5 == 0) or (step_i == 1):
            print(f"[step {step_i:06d}] rkd={rkd_loss.item():.6f}  inv={inversion_loss.item():.6f}   invinv={invinv_loss.item():.6f}  fid_loss={fid_loss.item():.6f}  total={loss.item():.6f}")


        if cfg["use_wandb"]:
            wandb.log({
                "step": step_i,
                "t": t_sel,
                "loss/total": float(loss),
                "loss/rkd": float(rkd_loss),
                "loss/inv": float(inversion_loss),
                "loss/invinv": float(invinv_loss),
                "loss/fid": float(fid_loss),
                "loss/diff_loss": float(diff_loss),
                "lr": opt.param_groups[0]["lr"],
            }, step=step_i)


        if cfg["use_wandb"]:
            wandb.log({
                "loss_raw/rkd": float(rkd_loss) / cfg["W_RKD"] if cfg["W_RKD"] != 0 else 0.0,
                "loss_raw/inv": float(inversion_loss) / cfg["W_INV"] if cfg["W_INV"] != 0 else 0.0,
                "loss_raw/invinv": float(invinv_loss) / cfg["W_INVINV"] if cfg["W_INVINV"] != 0 else 0.0,
                "loss_raw/fid": float(fid_loss) / cfg["W_FID"] if cfg["W_FID"] != 0 else 0.0,
                "loss_raw/diff_loss": float(diff_loss) / cfg["W_DIFF"] if cfg["W_DIFF"] != 0 else 0.0
            }, step=step_i)

        # 7) (옵션) 시각화: 그대로 유지 (원 코드와 동일)
        if (step_i % cfg["vis_interval_epochs"] == 0) or (step_i == total_steps):
            with torch.no_grad():
                B_plot = min(int(cfg.get("n_vis", 1024)), B)
                z_vis  = sample_noise_batch(noise_pool, B_plot, device)

                seq_T, ts = collect_xt_seq_ddim(teacher, ddim, z_vis, t_stop=0, return_ts=True, device=device, sample_steps=int(cfg["rkd_ddim_steps_to_t"]))
                seq_S, _  = collect_xt_seq_ddim(student.eval(), ddim, z_vis, t_stop=0, return_ts=True, device=device, sample_steps=int(cfg["rkd_ddim_steps_to_t"]))

                stride = 1
                idxs = np.arange(0, len(ts), stride)
                seq_T_s = seq_T[idxs]
                seq_S_s = seq_S[idxs]
                ts_s    = ts[idxs]


                seq_T_s_plot = denormalize_np(seq_T_s, mu_teacher, sigma_teacher)
                seq_S_s_plot = denormalize_np(seq_S_s, mu_student, sigma_student)
                
                _ = save_and_log_xt_pairs_for_all_t(
                    left_seq   = seq_T_s_plot,
                    right_seq  = seq_S_s_plot,
                    ts         = ts_s,
                    noise_ids  = np.arange(B_plot),
                    out_dir    = out_dir,
                    step_i     = step_i,
                    sync_axes  = bool(cfg.get("vis_xt_sync_axes", False)),
                    dot_size   = 6,
                    use_wandb  = bool(cfg["use_wandb"]),
                )

        # 7) (옵션) 시각화: 랜덤 노이즈에서 학생 모델로 x0 샘플링하여 저장
        if (step_i % cfg["vis_interval_epochs"] == 0) or (step_i == total_steps):
            @torch.no_grad()
            def sample_x0_ddim(model, sample_scheduler, num_samples, device, sample_steps, dim=2, eta=0.0):
                sample_scheduler = DDIMScheduler.from_config(sample_scheduler.config)
                sample_scheduler.set_timesteps(sample_steps, device=device)                
                x = torch.randn(num_samples, dim, device=device)
                model.eval()
                for t in sample_scheduler.timesteps:      # [T-1, ..., 0]
                    t_b  = torch.full((num_samples,), int(t), device=device, dtype=torch.long)
                    x_in = sample_scheduler.scale_model_input(x, t)
                    eps  = model(x_in, t_b)
                    x    = sample_scheduler.step(model_output=eps, timestep=t, sample=x, eta=eta).prev_sample
                model.train()
                return x

            student.eval()

            B_plot = 8192
            x0_s = sample_x0_ddim(
                model=student,
                sample_scheduler=ddim,
                num_samples=B_plot,
                device=device,
                sample_steps=int(cfg.get("rkd_ddim_steps_to_t", 100)),
                dim=int(cfg.get("dim", 2)),
                eta=float(cfg.get("ddim_eta", 0.0)),
            )

            # 역정규화
            x0_s_plot = denormalize_np(x0_s.detach().cpu().numpy(), mu_student, sigma_student)

            # === 여기서 확실히 디렉토리 보장 ===
            figs_dir = Path(cfg["out_dir"]) / "figs"
            figs_dir.mkdir(parents=True, exist_ok=True)

            png_path = figs_dir / f"samples_step{step_i:06d}.png"

            # 실제 저장 (정사각형 축 범위 적용)
            plt.figure(figsize=(4, 4))
            plt.scatter(x0_s_plot[:, 0], x0_s_plot[:, 1], s=6, edgecolors="none")
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")
            xlim, ylim = _square_limits_from(x0_s_plot, pad_ratio=0.05)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            plt.title(f"Student samples (x0) @ step {step_i}")
            plt.tight_layout()
            plt.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close()

            if cfg["use_wandb"]:
                wandb.log({"img_T/student_samples": wandb.Image(str(png_path))}, step=step_i)


        # 8) (옵션) 주기적 체크포인트
        if (step_i % (cfg["vis_interval_epochs"]) == 0) or (step_i == total_steps):
            ckpt_path = out_dir / f"ckpt_student_step{step_i:06d}.pt"
            torch.save(student.state_dict(), ckpt_path)
            print("[CKPT]", ckpt_path)

    print("\n[DONE] Out dir:", out_dir.resolve())
    if cfg["use_wandb"]:
        wandb.finish()


# ===================== MAIN ===================== #

def main(cfg: Dict):
    set_seed(cfg["seed"])
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    train_student_uniform_xt(cfg)

if __name__ == "__main__":
    main(CONFIG)
