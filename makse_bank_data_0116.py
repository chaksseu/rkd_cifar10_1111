#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline Teacher bank (final x0 only, NO store_all_k):
Save per sample:
  - z: noise (B,3,H,W)
  - x0_final: teacher final denoised sample after full DDIM loop (B,3,H,W) (optional)
  - feats on x0_final:
      feats_pixel:  (B, 3*H*W)
      feats_clip:   (B, D_clip)
      feats_dinov3: (B, D_dino)

tqdm progress bars:
  - steps loop progress
  - per-steps saving progress (images)

Deps:
  pip install diffusers torch torchvision transformers tqdm
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import torch
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from transformers import CLIPModel, AutoModel, AutoImageProcessor

try:
    from tqdm import tqdm
except Exception as e:
    raise RuntimeError(f"tqdm import failed. Install via `pip install tqdm`. ({e})")


# ------------------------- utils -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

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
        return torch.device("cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev

def load_teacher_scheduler_or_fallback(teacher_dir: Path, train_timesteps: int, beta_schedule: str) -> DDPMScheduler:
    try:
        return DDPMScheduler.from_pretrained(teacher_dir.as_posix())
    except Exception:
        return DDPMScheduler(
            num_train_timesteps=train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type="epsilon",
        )

def make_ddim(ddpm: DDPMScheduler, prediction_type: str = "epsilon") -> DDIMScheduler:
    ddim = DDIMScheduler.from_config(ddpm.config)
    ddim.config.clip_sample = False
    ddim.config.prediction_type = prediction_type
    return ddim

def parse_steps_list(s: str) -> List[int]:
    s = s.strip()
    if len(s) == 0:
        return []
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if len(x.strip()) > 0]
    return [int(s)]


# ------------------------- embedders -------------------------

class ClipEmbedder(torch.nn.Module):
    """
    CLIP vision embeddings (pooler_output).
    Input x: (N,3,H,W) in [-1,1]
    """
    def __init__(self, model_name: str, device: torch.device):
        super().__init__()
        vision = CLIPModel.from_pretrained(model_name).vision_model.to(device)
        vision.eval()
        for p in vision.parameters():
            p.requires_grad = False
        self.net = vision

        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
        self.std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)
        self.target_size = (224, 224)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(-1, 1)
        x01 = (x + 1) * 0.5
        x_up = F.interpolate(x01, size=self.target_size, mode="bilinear",
                             align_corners=False, antialias=True)
        x_norm = (x_up - self.mean) / self.std
        out = self.net(pixel_values=x_norm)
        return out.pooler_output  # (N,D)


class DinoV3Embedder(torch.nn.Module):
    """
    DINOv3 embeddings (pooler_output if exists else CLS token).
    Input x: (N,3,H,W) in [-1,1]
    """
    def __init__(self, model_name: str, device: torch.device):
        super().__init__()
        try:
            proc = AutoImageProcessor.from_pretrained(model_name)
        except Exception as e:
            print(f"[Warn] AutoImageProcessor load failed ({e}). Use ImageNet mean/std and 224.", flush=True)
            proc = None

        net = AutoModel.from_pretrained(model_name).to(device)
        net.eval()
        for p in net.parameters():
            p.requires_grad = False
        self.net = net

        if proc is not None and hasattr(proc, "image_mean") and hasattr(proc, "image_std"):
            mean = proc.image_mean
            std = proc.image_std
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        self.mean = torch.tensor(mean, device=device).view(1,3,1,1)
        self.std  = torch.tensor(std,  device=device).view(1,3,1,1)

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

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(-1, 1)
        x01 = (x + 1) * 0.5
        x_up = F.interpolate(x01, size=self.target_size, mode="bilinear",
                             align_corners=False, antialias=True)
        x_norm = (x_up - self.mean) / self.std
        out = self.net(pixel_values=x_norm)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0, :]


# ------------------------- teacher sampling (final x0) -------------------------

@torch.no_grad()
def teacher_ddim_final_x0(
    teacher: UNet2DModel,
    ddim_base: DDIMScheduler,
    z: torch.Tensor,
    steps: int,
    eta: float,
    device: torch.device,
    use_amp: bool,
    amp_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    """
    Returns final denoised sample after full DDIM loop: x0_final (B,3,H,W)
    """
    local = DDIMScheduler.from_config(ddim_base.config)
    local.set_timesteps(steps, device=device)

    x = z.to(device)
    teacher.eval()

    from contextlib import nullcontext
    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
        if (use_amp and device.type == "cuda")
        else nullcontext()
    )

    with ctx:
        for t in local.timesteps:
            x_in = local.scale_model_input(x, t)
            eps = teacher(x_in, t).sample
            out = local.step(model_output=eps, timestep=t, sample=x, eta=eta)
            x = out.prev_sample

    return x  # final sample


@torch.no_grad()
def compute_feats_on_x0(
    x0: torch.Tensor,               # (B,3,H,W) on GPU
    feat_batch: int,
    want_pixel: bool,
    clip_emb: Optional[ClipEmbedder],
    dino_emb: Optional[DinoV3Embedder],
    use_amp: bool,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, torch.Tensor]:
    """
    Returns CPU tensors:
      feats_pixel, feats_clip, feats_dinov3 (as available)
    """
    B, C, H, W = x0.shape
    out: Dict[str, torch.Tensor] = {}

    if want_pixel:
        out["feats_pixel"] = x0.reshape(B, -1).detach().to("cpu")

    if (clip_emb is None) and (dino_emb is None):
        return out

    from contextlib import nullcontext
    ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
        if (use_amp and x0.is_cuda)
        else nullcontext()
    )

    def run_chunks(embedder, key: str):
        feats = []
        n = x0.shape[0]
        for s in range(0, n, feat_batch):
            e = min(n, s + feat_batch)
            xb = x0[s:e]
            with ctx:
                fb = embedder(xb)
            feats.append(fb.detach().to("cpu"))
        out[key] = torch.cat(feats, dim=0).contiguous()

    if clip_emb is not None:
        run_chunks(clip_emb, "feats_clip")
    if dino_emb is not None:
        run_chunks(dino_emb, "feats_dinov3")

    return out


# ------------------------- main -------------------------

def main():
    p = argparse.ArgumentParser("Make teacher (z, x0_final, pixel/clip/dino feats) bank for multiple steps")

    # teacher_dir: default set as requested
    p.add_argument(
        "--teacher_dir",
        type=str,
        default="ddpm_cifar10_rgb_T400_DDIM50/ckpt_step150000",
        help="Default teacher checkpoint path.",
    )
    p.add_argument("--out_dir", type=str, required=True)

    # steps control
    p.add_argument("--steps_list", type=str, default="", help="Comma list, e.g. '40,45,50'")
    p.add_argument("--steps_min", type=int, default=0, help="If steps_list empty, use range [min,max]")
    p.add_argument("--steps_max", type=int, default=0)

    p.add_argument("--num_samples_per_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=256)

    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--channels", type=int, default=3)

    p.add_argument("--train_timesteps", type=int, default=400)
    p.add_argument("--beta_schedule", type=str, default="linear")
    p.add_argument("--ddim_eta", type=float, default=0.0)

    # which features
    p.add_argument("--feat_pixel", action="store_true")
    p.add_argument("--feat_clip", action="store_true")
    p.add_argument("--feat_dinov3", action="store_true")

    p.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--dino_model_name", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")

    p.add_argument("--feat_batch", type=int, default=1024)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"],
                  help="Storage dtype for saved tensors/features.")
    p.add_argument("--use_amp", action="store_true", help="Use autocast for teacher+feature forward on GPU")
    p.add_argument("--save_feats_only", action="store_true",
                  help="If set, do NOT save x0_final (only save z + feats).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")

    args = p.parse_args()

    # default: store all 3 features if none specified
    if (not args.feat_pixel) and (not args.feat_clip) and (not args.feat_dinov3):
        args.feat_pixel = True
        args.feat_clip = True
        args.feat_dinov3 = True

    set_seed(args.seed)
    device = resolve_device(args.device)
    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    teacher_dir = Path(args.teacher_dir)
    if not teacher_dir.exists():
        raise FileNotFoundError(f"--teacher_dir not found: {teacher_dir.as_posix()}")

    # steps list
    steps_list = parse_steps_list(args.steps_list)
    if len(steps_list) == 0:
        if args.steps_min <= 0 or args.steps_max <= 0 or args.steps_max < args.steps_min:
            raise ValueError("Provide --steps_list or valid --steps_min/--steps_max.")
        steps_list = list(range(int(args.steps_min), int(args.steps_max) + 1))

    # dtype
    if args.dtype == "fp16":
        save_dtype = torch.float16
        amp_dtype = torch.float16
    elif args.dtype == "bf16":
        save_dtype = torch.bfloat16
        amp_dtype = torch.bfloat16
    else:
        save_dtype = torch.float32
        amp_dtype = torch.float16

    use_amp = bool(args.use_amp) and (device.type == "cuda")

    # meta
    meta = {
        "teacher_dir": args.teacher_dir,
        "steps_list": steps_list,
        "num_samples_per_steps": int(args.num_samples_per_steps),
        "batch_size": int(args.batch_size),
        "image_size": int(args.image_size),
        "channels": int(args.channels),
        "train_timesteps": int(args.train_timesteps),
        "beta_schedule": args.beta_schedule,
        "ddim_eta": float(args.ddim_eta),
        "feat_pixel": bool(args.feat_pixel),
        "feat_clip": bool(args.feat_clip),
        "feat_dinov3": bool(args.feat_dinov3),
        "clip_model_name": args.clip_model_name,
        "dino_model_name": args.dino_model_name,
        "feat_batch": int(args.feat_batch),
        "save_dtype": args.dtype,
        "use_amp": bool(use_amp),
        "save_feats_only": bool(args.save_feats_only),
        "seed": int(args.seed),
        "device": str(device),
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[Info] device={device} | use_amp={use_amp} | save_dtype={save_dtype}", flush=True)
    print(f"[Info] teacher_dir={teacher_dir.as_posix()}", flush=True)
    print(f"[Info] steps_list={steps_list} | num_samples_per_steps={args.num_samples_per_steps}", flush=True)

    # teacher + sched
    teacher = UNet2DModel.from_pretrained(teacher_dir.as_posix()).to(device)
    teacher.eval()
    for pp in teacher.parameters():
        pp.requires_grad = False

    ddpm = load_teacher_scheduler_or_fallback(teacher_dir, args.train_timesteps, args.beta_schedule)
    ddim_base = make_ddim(ddpm, prediction_type="epsilon")

    # embedders
    clip_emb = None
    dino_emb = None
    if args.feat_clip:
        print(f"[Embedder] Loading CLIP: {args.clip_model_name}", flush=True)
        clip_emb = ClipEmbedder(args.clip_model_name, device=device)
    if args.feat_dinov3:
        print(f"[Embedder] Loading DINOv3: {args.dino_model_name}", flush=True)
        dino_emb = DinoV3Embedder(args.dino_model_name, device=device)

    # deterministic generator
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))

    steps_pbar = tqdm(steps_list, desc="DDIM steps", unit="steps")

    for steps in steps_pbar:
        sub = out_root / f"ddim_steps{steps:03d}_eta{args.ddim_eta:g}"
        shard_dir = sub / "shards"
        ensure_dir(shard_dir)

        info = {"steps": int(steps), "ddim_eta": float(args.ddim_eta)}
        (sub / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

        total = int(args.num_samples_per_steps)
        bs = int(args.batch_size)
        num_shards = math.ceil(total / bs)

        steps_pbar.set_postfix_str(f"steps={steps} total={total} bs={bs}")

        img_pbar = tqdm(total=total, desc=f"Save (steps={steps})", unit="img", leave=False)

        produced = 0
        for shard_idx in range(num_shards):
            cur = min(bs, total - produced)
            if cur <= 0:
                break

            # noise
            z = torch.randn(
                (cur, args.channels, args.image_size, args.image_size),
                device=device,
                dtype=torch.float32,
                generator=gen,
            )

            # teacher final x0
            x0_final = teacher_ddim_final_x0(
                teacher=teacher,
                ddim_base=ddim_base,
                z=z,
                steps=int(steps),
                eta=float(args.ddim_eta),
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

            # features on x0_final
            feats = compute_feats_on_x0(
                x0=x0_final,
                feat_batch=int(args.feat_batch),
                want_pixel=bool(args.feat_pixel),
                clip_emb=clip_emb,
                dino_emb=dino_emb,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

            payload = {
                "start_index": int(produced),
                "count": int(cur),
                "steps": int(steps),
                "ddim_eta": float(args.ddim_eta),
                "z": z.detach().to("cpu", dtype=save_dtype),
            }

            if not args.save_feats_only:
                payload["x0_final"] = x0_final.detach().to("cpu", dtype=save_dtype)

            for kk, vv in feats.items():
                payload[kk] = vv.to(dtype=save_dtype)

            shard_path = shard_dir / f"shard_{shard_idx:06d}.pt"
            torch.save(payload, shard_path)

            produced += cur
            img_pbar.update(cur)
            img_pbar.set_postfix_str(f"shard={shard_idx} saved={shard_path.name}")

            # cleanup
            del z, x0_final, feats, payload
            if device.type == "cuda":
                torch.cuda.empty_cache()

        img_pbar.close()

    steps_pbar.close()
    print(f"[All Done] saved to {out_root.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
