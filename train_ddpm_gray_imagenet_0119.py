#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stable Diffusion 1.5 LoRA fine-tuning with plain diffusion loss (DDPM epsilon objective)

What this does
- Loads SD1.5 (tokenizer/text_encoder/vae/unet)
- Freezes everything except UNet LoRA weights
- Training objective (standard diffusion):
    1) Encode GT image -> VAE latent (scaled)
    2) Sample timestep t
    3) Add noise: z_t = q(z_t | z_0)
    4) UNet predicts epsilon from (z_t, t, text_cond)
    5) MSE(eps_pred, eps_gt)

Eval
- Periodic DDIM sampling (conditional on folder-name prompts)
- Optional FID via pytorch-fid (same style as your reference code)

Deps
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
from PIL import Image

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from peft import LoraConfig, get_peft_model, PeftModel

from transformers import CLIPTokenizer, CLIPTextModel


# ------------------------- Utils -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())

def set_seed(seed: int):
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

def get_model_dtype(args, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if args.mixed_precision == "fp16":
        return torch.float16
    if args.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32

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

def flatten_real_cache(
    test_dir: Path,
    cache_dir: Path,
    use_symlink: bool = False,
    do_preprocess: bool = False,
    image_size: int = 256,
) -> int:
    ensure_dir(cache_dir)
    existing = list(cache_dir.glob("*"))
    if len(existing) > 0:
        return len(existing)

    paths = collect_image_paths_recursive(test_dir)
    print(f"[FID] Flattening test set ({len(paths)} imgs) to {cache_dir} ...", flush=True)

    if do_preprocess:
        fid_tf = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(image_size),
        ])

    for i, src in enumerate(paths, 1):
        if do_preprocess:
            dst = cache_dir / f"real_{i:06d}.png"
            with Image.open(src) as img:
                img = img.convert("RGB")
                img = fid_tf(img)
                img.save(dst)
        else:
            dst = cache_dir / f"real_{i:06d}{src.suffix.lower()}"
            try:
                if use_symlink:
                    os.symlink(src.resolve(), dst)
                else:
                    shutil.copy2(src, dst)
            except Exception:
                shutil.copy2(src, dst)

    return len(paths)

def compute_fid_pytorch_fid(real_dir: Path, gen_dir: Path, device: torch.device, batch_size: int, dims: int) -> float:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    fid = calculate_fid_given_paths(
        [real_dir.as_posix(), gen_dir.as_posix()],
        batch_size=batch_size,
        device=device,
        dims=dims,
    )
    return float(fid)


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
    def __init__(
        self,
        root: str,
        image_size: int = 256,
        split: str = "train",  # "train" or "eval"
        horizontal_flip: bool = True,
        rrc_scale: Tuple[float, float] = (0.8, 1.0),
        rrc_ratio: Tuple[float, float] = (3/4, 4/3),
    ):
        self.root = Path(root)
        exts = {".png", ".jpg", ".jpeg"}
        self.files: List[Path] = []
        for p in self.root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                self.files.append(p)
        if len(self.files) == 0:
            raise FileNotFoundError(f"No images found under {self.root}!")

        self.class_names = sorted({p.parent.name for p in self.files})

        if split == "train":
            tfms = [
                T.RandomResizedCrop(
                    image_size,
                    scale=rrc_scale,
                    ratio=rrc_ratio,
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                )
            ]
            if horizontal_flip:
                tfms.append(T.RandomHorizontalFlip(p=0.5))
            tfms.append(T.ToTensor())
        else:
            tfms = [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]

        self.to_tensor = T.Compose(tfms)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        prompt = path.parent.name
        with Image.open(path) as img:
            img = img.convert("RGB")
            x01 = self.to_tensor(img)
        x = x01 * 2.0 - 1.0
        return x, prompt

def collate_image_text(batch: List[Tuple[torch.Tensor, str]]):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    texts = [b[1] for b in batch]
    return imgs, texts


# ------------------------- Text cond cache -------------------------

class TextCondCache:
    """
    Cache text encoder outputs per unique prompt string.
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
            self.cache[key] = emb
            out_list.append(emb)

        return torch.cat(out_list, dim=0)


# ------------------------- VAE encode / decode -------------------------

@torch.no_grad()
def encode_images_to_latents(vae: AutoencoderKL, images: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    vae_param = next(vae.parameters())
    images = images.to(device=vae_param.device, dtype=vae_param.dtype)
    latents = vae.encode(images).latent_dist.mean
    return latents * scaling_factor

@torch.no_grad()
def decode_latents_to_images(vae: AutoencoderKL, latents: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    vae_param = next(vae.parameters())
    latents = latents.to(device=vae_param.device, dtype=vae_param.dtype)
    imgs = vae.decode(latents / scaling_factor).sample
    return imgs


# ------------------------- Sampling (DDIM) for eval -------------------------

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


# ------------------------- Train (DDPM loss) -------------------------

def train(args):
    torch.backends.cudnn.benchmark = True
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    set_seed(args.seed)
    device = resolve_device(args.device)
    model_dtype = get_model_dtype(args, device)

    if device.type == "cuda":
        torch.cuda.set_device(device)

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

    # dataset
    dataset = ImageTextFolderDataset(
        args.student_data_dir,
        image_size=args.image_size,
        split="train",
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
            num_test_imgs_all = flatten_real_cache(
                test_dir,
                fid_real_all_dir,
                use_symlink=False,
                do_preprocess=args.fid_preprocess,
                image_size=args.image_size,
            )
            print(f"[FID] Real cache: N={num_test_imgs_all} @ {fid_real_all_dir}", flush=True)
        else:
            args.disable_fid = True
    else:
        args.disable_fid = True

    # load SD
    print(f"[Info] Loading SD model: {args.sd_model_id}", flush=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.sd_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.sd_model_id, subfolder="text_encoder", torch_dtype=model_dtype
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        args.sd_model_id, subfolder="vae", torch_dtype=model_dtype
    ).to(device)

    vae_scaling = float(getattr(vae.config, "scaling_factor", args.vae_scaling_factor))

    # Freeze text encoder / VAE
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # UNet base -> LoRA only trainable
    base_unet = UNet2DConditionModel.from_pretrained(
        args.sd_model_id, subfolder="unet", torch_dtype=model_dtype
    ).to(device)
    base_unet.requires_grad_(False)

    # Resume or init LoRA
    global_step = 0
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"[Info] Resuming from: {args.resume_checkpoint}", flush=True)
        student = PeftModel.from_pretrained(base_unet, args.resume_checkpoint, is_trainable=True)
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
            target_modules=args.lora_targets.split(","),
            init_lora_weights=args.lora_init,
        )
        student = get_peft_model(base_unet, lora_config)

    student.print_trainable_parameters()
    student.train()

    print(f"[Info] Base UNet params: {count_parameters(base_unet):,}", flush=True)
    print(f"[Info] Student (LoRA-wrapped) params: {count_parameters(student):,}", flush=True)

    # DDPM scheduler for training
    ddpm = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
    )
    # DDIM scheduler for evaluation sampling
    ddim = DDIMScheduler.from_pretrained(args.sd_model_id, subfolder="scheduler")
    ddim.config.clip_sample = False
    ddim.config.prediction_type = args.prediction_type

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad(set_to_none=True)

    text_cache = TextCondCache()

    # Step-based stopping is usually easier than huge epochs
    max_steps = int(args.max_train_steps) if args.max_train_steps > 0 else None

    epoch = 0
    while True:
        epoch += 1
        student.train()
        print(f"[Epoch {epoch}] start (Global Step: {global_step})", flush=True)

        for it, (x0_real_img, prompts) in enumerate(loader, start=1):
            x0_real_img = x0_real_img.to(device, non_blocking=True)
            B = x0_real_img.shape[0]

            # Text cond (no grad)
            cond_dtype = next(student.parameters()).dtype
            with torch.no_grad():
                cond = text_cache.get(tokenizer, text_encoder, prompts, device=device, dtype=cond_dtype)

            # Encode to latents (no grad; VAE frozen)
            with torch.no_grad():
                x0_lat = encode_images_to_latents(vae, x0_real_img, scaling_factor=vae_scaling).to(dtype=cond_dtype)

            # Sample timesteps and noise
            timesteps = torch.randint(
                0, ddpm.config.num_train_timesteps, (B,), device=device, dtype=torch.long
            )
            noise = torch.randn_like(x0_lat)
            zt = ddpm.add_noise(x0_lat, noise, timesteps)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                if (use_amp and device.type == "cuda")
                else nullcontext()
            )

            with autocast_ctx:
                eps_pred = student(zt, timesteps, encoder_hidden_states=cond).sample

                if args.prediction_type == "epsilon":
                    target = noise
                elif args.prediction_type == "v_prediction":
                    # v = alpha_t * eps - sigma_t * x0  (diffusers provides helper)
                    target = ddpm.get_velocity(x0_lat, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported prediction_type: {args.prediction_type}")

                loss = F.mse_loss(eps_pred.float(), target.float(), reduction="mean")

                # optional: simple latent-scale normalization (rarely needed)
                if args.loss_scale != 1.0:
                    loss = loss * float(args.loss_scale)

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

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

            # logging
            if global_step % args.log_interval == 0:
                line = f"[Train] step={global_step:08d} loss={float(loss.detach().item()):.6f}"
                print(line, flush=True)
                with summary_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                if wandb_run is not None and wandb is not None:
                    try:
                        wandb.log(
                            {"loss/ddpm": float(loss.detach().item()), "train/epoch": int(epoch), "train/step": int(global_step)},
                            step=global_step,
                        )
                    except Exception:
                        pass

            # eval samples / fid
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

            # save
            if args.save_interval > 0 and (global_step % args.save_interval == 0):
                save_dir = out_dir / "ckpts" / f"ckpt_step{global_step:06d}"
                ensure_dir(save_dir)
                student.save_pretrained(save_dir.as_posix())
                # save schedulers for reproducibility
                ddpm.save_pretrained(save_dir.as_posix())
                ddim.save_pretrained(save_dir.as_posix())
                with (save_dir / "base_model.txt").open("w", encoding="utf-8") as f:
                    f.write(args.sd_model_id + "\n")
                print(f"[CKPT] Saved LoRA to {save_dir}", flush=True)

            # stopping
            if max_steps is not None and global_step >= max_steps:
                print(f"[Done] Reached max_train_steps={max_steps}.", flush=True)
                return


# ------------------------- Args (DDPM LoRA baseline; defaults aligned to your KD script style) -------------------------

DATE = "0118"
BATCH_SIZE = 32
CUDA_NUM = 6
LR = 1e-5

# keep the same dataset convention you used
N_IMAGES = 1
CLASS_PCT = 50

def build_argparser():
    p = argparse.ArgumentParser("SD1.5 LoRA training with plain DDPM diffusion loss (image+text; prompt=folder name)")

    # SD
    p.add_argument("--sd_model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--vae_scaling_factor", type=float, default=0.18215)
    p.add_argument("--fallback_prompt", type=str, default="")  # if no class names found

    # resume
    p.add_argument("--resume_checkpoint", type=str, default="")

    # data (same train/eval dirs as your KD defaults)
    p.add_argument(
        "--student_data_dir",
        type=str,
        default=f"/workspace/rkd_cifar10_1111/imagenet1k_export/gray3_subset_class{CLASS_PCT}pct_per{N_IMAGES}_seed0/train",
    ) # gray3_subset_per10
    p.add_argument("--test_dir", type=str, default="/workspace/rkd_cifar10_1111/imagenet1k_export/gray3/val_256cc")
    p.add_argument(
        "--output_dir",
        type=str,
        default=f"{DATE}_sd_lora_ddpm_gray-imagenet/"
                f"ddpm-loss-B{BATCH_SIZE}-LR{LR}_class{CLASS_PCT}pct_per{N_IMAGES}",
    )

    # device / logging identity (match your style)
    p.add_argument("--device", type=str, default=f"cuda:{CUDA_NUM}")
    p.add_argument("--project", type=str, default=f"{DATE}_sd15-ddpm-lora")
    p.add_argument(
        "--run_name",
        type=str,
        default=f"sd15-lora-ddpm-gray-imagenet-"
                f"B{BATCH_SIZE}-LR{LR}_class{CLASS_PCT}pct_per{N_IMAGES}",
    )
    p.add_argument("--wandb_offline", action="store_true")
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    # image
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--center_crop", action="store_true")  # kept for interface compatibility (not strictly needed)
    p.add_argument("--no_hflip", action="store_true")
    p.add_argument("--num_workers", type=int, default=8)

    # train (keep your epoch convention, but DDPM script may also support max_train_steps separately if you added it)
    p.add_argument("--epochs", type=int, default=1000000)
    p.add_argument("--real_batch", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    # DDPM training objective configs (new, but stable defaults)
    p.add_argument("--num_train_timesteps", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=0.00085)
    p.add_argument("--beta_end", type=float, default=0.012)
    p.add_argument("--beta_schedule", type=str, default="scaled_linear",
                   choices=["linear", "scaled_linear", "squaredcos_cap_v2"])
    p.add_argument("--prediction_type", type=str, default="epsilon", choices=["epsilon", "v_prediction"])
    p.add_argument("--loss_scale", type=float, default=1.0)

    # LoRA (same defaults)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_targets", type=str, default="to_q,to_k,to_v,to_out.0")
    p.add_argument("--lora_init", type=str, default="gaussian", choices=["gaussian", "default"])

    # logging / eval (same cadence)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=250)
    p.add_argument("--sample_interval", type=int, default=250)
    p.add_argument("--sample_n", type=int, default=25)
    p.add_argument("--sample_steps", type=int, default=20)
    p.add_argument("--sample_eta", type=float, default=0.0)

    # fid (same defaults)
    p.add_argument("--disable_fid", action="store_true")
    p.add_argument("--fid_batch_size", type=int, default=32)
    p.add_argument("--fid_gen_batch", type=int, default=128)
    p.add_argument("--fid_dims", type=int, default=2048)
    p.add_argument("--fid_keep_gen", action="store_true")
    p.add_argument("--fid_num_samples", type=int, default=0)
    p.add_argument("--fid_no_symlink", action="store_true")  # interface compatibility

    return p



if __name__ == "__main__":
    args = build_argparser().parse_args()
    ensure_dir(Path(args.output_dir))
    train(args)
