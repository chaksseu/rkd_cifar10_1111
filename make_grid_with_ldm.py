import os
import argparse
import math
import torch
import torch.nn as nn
import torchvision.utils as vutils
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Diffusers
from diffusers import UNet2DModel, DDIMScheduler, AutoencoderKL

# --------------------------------------------------------------------------
# [설정] 경로 및 하이퍼파라미터
# --------------------------------------------------------------------------
DEFAULT_VAE_PATH = "1227_b64_lr0.0001_MSE_klW_1e-08_block_64_128-checkpoint-1000000"

# 비교할 LDM 모델 경로 리스트
DEFAULT_MODELS = [
    "1228_cifar10_unet_64_128_b256_lr0.0001_rgb_unet_step150000",
    "out_1228_rkd_pixel_LDM_feature_cifar10_rgb_to_gray_single_batch8_N100_LR1e-05-FD-rkdW0.1-invW0.1-invinvW1.0-fdW0.0001-sameW0.1-teacher-init-eps/ckpt_step005000",
    "out_1228_rkd_pixel_LDM_feature_cifar10_rgb_to_gray_single_batch8_N100_LR1e-05-FD-rkdW0.1-invW0.1-invinvW1.0-fdW1e-05-sameW0.1-teacher-init-eps/ckpt_step005000",
]

DEFAULT_DEVICE = "cuda:0"
DEFAULT_OUTPUT_DIR = "1229_LDM_comparison_grids"
DEFAULT_IMAGE_SIZE = 32
DEFAULT_NUM_IMAGES = 36  # 6x6 Grid
DEFAULT_STEPS = 50
DEFAULT_SEED = 42

# 학습 코드 설정에 맞춘 Scale Factor (User Custom: 1 / 2.4774)
DEFAULT_LATENT_SCALE_FACTOR = 1.0 / 2.4774
# --------------------------------------------------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def to_grid(images: torch.Tensor, nrow: int) -> Image.Image:
    """
    Tensor Batch [-1, 1] -> PIL Image Grid [0, 255]
    """
    imgs = (images.clamp(-1, 1) + 1) / 2.0
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2)
    grid = (grid * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(grid)

@torch.no_grad()
def sample_ldm(
    unet: UNet2DModel,
    vae: AutoencoderKL,
    scheduler: DDIMScheduler,
    latent_noise: torch.Tensor,
    scale_factor: float,
    steps: int,
    eta: float = 0.0,
    device: torch.device = torch.device("cuda")
):
    """
    Latent Diffusion Sampling:
    1. Latent Noise -> Denoise (UNet)
    2. Rescale Latent (z / scale_factor)
    3. Decode (VAE) -> Pixel
    """
    unet.eval()
    vae.eval()
    
    scheduler.set_timesteps(steps, device=device)
    
    # 1. Latent Space Sampling
    # shared_noise를 직접 수정하지 않도록 clone 필수
    latents = latent_noise.clone().to(device)

    for t in tqdm(scheduler.timesteps, desc="LDM Sampling", leave=False):
        # Scale input (if needed by scheduler, usually 1 for DDIM)
        latent_model_input = scheduler.scale_model_input(latents, t)
        
        # Predict noise/epsilon
        noise_pred = unet(latent_model_input, t).sample
        
        # Step
        latents = scheduler.step(noise_pred, t, latents, eta=eta).prev_sample

    # 2. Rescale Latent back to VAE space
    latents = latents / scale_factor

    # 3. Decode to Pixels
    images = vae.decode(latents).sample
    
    return images

def main(args):
    # Device 설정
    if torch.cuda.is_available():
        device = torch.device(args.device)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    
    print(f"[Info] Device: {device}")
    
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    # Grid 계산 (nrow)
    nrow = int(math.ceil(math.sqrt(args.num_images)))
    print(f"[Info] Grid Layout: {args.num_images} images -> {nrow}x{nrow}")

    # -----------------------------------------------------------
    # 1. Load VAE (Frozen) - Latent Shape 계산을 위해 먼저 로드
    # -----------------------------------------------------------
    print(f"[Info] Loading VAE from {args.vae_path} ...")
    try:
        vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
        vae.eval()
        vae.requires_grad_(False)
    except Exception as e:
        print(f"[Error] Failed to load VAE: {e}")
        return

    # -----------------------------------------------------------
    # 2. Determine Latent Shape & Generate Shared Noise
    # -----------------------------------------------------------
    # Dummy encoding to find latent shape (C, H, W)
    with torch.no_grad():
        dummy_img = torch.zeros(1, 3, args.image_size, args.image_size).to(device)
        dummy_dist = vae.encode(dummy_img).latent_dist
        dummy_z = dummy_dist.sample()
        latent_shape = dummy_z.shape[1:] # e.g., (4, 4, 4) for 32x32 img
    
    print(f"[Info] Latent Shape determined: {latent_shape}")

    # Shared Latent Noise 생성
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    
    print(f"[Info] Generating shared LATENT noise for {args.num_images} images...")
    shared_latent_noise = torch.randn(
        (args.num_images, *latent_shape),
        generator=generator,
        device=device
    )

    # -----------------------------------------------------------
    # 3. Load Models & Sample
    # -----------------------------------------------------------
    for idx, model_path in enumerate(args.model_paths):
        if not os.path.exists(model_path):
            print(f"[Skip] Path not found: {model_path}")
            continue

        print(f"\n[Model #{idx+1}] Loading UNet from {model_path} ...")
        
        try:
            # Load UNet
            # 1) subfolder='unet' 시도
            try:
                unet = UNet2DModel.from_pretrained(model_path, subfolder="unet").to(device)
            except:
                # 2) 직접 로드 시도
                unet = UNet2DModel.from_pretrained(model_path).to(device)
            
            # Load Scheduler
            # config.json이 있으면 로드, 없으면 기본값 사용
            try:
                scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
            except:
                # Fallback: 학습 코드의 설정(timesteps=400)에 맞춰 기본 생성
                print(f"  [Warning] Scheduler config not found. Using default DDIM (T=400).")
                scheduler = DDIMScheduler(
                    num_train_timesteps=400,   # 학습 코드와 동일하게 400으로 설정
                    beta_schedule="linear", 
                    prediction_type="epsilon",
                    clip_sample=False
                )
            
            # LDM Sampling
            generated_imgs = sample_ldm(
                unet=unet,
                vae=vae,
                scheduler=scheduler,
                latent_noise=shared_latent_noise,
                scale_factor=args.latent_scale_factor,
                steps=args.steps,
                eta=args.eta,
                device=device
            )

            # Save Grid
            path_obj = Path(model_path)
            # 이름 다듬기: 상위폴더_현재폴더 조합
            model_name = path_obj.name
            if model_name in ["unet", "checkpoint", "ckpt"] or model_name.startswith("ckpt_"):
                model_name = f"{path_obj.parent.name}_{model_name}"
            
            save_name = f"grid_{idx+1:02d}_{model_name}.png"
            save_path = out_dir / save_name
            
            grid_img = to_grid(generated_imgs, nrow=nrow)
            grid_img.save(save_path)
            print(f"[Done] Saved to: {save_path}")

        except Exception as e:
            print(f"[Error] Failed to process model {model_path}: {e}")
            import traceback
            traceback.print_exc()

    print("\n[All Done]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LDM Grid Generation with Shared Latent Noise")
    
    parser.add_argument("--vae_path", type=str, default=DEFAULT_VAE_PATH, help="Path to Pretrained VAE")
    parser.add_argument("--model_paths", type=str, nargs='+', default=DEFAULT_MODELS, help="List of LDM model paths")
    
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE, help="Pixel image size")
    parser.add_argument("--num_images", type=int, default=DEFAULT_NUM_IMAGES)
    
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="DDIM sampling steps")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    
    # 중요: 학습 코드와 동일하게 맞춰야 함 (1 / 2.4774)
    parser.add_argument("--latent_scale_factor", type=float, default=DEFAULT_LATENT_SCALE_FACTOR, 
                        help="Factor to scale latents before decoding")

    args = parser.parse_args()
    main(args)