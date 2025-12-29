import os
import argparse
import math
import torch
import torchvision.utils as vutils
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Diffusers & PEFT
from diffusers import UNet2DModel, DDIMScheduler
from peft import PeftModel, PeftConfig

# --------------------------------------------------------------------------
# [설정] 경로 및 하이퍼파라미터
# --------------------------------------------------------------------------
# 1. Base Model (LoRA가 붙을 원본 모델 - Teacher 혹은 Pretrained Base)
# LoRA 학습 시 --pretrained_model_path 로 사용했던 경로여야 합니다.
DEFAULT_BASE_MODEL = "ddpm_cifar10_rgb_T400_DDIM50/ckpt_step150000"

# 2. 평가할 모델 리스트 (일반 UNet 폴더와 LoRA 폴더를 섞어서 넣어도 됩니다)

DEFAULT_MODELS = [
    "ddpm_LoRA_gray_r32_a32_cifar10_rgb_T400_DDIM50-b256-lr1e-05-N100/lora_step025000",
    "ddpm_LoRA_gray_r32_a32_cifar10_rgb_T400_DDIM50-b256-lr1e-05-ALL/lora_step115000",
]
DEFAULT_DEVICE = "cuda:1"
DEFAULT_OUTPUT_DIR = "1229_DDPM_LoRA_comparison_grids"
DEFAULT_IMAGE_SIZE = 32
DEFAULT_NUM_IMAGES = 36  # 6x6 Grid
DEFAULT_STEPS = 50
DEFAULT_SEED = 42
# --------------------------------------------------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def to_grid(images: torch.Tensor, nrow: int) -> Image.Image:
    """Tensor Batch [-1, 1] -> PIL Image Grid [0, 255]"""
    imgs = (images.clamp(-1, 1) + 1) / 2.0
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2)
    grid = (grid * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(grid)

@torch.no_grad()
def sample_ddim(
    model: torch.nn.Module,
    scheduler: DDIMScheduler,
    noise: torch.Tensor,
    steps: int,
    eta: float = 0.0,
    device: torch.device = torch.device("cuda")
):
    model.eval()
    scheduler.set_timesteps(steps, device=device)
    
    # 초기 노이즈 복사 (공유 노이즈 사용)
    image = noise.clone().to(device)

    for t in tqdm(scheduler.timesteps, desc="Sampling", leave=False):
        # Scale input
        model_input = scheduler.scale_model_input(image, t)
        
        # Predict noise
        # LoRA 모델(PeftModel)도 forward 사용법은 동일합니다.
        model_output = model(model_input, t).sample
        
        # Step
        output = scheduler.step(model_output, t, image, eta=eta)
        image = output.prev_sample

    return image

def load_model_smart(path: str, base_path: str, device: torch.device):
    """
    경로를 확인하여 LoRA 모델인지 일반 UNet인지 자동으로 판별하여 로드합니다.
    """
    path_obj = Path(path)
    
    # 1. Scheduler 로드 (LoRA 폴더에도 scheduler가 저장되어 있으면 거기서 로드, 아니면 Base에서)
    try:
        scheduler = DDIMScheduler.from_pretrained(path, subfolder="scheduler")
    except:
        try:
            scheduler = DDIMScheduler.from_pretrained(path)
        except:
            print(f"  [Info] Scheduler not found in {path}, loading from Base {base_path}")
            scheduler = DDIMScheduler.from_pretrained(base_path, subfolder="scheduler")

    # 2. 모델 로드 로직
    # LoRA 여부 판별: folder 내에 'adapter_config.json'이 있는지 확인
    is_lora = (path_obj / "adapter_config.json").exists() or (path_obj / "adapter_model.bin").exists() or (path_obj / "adapter_model.safetensors").exists()
    
    if is_lora:
        print(f"  [Type] Detected LoRA Adapter")
        print(f"  [Info] Loading Base UNet: {base_path}")
        # Base Model 로드
        base_model = UNet2DModel.from_pretrained(base_path).to(device)
        
        # LoRA Adapter 병합
        print(f"  [Info] Loading LoRA Weights: {path}")
        model = PeftModel.from_pretrained(base_model, path)
        model.to(device)
    else:
        print(f"  [Type] Detected Standard UNet")
        try:
            model = UNet2DModel.from_pretrained(path, subfolder="unet").to(device)
        except:
            model = UNet2DModel.from_pretrained(path).to(device)
    
    model.eval()
    return model, scheduler

def main(args):
    # Device 설정
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")
    
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    # Grid 계산 (nrow)
    nrow = int(math.ceil(math.sqrt(args.num_images)))
    print(f"[Info] Grid Layout: {args.num_images} images -> {nrow}x{nrow}")

    # 1. 공통 노이즈 생성 (Shared Noise)
    # LoRA 비교 시 동일한 시드/노이즈를 사용하는 것이 중요합니다.
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    
    print(f"[Info] Generating shared noise for {args.num_images} images...")
    shared_noise = torch.randn(
        (args.num_images, 3, args.image_size, args.image_size),
        generator=generator,
        device=device
    )

    # 2. Base Model (Teacher) 먼저 샘플링 (비교 기준)
    if args.base_model_path and os.path.exists(args.base_model_path):
        print(f"\n[Base/Teacher] {args.base_model_path}")
        try:
            # Base는 보통 LoRA가 아닌 일반 UNet
            t_model, t_scheduler = load_model_smart(args.base_model_path, args.base_model_path, device)
            
            gen_t = sample_ddim(
                t_model, t_scheduler, shared_noise, 
                steps=args.steps, eta=args.eta, device=device
            )
            
            save_path = out_dir / "00_base_teacher.png"
            to_grid(gen_t, nrow=nrow).save(save_path)
            print(f"  [Done] Saved to: {save_path}")
            
            # 메모리 정리
            del t_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  [Error] Failed Base Model: {e}")

    # 3. Student / LoRA Models 샘플링
    for idx, model_path in enumerate(args.model_paths):
        if not os.path.exists(model_path):
            print(f"\n[Skip] Path not found: {model_path}")
            continue

        print(f"\n[Model #{idx+1}] {model_path}")
        try:
            # 모델 로드 (LoRA 자동 감지)
            model, scheduler = load_model_smart(model_path, args.base_model_path, device)
            
            # 샘플링
            gen_img = sample_ddim(
                model, scheduler, shared_noise,
                steps=args.steps, eta=args.eta, device=device
            )
            
            # 저장 파일명 생성
            path_obj = Path(model_path)
            model_name = path_obj.name
            # ckpt_step... 또는 lora_step... 인 경우 상위 폴더명까지 포함
            if model_name.startswith("ckpt_") or model_name.startswith("lora_") or model_name in ["unet", "final"]:
                model_name = f"{path_obj.parent.name}_{model_name}"
            
            save_name = f"model_{idx+1:02d}_{model_name}.png"
            save_path = out_dir / save_name
            
            to_grid(gen_img, nrow=nrow).save(save_path)
            print(f"  [Done] Saved to: {save_path}")
            
            # 메모리 정리
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [Error] Failed to process {model_path}: {e}")
            import traceback
            traceback.print_exc()

    print("\n[All Done]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM + LoRA Comparison Grid Generator")
    
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL, 
                        help="Path to the Base UNet (required for loading LoRA adapters)")
    
    parser.add_argument("--model_paths", type=str, nargs='+', default=DEFAULT_MODELS, 
                        help="List of model paths (can be full UNets or LoRA adapter folders)")
    
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--num_images", type=int, default=DEFAULT_NUM_IMAGES)
    
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="DDIM sampling steps")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    args = parser.parse_args()
    main(args)