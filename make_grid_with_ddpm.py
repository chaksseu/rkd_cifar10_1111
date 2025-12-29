import os
import argparse
import torch
import torchvision.utils as vutils
from PIL import Image
from pathlib import Path
from diffusers import UNet2DModel, DDIMScheduler
from tqdm import tqdm
import math  # [추가] 제곱근 계산을 위해 추가

# --------------------------------------------------------------------------
# [기본값 설정] train.py의 설정을 기반으로 기본값 지정
# --------------------------------------------------------------------------
DEFAULT_TEACHER = "ddpm_cifar10_rgb_T400_DDIM50/ckpt_step150000"
DEFAULT_STUDENT = [
    "out_1228_rkd/rkd_pixel_feature_cifar10_rgb_to_gray_single_batch8_N100_LR1e-05-FD-rkdW0.1-invW0.1-invinvW1.0-fdW1e-06-sameW0.1-teacher-init-eps/ckpts/ckpt_step004000",
]
DEFAULT_DEVICE = "cuda:0"
DEFAULT_OUTPUT_DIR = "1229_comparison_grids_teacher_RKD_pixel_feature_teacher_init"
DEFAULT_IMAGE_SIZE = 32
DEFAULT_NUM_IMAGES = 36  # [변경] 정사각형(6x6)을 위해 36으로 설정
DEFAULT_STEPS = 50
DEFAULT_SEED = 42
# --------------------------------------------------------------------------

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def to_grid(images: torch.Tensor, nrow: int) -> Image.Image:
    """텐서 배치를 그리드 이미지로 변환"""
    imgs = (images.clamp(-1, 1) + 1) / 2.0
    grid = vutils.make_grid(imgs, nrow=nrow, padding=2)
    grid = (grid * 255.0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(grid)

@torch.no_grad()
def sample_ddim(
    model: UNet2DModel,
    scheduler: DDIMScheduler,
    noise: torch.Tensor,
    steps: int,
    eta: float = 0.0,
    device: torch.device = torch.device("cuda")
):
    model.eval()
    scheduler.set_timesteps(steps, device=device)
    
    # 초기 노이즈 복사
    image = noise.clone().to(device)

    for t in tqdm(scheduler.timesteps, desc="Sampling", leave=False):
        model_input = scheduler.scale_model_input(image, t)
        model_output = model(model_input, t).sample
        output = scheduler.step(model_output, t, image, eta=eta)
        image = output.prev_sample

    return image

def main(args):
    # Device 설정
    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
        
    print(f"[Info] Device: {device}")
    
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    # [수정] 정사각형 그리드를 위해 nrow 자동 계산
    # 예: num_images가 36이면 sqrt(36)=6 -> 한 줄에 6개
    nrow = int(math.ceil(math.sqrt(args.num_images)))
    print(f"[Info] Calculating grid layout: {args.num_images} images -> {nrow}x{nrow} grid (nrow={nrow})")

    # 1. 공통 노이즈 생성 (Shared Noise)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    
    print(f"[Info] Generating shared noise for {args.num_images} images...")
    shared_noise = torch.randn(
        (args.num_images, 3, args.image_size, args.image_size),
        generator=generator,
        device=device
    )

    # 2. Teacher 모델 샘플링
    if args.teacher_path and os.path.exists(args.teacher_path):
        print(f"\n[Teacher] Loading from {args.teacher_path} ...")
        try:
            teacher_model = UNet2DModel.from_pretrained(args.teacher_path).to(device)
            try:
                scheduler = DDIMScheduler.from_pretrained(args.teacher_path)
            except:
                scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear", prediction_type="epsilon")
            
            generated_t = sample_ddim(
                teacher_model, scheduler, shared_noise, 
                steps=args.steps, eta=args.eta, device=device
            )
            
            save_path = out_dir / "teacher.png"
            # [변경] 계산된 nrow 사용
            grid_img = to_grid(generated_t, nrow=nrow)
            grid_img.save(save_path)
            print(f"[Done] Teacher image saved to: {save_path}")
            
        except Exception as e:
            print(f"[Error] Failed to process Teacher: {e}")
    else:
        print(f"[Skip] Teacher path not found or empty: {args.teacher_path}")

    # 3. Student 모델들 샘플링
    for idx, s_path in enumerate(args.student_paths):
        if not os.path.exists(s_path):
            print(f"[Skip] Student path not found: {s_path}")
            continue

        print(f"\n[Student #{idx+1}] Loading from {s_path} ...")
        try:
            student_model = UNet2DModel.from_pretrained(s_path).to(device)
            try:
                s_scheduler = DDIMScheduler.from_pretrained(s_path)
            except:
                s_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="linear", prediction_type="epsilon")
            
            generated_s = sample_ddim(
                student_model, s_scheduler, shared_noise, 
                steps=args.steps, eta=args.eta, device=device
            )
            
            # 파일명 정리
            path_obj = Path(s_path)
            model_name = path_obj.name
            if model_name in ["last", "checkpoint", "ckpt"]:
                # 상위 폴더명을 모델명으로 사용
                model_name = path_obj.parent.name
            
            save_name = f"student_{idx+1}_{model_name}.png"
            save_path = out_dir / save_name
            
            # [변경] 계산된 nrow 사용
            grid_img = to_grid(generated_s, nrow=nrow)
            grid_img.save(save_path)
            print(f"[Done] Student image saved to: {save_path}")

        except Exception as e:
            print(f"[Error] Failed to process Student {s_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Teacher and Student models")
    
    # --- Default 값들이 적용된 인자들 ---
    parser.add_argument("--teacher_path", type=str, default=DEFAULT_TEACHER, 
                        help="Path to teacher model")
    
    parser.add_argument("--student_paths", type=str, nargs='+', default=DEFAULT_STUDENT, 
                        help="List of paths to student models")
    
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, 
                        help="Device (e.g., cuda:0, cuda:1)")

    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, 
                        help="Directory to save images")
    
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--num_images", type=int, default=DEFAULT_NUM_IMAGES, help="Total images")
    
    # [제거] nrow 인자 제거 (항상 정사각형 자동 계산)
    
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="DDIM sampling steps")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    args = parser.parse_args()
    main(args)