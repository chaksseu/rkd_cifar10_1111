import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder # 또는 기존 Dataset 클래스 사용

# 설정
# VAE_PATH = "vae_out_dir/1227_b64_lr0.0001_MSE_klW_1e-08_block_64_128/checkpoint-1000000" # 수정 필요
VAE_PATH = "1227_b64_lr0.0001_MSE_klW_1e-08_block_64_128-checkpoint-1000000" # 수정 필요

DATA_DIR = "cifar10_png_linear_only/rgb/train" # 수정 필요
DEVICE = "cuda"

def check_std():
    # 1. Load Model
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(DEVICE)
    vae.eval()

    # 2. Load Data (Small subset)
    tfm = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize([0.5], [0.5])])
    # ImageFolderDataset 대신 간단히 테스트하려면 torchvision ImageFolder 써도 됨
    # 여기선 복잡하니 그냥 임의의 데이터로 가정하거나 사용자 Dataset 클래스 사용
    # 간단히:
    dummy_batch = torch.randn(32, 3, 32, 32).to(DEVICE) # 정규분포 데이터라 가정

    print("Calculating z_std from dummy noise (approx) or real data...")
    
    # 실제 데이터가 있다면 더 정확함
    with torch.no_grad():
        dist = vae.encode(dummy_batch).latent_dist
        z = dist.sample()
        print(f"Measured z_mean: {z.mean():.4f}")
        print(f"Measured z_std:  {z.std():.4f}")
        print(f"Recommended LATENT_SCALE = 1 / {z.std():.4f} = {1/z.std():.4f}")

if __name__ == "__main__":
    check_std()