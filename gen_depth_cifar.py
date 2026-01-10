import os
import glob
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# -----------------------------------------------------------------------------
# 1. Dataset 정의
# -----------------------------------------------------------------------------
class CifarImageDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = sorted(
            glob.glob(os.path.join(root_dir, '**', '*.*'), recursive=True)
        )
        self.image_paths = [
            p for p in self.image_paths 
            if p.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.root_dir = root_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        rel_path = os.path.relpath(path, self.root_dir)
        return image, rel_path, image.size

# -----------------------------------------------------------------------------
# 2. 메인 변환 로직
# -----------------------------------------------------------------------------
def main(args):
    # CUDA 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)} (ID: {args.gpu_id})")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    # 모델 로드 (Depth Anything V2 Small)
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    
    print(f"Loading model: {model_id}...")
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()

    # 데이터셋 & 데이터로더
    dataset = CifarImageDataset(args.input_root)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=lambda x: list(zip(*x))
    )

    print(f"Total images to process: {len(dataset)}")

    for batch in tqdm(dataloader):
        images_pil = batch[0]
        rel_paths = batch[1]
        orig_sizes = batch[2]

        inputs = image_processor(images=images_pil, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        for i, (depth, rel_path, (w, h)) in enumerate(zip(predicted_depth, rel_paths, orig_sizes)):
            # 1. 원본 크기(32x32)로 복원
            depth_unsqueezed = depth.unsqueeze(0).unsqueeze(0)
            depth_resized = torch.nn.functional.interpolate(
                depth_unsqueezed,
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            # 2. 0-255 정규화
            depth_min = depth_resized.min()
            depth_max = depth_resized.max()
            
            if depth_max - depth_min > 0:
                depth_norm = (depth_resized - depth_min) / (depth_max - depth_min)
            else:
                depth_norm = torch.zeros_like(depth_resized)
            
            depth_uint8 = (depth_norm * 255.0).cpu().numpy().astype(np.uint8)

            # 3. 저장
            save_path = os.path.join(args.output_root, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            Image.fromarray(depth_uint8).save(save_path)

    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 경로 설정
    parser.add_argument("--input_root", type=str, 
                        default="/workspace/rkd_cifar10_1111/cifar10_png_linear_only/rgb",
                        help="Root directory of RGB images")
    parser.add_argument("--output_root", type=str, 
                        default="/workspace/rkd_cifar10_1111/cifar10_png_linear_only/depth",
                        help="Root directory to save Depth images")
    
    # 하이퍼파라미터 및 GPU 설정
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--gpu_id", type=int, default=0, help="CUDA GPU ID to use (e.g., 0, 1, 2...)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_root, exist_ok=True)
    main(args)