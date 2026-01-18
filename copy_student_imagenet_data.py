import os
import shutil
import random
from tqdm import tqdm  # 진행률 표시를 위한 라이브러리

def create_imagenet_subset(source_root, target_root, n_per_class):
    """
    ImageNet 데이터셋에서 각 클래스별로 n개의 이미지를 샘플링하여 새로운 데이터셋을 만듭니다.
    
    Args:
        source_root (str): 원본 train 데이터 경로 (클래스 폴더들이 있는 곳)
        target_root (str): 데이터가 저장될 새로운 경로
        n_per_class (int): 각 클래스당 복사할 이미지 개수
    """
    
    # 1. 소스 경로 확인
    if not os.path.exists(source_root):
        print(f"Error: 원본 경로를 찾을 수 없습니다: {source_root}")
        return

    # 2. 클래스 폴더 목록 가져오기
    # .으로 시작하는 숨김 파일/폴더는 제외
    classes = [d for d in os.listdir(source_root) 
               if os.path.isdir(os.path.join(source_root, d)) and not d.startswith('.')]
    
    classes.sort() # 순서 정렬
    
    print(f"--- 작업 시작 ---")
    print(f"원본 경로: {source_root}")
    print(f"대상 경로: {target_root}")
    print(f"총 클래스 수: {len(classes)}개")
    print(f"클래스 당 복사할 이미지 수: {n_per_class}개")
    print("-" * 30)

    # 3. 각 클래스별로 순회하며 복사 (tqdm으로 진행바 표시)
    for class_name in tqdm(classes, desc="Processing Classes"):
        source_class_dir = os.path.join(source_root, class_name)
        target_class_dir = os.path.join(target_root, class_name)
        
        # 대상 클래스 폴더 생성
        os.makedirs(target_class_dir, exist_ok=True)
        
        # 이미지 파일 목록 가져오기
        images = [f for f in os.listdir(source_class_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        # 4. 이미지 샘플링 (이미지 수가 n보다 적으면 전체 복사)
        if len(images) <= n_per_class:
            selected_images = images
        else:
            # 랜덤하게 n개 선택 (데이터 분포를 위해 랜덤 추천)
            selected_images = random.sample(images, n_per_class)
            
        # 5. 파일 복사 실행
        for img_name in selected_images:
            src_file = os.path.join(source_class_dir, img_name)
            dst_file = os.path.join(target_class_dir, img_name)
            shutil.copy2(src_file, dst_file) # copy2는 메타데이터까지 보존

    print("\n--- 작업 완료 ---")
    print(f"새로운 데이터셋이 '{target_root}'에 생성되었습니다.")

# ==========================================
# 설정 변수 (이곳을 수정하세요)
# ==========================================

# 원본 데이터 경로 (제공해주신 경로 기준)
SOURCE_DIR = "/workspace/rkd_cifar10_1111/imagenet1k_export/gray3/train"

# 새로 만들 데이터 경로 (원하는 이름으로 변경 가능)
N_IMAGES = 5
TARGET_DIR = f"/workspace/rkd_cifar10_1111/imagenet1k_export/gray3_subset_per{N_IMAGES}/train"

# 각 클래스당 추출할 이미지 개수


if __name__ == "__main__":
    create_imagenet_subset(SOURCE_DIR, TARGET_DIR, N_IMAGES)