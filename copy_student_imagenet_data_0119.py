import os
import shutil
import random
from tqdm import tqdm  # 진행률 표시를 위한 라이브러리

def create_imagenet_subset(
    source_root,
    target_root,
    n_per_class,
    class_percent=100.0,
    seed=0,
):
    """
    ImageNet 데이터셋에서 '랜덤으로 선택된 일부 클래스'에 대해,
    각 클래스별로 n개의 이미지를 샘플링하여 새로운 데이터셋을 만듭니다.

    Args:
        source_root (str): 원본 train 데이터 경로 (클래스 폴더들이 있는 곳)
        target_root (str): 데이터가 저장될 새로운 경로
        n_per_class (int): 각 클래스당 복사할 이미지 개수
        class_percent (float): 사용할 클래스 비율 (0~100)
        seed (int): 랜덤 시드 (재현성)
    """

    # 1. 소스 경로 확인
    if not os.path.exists(source_root):
        print(f"Error: 원본 경로를 찾을 수 없습니다: {source_root}")
        return

    if not (0.0 < class_percent <= 100.0):
        raise ValueError("class_percent는 (0, 100] 범위여야 합니다.")

    rng = random.Random(seed)

    # 2. 클래스 폴더 목록 가져오기
    classes = [
        d for d in os.listdir(source_root)
        if os.path.isdir(os.path.join(source_root, d)) and not d.startswith('.')
    ]
    classes.sort()

    # 2-1. 클래스 중 일부만 랜덤 선택
    k = max(1, int(round(len(classes) * (class_percent / 100.0))))
    selected_classes = rng.sample(classes, k)
    selected_classes.sort()

    print(f"--- 작업 시작 ---")
    print(f"원본 경로: {source_root}")
    print(f"대상 경로: {target_root}")
    print(f"총 클래스 수: {len(classes)}개")
    print(f"선택된 클래스 비율: {class_percent}% -> {len(selected_classes)}개")
    print(f"클래스 당 복사할 이미지 수: {n_per_class}개")
    print(f"Seed: {seed}")
    print("-" * 30)

    # (옵션) 어떤 클래스가 선택됐는지 저장/출력
    os.makedirs(target_root, exist_ok=True)
    list_path = os.path.join(target_root, f"selected_classes_{class_percent}pct_seed{seed}.txt")
    with open(list_path, "w") as f:
        f.write("\n".join(selected_classes))
    print(f"선택된 클래스 목록 저장: {list_path}")

    # 3. 선택된 클래스만 순회하며 복사
    for class_name in tqdm(selected_classes, desc="Processing Selected Classes"):
        source_class_dir = os.path.join(source_root, class_name)
        target_class_dir = os.path.join(target_root, class_name)

        os.makedirs(target_class_dir, exist_ok=True)

        images = [
            f for f in os.listdir(source_class_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]

        # 4. 이미지 샘플링
        if len(images) <= n_per_class:
            selected_images = images
        else:
            selected_images = rng.sample(images, n_per_class)

        # 5. 파일 복사
        for img_name in selected_images:
            src_file = os.path.join(source_class_dir, img_name)
            dst_file = os.path.join(target_class_dir, img_name)
            shutil.copy2(src_file, dst_file)

    print("\n--- 작업 완료 ---")
    print(f"새로운 데이터셋이 '{target_root}'에 생성되었습니다.")

# ==========================================
# 설정 변수 (이곳을 수정하세요)
# ==========================================

SOURCE_DIR = "/workspace/rkd_cifar10_1111/imagenet1k_export/gray3/train"

N_IMAGES = 1
CLASS_PERCENT = 10.0   # 예: 전체 클래스 중 10%만 사용
SEED = 0

TARGET_DIR = (
    f"/workspace/rkd_cifar10_1111/imagenet1k_export/gray3_subset_"
    f"class{int(CLASS_PERCENT)}pct_per{N_IMAGES}_seed{SEED}/train"
)

if __name__ == "__main__":
    create_imagenet_subset(
        SOURCE_DIR,
        TARGET_DIR,
        N_IMAGES,
        class_percent=CLASS_PERCENT,
        seed=SEED,
    )
