import os
import shutil
import random
from pathlib import Path

# ================= 설정 (이 부분을 수정하세요) =================

# 1. 원본 데이터가 있는 최상위 경로
SOURCE_ROOT = 'cifar10_png_linear_only'

# 2. 클래스 당 추출할 이미지 개수 (N개)
SAMPLES_PER_CLASS = 100

# 3. 새로운 데이터가 저장될 경로
OUTPUT_ROOT = f'cifar10_student_data_n{SAMPLES_PER_CLASS}'

# 4. 추출하고 싶은 클래스 리스트
TARGET_CLASSES = ['airplane', 'automobile', 'bird', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'cat']

# 5. 대상 데이터 타입
DATA_TYPES = ['rgb', 'gray3']

# ============================================================

def create_dataset_subset():
    # 1. 클래스 목록 확정
    # TARGET_CLASSES가 비어있다면, 소스 디렉토리(rgb 기준)에서 자동으로 클래스 목록을 가져옴
    if not TARGET_CLASSES:
        ref_path = os.path.join(SOURCE_ROOT, 'rgb', 'train')
        if os.path.exists(ref_path):
            classes = sorted([d for d in os.listdir(ref_path) if os.path.isdir(os.path.join(ref_path, d))])
        else:
            print(f"Error: 기준 경로를 찾을 수 없습니다: {ref_path}")
            return
    else:
        classes = TARGET_CLASSES

    print(f"--- 작업 시작 ---")
    print(f"대상 클래스: {classes}")
    print(f"클래스 당 복사할 개수: {SAMPLES_PER_CLASS}")
    print(f"동일한 파일명으로 {DATA_TYPES} 모두 복사합니다.")

    # 2. 클래스별로 루프를 먼저 돕니다.
    for cls_name in classes:
        print(f"\n[Processing Class: {cls_name}]")
        
        # (A) 기준이 되는 폴더(rgb)에서 파일 리스트를 가져옵니다.
        # 여기서는 rgb에 있는 파일명이 gray3에도 똑같이 있다고 가정합니다.
        ref_src_class_dir = os.path.join(SOURCE_ROOT, 'rgb', 'train', cls_name)
        
        if not os.path.exists(ref_src_class_dir):
            print(f"[Warning] 원본 클래스 폴더 없음 (Skip): {ref_src_class_dir}")
            continue

        files = [f for f in os.listdir(ref_src_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_files = len(files)

        # (B) 파일 샘플링 (여기서 한 번만 수행하여 리스트를 고정합니다)
        if total_files <= SAMPLES_PER_CLASS:
            selected_files = files
            print(f" -> 파일이 부족하여 전체 선택 ({total_files}개)")
        else:
            selected_files = random.sample(files, SAMPLES_PER_CLASS)
            print(f" -> 전체 {total_files}개 중 {SAMPLES_PER_CLASS}개 랜덤 선택 완료")

        # (C) 고정된 파일 리스트(selected_files)를 이용해 rgb와 gray3 각각 복사
        for dtype in DATA_TYPES:
            src_root_dir = os.path.join(SOURCE_ROOT, dtype, 'train', cls_name)
            dst_root_dir = os.path.join(OUTPUT_ROOT, dtype, 'train', cls_name)

            # 목적지 폴더 생성
            os.makedirs(dst_root_dir, exist_ok=True)

            # 실제 복사 수행
            copy_count = 0
            for file_name in selected_files:
                src_file = os.path.join(src_root_dir, file_name)
                dst_file = os.path.join(dst_root_dir, file_name)
                
                # 파일이 실제로 존재하는지 체크 (gray3에 파일이 없을 수도 있으므로)
                if os.path.exists(src_file):
                    shutil.copy2(src_file, dst_file)
                    copy_count += 1
                else:
                    print(f"[Missing] {dtype}에 파일이 없습니다: {file_name}")
            
            print(f"    -> [{dtype}] 복사 완료: {copy_count}개")

    print("\n--- 모든 작업이 완료되었습니다. ---")
    print(f"생성된 루트 경로: {OUTPUT_ROOT}")
    
    # 트리 구조 확인용 (리눅스/맥 환경인 경우)
    if shutil.which('tree'):
        print("\n[생성된 폴더 구조 (level 3)]")
        os.system(f"tree -L 3 {OUTPUT_ROOT}")

if __name__ == "__main__":
    create_dataset_subset()