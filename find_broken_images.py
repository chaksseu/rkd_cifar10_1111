import os
import argparse
from pathlib import Path
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def validate_image(file_path):
    """
    개별 이미지를 검사하는 워커 함수입니다.
    손상된 경우 파일 경로를 반환하고, 정상이면 None을 반환합니다.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()  # 이미지를 decode하지 않고 헤더와 구조만 빠르게 확인
        return None
    except (IOError, SyntaxError, OSError):
        return str(file_path)
    except Exception:
        # 그 외 알 수 없는 에러도 일단 손상으로 간주
        return str(file_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./imagenet1k_export/gray3/val", help="데이터셋 경로")
    parser.add_argument("--workers", type=int, default=64, help="사용할 CPU 코어 수")
    parser.add_argument("--delete", action="store_true", help="발견 즉시 삭제하려면 추가")
    args = parser.parse_args()

    root_dir = Path(args.data_dir)
    
    print(f"[{args.data_dir}] 파일 리스트 스캔 중... (잠시만 기다려주세요)")
    # 확장자는 필요에 따라 추가하세요
    extensions = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff'}
    all_files = []
    
    # rglob이 느릴 경우를 대비해 scandir을 쓸 수도 있지만, 일반적으로 이 정도면 충분합니다.
    for ext in extensions:
        all_files.extend(list(root_dir.rglob(ext)))

    print(f"총 {len(all_files)}개의 이미지 파일을 검사합니다. (Workers: {args.workers})")

    bad_files = []
    
    # 멀티프로세싱으로 병렬 처리
    with Pool(args.workers) as pool:
        # imap_unordered가 순서 상관없이 처리되는 대로 반환하여 더 효율적입니다.
        for result in tqdm(pool.imap_unordered(validate_image, all_files), total=len(all_files)):
            if result:
                bad_files.append(result)

    print("\n" + "="*30)
    print(f"검사 완료. 발견된 손상 파일: {len(bad_files)}개")
    print("="*30)

    if bad_files:
        # 로그 파일로 저장
        with open("broken_images.log", "w") as f:
            for path in bad_files:
                f.write(f"{path}\n")
        print("손상된 파일 목록이 'broken_images.log'에 저장되었습니다.")

        # 삭제 옵션이 켜져있으면 삭제 수행
        if args.delete:
            print("삭제 옵션이 활성화되어 파일을 삭제합니다...")
            for path in bad_files:
                try:
                    os.remove(path)
                    print(f"Deleted: {path}")
                except OSError as e:
                    print(f"Error deleting {path}: {e}")
        else:
            print("\n[Tip] 삭제하려면 실행 시 --delete 옵션을 붙이거나, 다음 명령어를 사용하세요:")
            print("xargs rm < broken_images.log")

if __name__ == "__main__":
    main()