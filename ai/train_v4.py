"""
Baby Safety Monitor — v4 학습 스크립트
SabiCare + VTAR + 오버헤드 3종 데이터셋 기반
클래스: 정면(0) / 측면(1) / 후면(2)

실행:
  python3 ai/train_v4.py           # last.pt 있으면 이어서, 없으면 처음부터
  python3 ai/train_v4.py --fresh   # last.pt 삭제 후 yolov8n.pt 부터 완전 재학습
"""

import argparse, shutil
from ultralytics import YOLO
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "ai" / "dataset_v4" / "data.yaml"
PROJECT = ROOT / "ai" / "runs"

CONFIG = {
    "model"        : "yolov8n.pt",
    "data"         : str(DATA),
    "epochs"       : 200,
    "imgsz"        : 640,
    "batch"        : 4,
    "cache"        : False,
    "patience"     : 20,
    "project"      : str(PROJECT),
    "name"         : "baby_monitor_v4",
    "exist_ok"     : True,
    "device"       : "mps",
    "workers"      : 0,
    "amp"          : False,
    "cos_lr"       : True,
    "close_mosaic" : 10,

    # 증강 — 오버헤드 포함 다양한 각도·조명 대응
    "hsv_h"       : 0.02,
    "hsv_s"       : 0.5,
    "hsv_v"       : 0.5,
    "flipud"      : 0.5,        # 위아래 반전 (오버헤드 카메라 각도 다양성)
    "fliplr"      : 0.5,
    "mosaic"      : 1.0,
    "degrees"     : 45.0,       # 대각선 누운 자세(정면 오탐) 최대 대응
    "translate"   : 0.1,
    "scale"       : 0.5,
    "perspective" : 0.0005,     # 오버헤드 원근 왜곡 대응

    # 후면 클래스에 더 높은 가중치 (안전 우선)
    "cls"  : 1.5,
    "box"  : 7.5,
    "dfl"  : 1.5,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true",
                        help="last.pt 삭제 후 yolov8n.pt 부터 완전 재학습")
    args = parser.parse_args()

    if not DATA.exists():
        print(f"[Error] 데이터셋 없음: {DATA}")
        print("먼저 실행하세요:")
        print("  python3 ai/download_dataset.py --api-key <YOUR_KEY>")
        return

    last_ckpt = PROJECT / "baby_monitor_v4" / "weights" / "last.pt"

    if args.fresh and last_ckpt.exists():
        last_ckpt.unlink()
        print(f"[Fresh] last.pt 삭제 완료 — yolov8n.pt 부터 재학습")

    for split in ("train", "valid", "test"):
        cache = DATA.parent / split / "labels.cache"
        if cache.exists():
            cache.unlink()

    train_imgs = list((DATA.parent / "train" / "images").glob("*.*"))
    val_imgs   = list((DATA.parent / "valid" / "images").glob("*.*"))

    print("=" * 50)
    print(" Baby Safety Monitor v4 학습 (프로덕션)")
    print("=" * 50)
    print(f" 클래스  : 정면 / 측면 / 후면")
    print(f" train   : {len(train_imgs)}장")
    print(f" valid   : {len(val_imgs)}장")
    print(f" 에폭    : {CONFIG['epochs']} (patience {CONFIG['patience']})")
    print(f" imgsz   : {CONFIG['imgsz']}")
    print(f" 디바이스: {CONFIG['device']}")
    print("=" * 50)

    if last_ckpt.exists():
        # last.pt를 초기 가중치로만 사용, 새 CONFIG 완전 적용 (resume=True 사용 안 함)
        print(f"\n[Finetune] last.pt 기반 재학습 (현재 CONFIG 적용): {last_ckpt}")
        start_model = str(last_ckpt)
    else:
        print("\n[Train] yolov8n.pt 부터 처음 학습")
        start_model = CONFIG["model"]

    train_cfg = {k: v for k, v in CONFIG.items() if k != "model"}
    model = YOLO(start_model)
    model.train(**train_cfg)

    best = PROJECT / "baby_monitor_v4" / "weights" / "best.pt"
    if best.exists():
        size_mb = best.stat().st_size / 1024 / 1024
        print(f"\n학습 완료! 모델: {best} ({size_mb:.1f}MB)")
        print("다음 실행:")
        print("  python3 vision/baby_monitor_v4.py --source 0")


if __name__ == "__main__":
    main()
