"""
================================================
 Baby Safety Monitor — YOLOv8 학습 스크립트
 파일명 : train.py
 위치   : ai/train.py
------------------------------------------------
 Roboflow 데이터셋(baby 클래스)으로
 YOLOv8n 모델을 파인튜닝한다.

 실행 방법
   source venv/bin/activate
   python3 ai/train.py

 결과물
   ai/runs/detect/baby_monitor/weights/best.pt
================================================
"""

from ultralytics import YOLO
import os

# ── 경로 설정 ─────────────────────────────────
# 프로젝트 루트 기준으로 경로 고정
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA     = os.path.join(ROOT, "ai", "dataset", "data.yaml")
PROJECT  = os.path.join(ROOT, "ai", "runs")

# ── 학습 설정 ─────────────────────────────────
CONFIG = {
    "model"      : "yolov8n.pt",   # nano 모델 (M1 맥북 최적 크기)
    "data"       : DATA,
    "epochs"     : 30,             # 데이터셋 규모 고려 (부족하면 50으로 올려도 됨)
    "imgsz"      : 640,            # 입력 이미지 크기
    "batch"      : 8,              # M1 메모리 고려 (부족하면 4로 낮추기)
    "patience"   : 10,             # Early stopping (10 epoch 개선 없으면 종료)
    "project"    : PROJECT,
    "name"       : "baby_monitor",
    "exist_ok"   : True,           # 같은 이름 폴더 덮어쓰기 허용
    "device"     : "mps",          # M1 GPU 사용 (안 되면 "cpu"로 변경)
    "workers"    : 4,
    "verbose"    : True,
}

# ════════════════════════════════════════════════
def main():
    print("=" * 50)
    print(" Baby Safety Monitor — YOLOv8 학습 시작")
    print("=" * 50)
    print(f" 데이터셋 : {DATA}")
    print(f" 저장 경로: {PROJECT}/baby_monitor/weights/best.pt")
    print(f" 디바이스 : {CONFIG['device']}")
    print("=" * 50)

    # ── 사전학습 모델 로드 ─────────────────────
    model = YOLO(CONFIG["model"])
    print(f"\n[Model] {CONFIG['model']} 로드 완료")

    # ── 학습 실행 ──────────────────────────────
    results = model.train(
        data     = CONFIG["data"],
        epochs   = CONFIG["epochs"],
        imgsz    = CONFIG["imgsz"],
        batch    = CONFIG["batch"],
        patience = CONFIG["patience"],
        project  = CONFIG["project"],
        name     = CONFIG["name"],
        exist_ok = CONFIG["exist_ok"],
        device   = CONFIG["device"],
        workers  = CONFIG["workers"],
        verbose  = CONFIG["verbose"],
    )

    # ── 학습 결과 출력 ─────────────────────────
    print("\n" + "=" * 50)
    print(" 학습 완료!")
    print("=" * 50)

    best_model = os.path.join(PROJECT, "baby_monitor", "weights", "best.pt")

    if os.path.exists(best_model):
        print(f"\n 최적 모델 저장 위치:\n  {best_model}")
    else:
        print("\n [Warning] best.pt 파일을 찾을 수 없어요. 경로를 확인해주세요.")

    # ── 정확도 요약 출력 ───────────────────────
    print("\n 학습 결과 요약")
    print(f"  mAP50    : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"  mAP50-95 : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"  Recall   : {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")
    print("\n 결과 그래프: ai/runs/baby_monitor/results.png")
    print("=" * 50)


# ════════════════════════════════════════════════
if __name__ == "__main__":
    main()