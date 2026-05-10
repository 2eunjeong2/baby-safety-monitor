"""
Baby Safety Monitor v4
YOLOv8n 기반 아기 자세 감지 — 정면 / 측면 / 후면

실행: python3 vision/baby_monitor_v4.py [--source 0|파일경로]
종료: q 또는 ESC
"""

import argparse, time, sys
from collections import deque
import cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

ROOT       = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "ai" / "runs" / "baby_monitor_v4" / "weights" / "best.pt"
FONT_PATH  = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

CLASSES     = {0: "정면", 1: "측면", 2: "후면"}
CONF_THRESH  = 0.40   # 최소 신뢰도
CONF_SIDE    = 0.60   # 측면은 더 높은 신뢰도 요구 (정면 오탐 방지)
BACK_ALERT   = 5.0    # 후면 N초 지속 → 위험
FRAME_SKIP   = 2      # N프레임마다 1회 추론
SMOOTH_WIN   = 20     # 최근 N 추론 다수결 (≈1초 @ 30fps)

COLOR_GREEN  = (50,  200,  50)
COLOR_YELLOW = (0,   200, 210)
COLOR_RED    = (50,   50, 220)
COLOR_WHITE  = (255, 255, 255)
COLOR_GRAY   = (160, 160, 160)
CLASS_COLOR  = {0: COLOR_GREEN, 1: COLOR_YELLOW, 2: COLOR_RED}


def load_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return None


def put_text(frame, text, pos, color, font):
    if font is None:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img).text(pos, text, font=font,
                              fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


ASPECT_FLAT    = 1.7   # W/H 임계값: 누운 자세(정면/후면) 특징
BACK_MIN_COUNT = 3     # 후면 25% 규칙 최소 감지 횟수 (초기 과민 방지)

def best_detection(boxes):
    """신뢰도 가장 높은 감지 1개만 반환 (아기는 한 명)."""
    if len(boxes) == 0:
        return None
    best_idx = boxes.conf.argmax().item()
    cls_id = int(boxes.cls[best_idx].item())
    conf   = float(boxes.conf[best_idx].item())
    # 측면은 더 높은 신뢰도 필요 — 정면 대각선 자세 오탐 방지
    if cls_id == 1 and conf < CONF_SIDE:
        return None
    box = boxes.xyxy[best_idx].cpu().numpy()
    x1, y1, x2, y2 = box
    w, h = (x2 - x1), (y2 - y1)
    # 오버헤드 기하학 보정: bbox가 옆으로 넓으면 누운 자세 → 측면 오탐 교정
    if cls_id == 1 and h > 0 and (w / h) > ASPECT_FLAT:
        cls_id = 0   # 측면→정면 교정
    return cls_id, conf, box


def smooth_class(history: deque):
    """최근 N 프레임 다수결. 후면은 25% 이상이고 최소 BACK_MIN_COUNT회면 우선."""
    counts = {}
    for c in history:
        counts[c] = counts.get(c, 0) + 1

    detected = {k: v for k, v in counts.items() if k is not None}
    if not detected:
        return None
    back_count = counts.get(2, 0)
    # 후면 안전 우선: 25% 이상 + 최소 횟수 동시 충족 시 확정
    # (초기 1~2프레임에서 후면 1회만으로 오경보하는 케이스 방지)
    if back_count >= BACK_MIN_COUNT and back_count >= len(history) * 0.25:
        return 2
    return max(detected, key=detected.get)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print("[오류] 모델 없음. 먼저 실행하세요:")
        print("  python3 ai/train_v4.py")
        sys.exit(1)

    source = int(args.source) if args.source.isdigit() else args.source

    print("[로드] 모델 초기화...")
    model = YOLO(str(MODEL_PATH))
    font    = load_font(28)
    font_sm = load_font(18)
    print("[로드] 완료\n")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[오류] 소스 열기 실패: {source}")
        sys.exit(1)

    back_start    = None   # 후면 감지 전용 타이머 (미감지와 분리)
    no_det_start  = None   # 미감지 전용 타이머
    frame_n       = 0
    prev_tick     = cv2.getTickCount()
    cached_det    = None   # (cls_id, conf, box) or None
    cached_status = (COLOR_GRAY, "초기화 중...")
    cls_history   = deque(maxlen=SMOOTH_WIN)  # 시간적 평활화용

    def reset_state():
        nonlocal back_start, no_det_start, cached_det, cached_status
        back_start   = None
        no_det_start = None
        cached_det   = None
        cached_status = (COLOR_GRAY, "초기화 중...")
        cls_history.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                reset_state()   # 루프 재시작 시 타이머·히스토리 초기화
                continue
            break

        curr_tick = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (curr_tick - prev_tick)
        prev_tick = curr_tick
        frame_n += 1

        # ── 추론 ──────────────────────────────────────
        if frame_n % FRAME_SKIP == 0:
            results = model(frame, conf=CONF_THRESH, verbose=False)
            det = best_detection(results[0].boxes)
            cached_det = det

            cls_history.append(det[0] if det is not None else None)
            smooth = smooth_class(cls_history)

            now = time.time()
            if smooth is None:
                # 미감지 타이머: 후면 타이머(back_start)와 완전히 분리
                if no_det_start is None:
                    no_det_start = now
                elapsed = now - no_det_start
                back_start = None   # 후면 타이머는 감지가 없으면 리셋
                if elapsed >= BACK_ALERT:
                    cached_status = (COLOR_RED, f"미감지 {elapsed:.1f}초 — 확인 필요!")
                else:
                    cached_status = (COLOR_GRAY, f"확인 중... ({elapsed:.1f}s)")
            else:
                no_det_start = None   # 감지되면 미감지 타이머 리셋
                if smooth == 2:   # 후면
                    if back_start is None:
                        back_start = now
                    elapsed = now - back_start
                    if elapsed >= BACK_ALERT:
                        cached_status = (COLOR_RED, f"후면 {elapsed:.1f}초 — 위험!")
                    else:
                        cached_status = (COLOR_RED, f"후면 감지 ({elapsed:.1f}s / {BACK_ALERT}s)")
                else:
                    back_start = None
                    if smooth == 1:
                        cached_status = (COLOR_YELLOW, "측면 — 주의")
                    else:
                        cached_status = (COLOR_GREEN, "정면 — 안전")

        sc, msg = cached_status

        # ── 바운딩박스 ────────────────────────────────
        if cached_det is not None:
            cls_id, conf, box = cached_det
            x1, y1, x2, y2 = map(int, box)
            c = CLASS_COLOR.get(cls_id, COLOR_GRAY)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            label = f"{CLASSES[cls_id]} {conf:.0%}"
            frame = put_text(frame, label, (x1, max(0, y1 - 32)), c, font_sm)

        # ── 상단 오버레이 ──────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 62), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        frame = put_text(frame, msg, (12, 8), sc, font)
        cv2.putText(frame, f"FPS:{fps:.1f}", (12, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)

        # ── 우하단 LED 원 ──────────────────────────────
        fh, fw = frame.shape[:2]
        cv2.circle(frame, (fw - 24, fh - 24), 14, sc, -1)
        cv2.circle(frame, (fw - 24, fh - 24), 14, COLOR_WHITE, 1)

        cv2.imshow("Baby Safety Monitor v4", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[종료]")


if __name__ == "__main__":
    main()
