"""
Baby Safety Monitor v4
YOLOv8n 기반 아기 자세 감지 — 정면 / 측면 / 후면

실행: python3 vision/baby_monitor_v4.py [--source 0|파일경로]
종료: q 또는 ESC
"""

import argparse, time, sys, threading, signal
from collections import deque
from datetime import datetime
import cv2, numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

ROOT       = Path(__file__).resolve().parent.parent
# Pi(ARM)에서는 ONNX+onnxruntime, Mac/PC에서는 .pt 사용
_onnx = ROOT / "model" / "best.onnx"
_pt   = ROOT / "model" / "best.pt"
MODEL_PATH = _onnx if _onnx.exists() else _pt
FONT_PATH  = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

CLASSES     = {0: "정면", 1: "측면", 2: "후면"}
CONF_THRESH  = 0.40   # 최소 신뢰도
CONF_SIDE    = 0.60   # 측면은 더 높은 신뢰도 요구 (정면 오탐 방지)
BACK_ALERT    = 5.0    # 후면 N초 지속 → 위험 캡처
CAUTION_ALERT = 5.0    # 측면 N초 지속 → 경고 캡처
CAPTURE_DIR   = ROOT / "captures"
FRAME_SKIP    = 4      # N프레임마다 1회 추론 (Pi 성능 고려)
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


def save_capture(frame, label: str) -> Path:
    """경고 발생 시 프레임을 captures/ 폴더에 저장."""
    CAPTURE_DIR.mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = CAPTURE_DIR / f"{label}_{ts}.jpg"
    cv2.imwrite(str(path), frame)
    print(f"[캡처] {path.name}")
    return path


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
    parser.add_argument("--no-stream", action="store_true",
                        help="Flask 스트리밍 서버 비활성화")
    parser.add_argument("--headless", action="store_true",
                        help="디스플레이 없이 실행 (SSH 환경)")
    args = parser.parse_args()

    headless = args.headless
    if headless:
        print("[헤드리스] 디스플레이 없이 실행 — 스트리밍으로 확인하세요")

    if not MODEL_PATH.exists():
        print("[오류] 모델 없음. 먼저 실행하세요:")
        print("  python3 ai/train_v4.py")
        sys.exit(1)

    source = int(args.source) if args.source.isdigit() else args.source

    # ── Flask 스트리밍 서버 (백그라운드 스레드) ───────────────
    _set_frame = None
    if not args.no_stream:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from alert_server import app as _flask_app, set_frame as _set_frame
            t = threading.Thread(
                target=lambda: _flask_app.run(
                    host="0.0.0.0", port=8080, debug=False, use_reloader=False),
                daemon=True)
            t.start()
            print("[스트림] http://0.0.0.0:8080/live")
        except Exception as e:
            print(f"[경고] 스트리밍 서버 시작 실패: {e}")
            _set_frame = None

    print("[로드] 모델 초기화...")
    model = YOLO(str(MODEL_PATH))
    font    = load_font(42)
    font_sm = load_font(27)
    print("[로드] 완료\n")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[오류] 소스 열기 실패: {source}")
        sys.exit(1)

    # 헤드리스 종료 플래그 (Ctrl+C)
    _running = True
    def _stop(*_):
        nonlocal _running
        _running = False
    signal.signal(signal.SIGINT, _stop)

    back_start       = None   # 후면 감지 전용 타이머 (미감지와 분리)
    no_det_start     = None   # 미감지 전용 타이머
    was_back         = False  # 마지막 감지가 후면이었는지 (경고 알림용)
    state_start      = None   # 현재 상태 누적 시간 타이머
    prev_smooth      = -1     # 상태 변화 감지용
    caution_captured = False  # CAUTION 5초 캡처 여부
    back_captured    = False  # DANGER(후면) 5초 캡처 여부
    nodet_captured   = False  # DANGER(미감지) 5초 캡처 여부
    pending_capture  = None   # 오버레이 후 저장할 라벨 (None이면 저장 없음)
    frame_n          = 0
    prev_tick     = cv2.getTickCount()
    cached_det    = None   # (cls_id, conf, box) or None
    cached_status = (COLOR_GRAY, "초기화 중...", "")
    cls_history   = deque(maxlen=SMOOTH_WIN)  # 시간적 평활화용

    def reset_state():
        nonlocal back_start, no_det_start, was_back, state_start, prev_smooth, \
                 caution_captured, back_captured, nodet_captured, cached_det, cached_status
        back_start       = None
        no_det_start     = None
        was_back         = False
        state_start      = None
        prev_smooth      = -1
        caution_captured = False
        back_captured    = False
        nodet_captured   = False
        cached_det       = None
        cached_status    = (COLOR_GRAY, "초기화 중...", "")
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
            if smooth != prev_smooth:
                state_start      = now
                prev_smooth      = smooth
                caution_captured = False
                back_captured    = False
                nodet_captured   = False
            state_dur = now - state_start

            if smooth is None:
                # 미감지 타이머: 후면 타이머(back_start)와 완전히 분리
                if no_det_start is None:
                    no_det_start = now
                elapsed = now - no_det_start
                back_start = None   # 후면 타이머는 감지가 없으면 리셋
                if elapsed >= BACK_ALERT:
                    cached_status = (COLOR_RED, "DANGER !", f"No Det  {elapsed:.1f}s")
                    if not nodet_captured:
                        pending_capture = "DANGER_NODET"
                        nodet_captured  = True
                else:
                    cached_status = (COLOR_GRAY, "감지 중...", f"{elapsed:.1f}s")
            else:
                no_det_start = None   # 감지되면 미감지 타이머 리셋
                if smooth == 2:   # 후면
                    was_back = True
                    if back_start is None:
                        back_start = now
                    elapsed = now - back_start
                    if elapsed >= BACK_ALERT:
                        cached_status = (COLOR_RED, "DANGER !", f"Back  {elapsed:.1f}s")
                        if not back_captured:
                            pending_capture = "DANGER_BACK"
                            back_captured   = True
                    else:
                        cached_status = (COLOR_RED, "DANGER", f"{elapsed:.1f}s / {BACK_ALERT:.0f}s")
                else:
                    was_back = False
                    back_start = None
                    if smooth == 1:
                        if state_dur >= CAUTION_ALERT and not caution_captured:
                            pending_capture  = "CAUTION"
                            caution_captured = True
                        cached_status = (COLOR_YELLOW, "CAUTION", f"{state_dur:.1f}s")
                    else:
                        cached_status = (COLOR_GREEN, "SAFE", f"{state_dur:.1f}s")

        sc, status_label, status_time = cached_status

        # ── 바운딩박스 ────────────────────────────────
        if cached_det is not None:
            cls_id, conf, box = cached_det
            x1, y1, x2, y2 = map(int, box)
            c = CLASS_COLOR.get(cls_id, COLOR_GRAY)
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            det_label = f"{CLASSES[cls_id]} {conf:.0%}"
            frame = put_text(frame, det_label, (x1, max(0, y1 - 36)), c, font_sm)

        # ── 상단 오버레이 ──────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 85), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        frame = put_text(frame, status_label, (12, 8), sc, font)
        if status_time:
            if font is not None:
                bb = font.getbbox(status_label)
                label_w = bb[2] - bb[0]
            else:
                label_w = len(status_label) * 24
            frame = put_text(frame, status_time, (12 + label_w + 20, 8), sc, font)
        fps_text = f"FPS:{fps:.1f}"
        (fps_tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv2.putText(frame, fps_text, (frame.shape[1] - fps_tw - 12, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_GRAY, 1)

        # ── 캡처 저장 (오버레이 포함된 프레임) ────────────
        if pending_capture:
            save_capture(frame, pending_capture)
            pending_capture = None

        # ── 우하단 LED 원 ──────────────────────────────
        fh, fw = frame.shape[:2]
        cv2.circle(frame, (fw - 24, fh - 24), 14, sc, -1)
        cv2.circle(frame, (fw - 24, fh - 24), 14, COLOR_WHITE, 1)

        # ── 스트리밍 프레임 업데이트 ───────────────────────
        if _set_frame is not None:
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            _set_frame(jpeg.tobytes())

        if headless:
            if not _running:
                break
        else:
            cv2.imshow("Baby Safety Monitor v4", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    print("[종료]")


if __name__ == "__main__":
    main()
