"""
================================================
 Baby Safety Monitor — YOLOv8 실시간 감지 + LED 제어
 파일명 : face_led_v2.py
 위치   : vision/face_led_v2.py
------------------------------------------------
 감지 우선순위
   1순위 : 아기 미감지                  → 'R' → RED    (자리 비움/가림)
   2순위 : 아기 감지 + keypoint 없음    → 'R' → RED    (뒷모습 의심)
   3순위 : 아기 감지 + 코 미감지        → 'R' → RED    (엎드림)
   4순위 : 아기 감지 + 코+눈 한쪽 감지 → 'Y' → YELLOW (측면)
   5순위 : 아기 감지 + 코+눈 양쪽 감지 → 'G' → GREEN  (정면 / 정상)

 불확실한 상황은 무조건 RED 처리
 → 오탐(잘못된 GREEN)이 미탐(잘못된 RED)보다 위험하기 때문

 실행 방법
   source venv/bin/activate
   python3 vision/face_led_v2.py

 종료
   영상 창에서 'q' 키 입력
================================================
"""

import cv2
import serial
import serial.tools.list_ports
import time
import sys
import numpy as np
from ultralytics import YOLO

# ── 설정값 ───────────────────────────────────────

# 웹캠 인덱스 (영상 파일 쓸 땐 경로로 교체)
# 예) CAM_INDEX = "ai/sample.mp4"
CAM_INDEX = 0

# 학습된 아기 감지 모델
MODEL_PATH = "ai/runs/baby_monitor_v1/weights/best.pt"

# Pose Estimation 모델 (자세 판별용)
POSE_MODEL_PATH = "yolov8n-pose.pt"

# 시리얼 통신 속도
BAUD_RATE = 9600

# 감지 신뢰도 임계값
CONF_THRESHOLD = 0.7   # 낮출수록 민감 (오탐 증가 주의)

# ROI 마스킹 비율 (모빌 장식 간섭 방지)
ROI_MARGIN_X = 0.10
ROI_MARGIN_Y = 0.05

# LED 전송 쿨다운 (초)
SEND_COOLDOWN = 0.3

# keypoint 신뢰도 임계값
KP_CONF = 0.3

# keypoint 인덱스 (COCO 기준)
KP_NOSE      = 0
KP_LEFT_EYE  = 1
KP_RIGHT_EYE = 2

# ── 오버레이 색상 (BGR) ───────────────────────────
COLOR_GREEN  = (50,  200,  50)
COLOR_YELLOW = (0,   200, 210)
COLOR_RED    = (50,   50, 220)
COLOR_WHITE  = (255, 255, 255)
COLOR_GRAY   = (160, 160, 160)


# ════════════════════════════════════════════════
# 함수: 아두이노 포트 자동 탐색
# ════════════════════════════════════════════════
def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "usbmodem" in port.device or "usbserial" in port.device:
            print(f"[Serial] 아두이노 포트 자동 감지: {port.device}")
            return port.device

    print("[Serial] 포트 자동 탐색 실패. 연결된 포트 목록:")
    for port in ports:
        print(f"         {port.device} — {port.description}")
    return input("포트 직접 입력: ").strip()


# ════════════════════════════════════════════════
# 함수: 시리얼 명령 전송 (쿨다운 적용)
# ════════════════════════════════════════════════
def send_command(ser, cmd, last_cmd, last_time):
    now = time.time()
    if cmd == last_cmd and (now - last_time) < SEND_COOLDOWN:
        return last_cmd, last_time
    ser.write(cmd.encode())
    return cmd, now


# ════════════════════════════════════════════════
# 함수: ROI 마스킹 (모빌 장식 간섭 방지)
# ════════════════════════════════════════════════
def get_roi(frame):
    h, w = frame.shape[:2]
    x1 = int(w * ROI_MARGIN_X)
    y1 = int(h * ROI_MARGIN_Y)
    x2 = int(w * (1 - ROI_MARGIN_X))
    y2 = int(h * (1 - ROI_MARGIN_Y))
    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1)


# ════════════════════════════════════════════════
# 함수: 자세 판별 (코 + 눈 keypoint 가시성 기반)
# ════════════════════════════════════════════════
def classify_pose(keypoints):
    """
    코(nose) + 눈(left_eye, right_eye) keypoint 가시성으로 자세 판별

    판별 기준
      코 미감지                     → prone  (엎드림/뒷모습) → RED
      코 감지 + 눈 한쪽만 감지      → side   (측면)          → YELLOW
      코 감지 + 눈 양쪽 감지        → supine (정면/정상)      → GREEN
      keypoint 신뢰도 낮음 / 오류   → prone  (안전 처리)      → RED
    """
    try:
        if keypoints.conf is None:
            return "prone"

        conf = keypoints.conf[0].cpu().numpy()

        nose_visible      = conf[KP_NOSE]      > KP_CONF
        left_eye_visible  = conf[KP_LEFT_EYE]  > KP_CONF
        right_eye_visible = conf[KP_RIGHT_EYE] > KP_CONF
        both_eyes_visible = left_eye_visible and right_eye_visible
        one_eye_visible   = left_eye_visible or right_eye_visible

        if not nose_visible:
            return "prone"   # 코 미감지 → 엎드림/뒷모습

        if nose_visible and one_eye_visible and not both_eyes_visible:
            return "side"    # 코 + 눈 한쪽 → 측면

        if nose_visible and both_eyes_visible:
            return "supine"  # 코 + 눈 양쪽 → 정면

        return "prone"       # 불확실 → 안전하게 RED

    except Exception as e:
        print(f"[Pose] 판별 오류: {e}")
        return "prone"


# ════════════════════════════════════════════════
# 함수: 화면 오버레이 그리기
# ════════════════════════════════════════════════
def draw_overlay(frame, cmd, baby_boxes, pose_label, fps, roi_offset):
    h, w = frame.shape[:2]
    ox, oy = roi_offset

    color_map = {"G": COLOR_GREEN, "Y": COLOR_YELLOW, "R": COLOR_RED}
    box_color = color_map.get(cmd, COLOR_GRAY)

    # ── 아기 바운딩박스 ──
    for box in baby_boxes:
        x1 = int(box[0]) + ox
        y1 = int(box[1]) + oy
        x2 = int(box[2]) + ox
        y2 = int(box[3]) + oy
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, f"baby [{pose_label}]", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

    # ── ROI 영역 표시 ──
    rx1 = int(w * ROI_MARGIN_X)
    ry1 = int(h * ROI_MARGIN_Y)
    rx2 = int(w * (1 - ROI_MARGIN_X))
    ry2 = int(h * (1 - ROI_MARGIN_Y))
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), COLOR_GRAY, 1)
    cv2.putText(frame, "ROI", (rx1 + 4, ry1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)

    # ── 상태 텍스트 ──
    status_map = {
        "G": "SAFE — 정면 감지",
        "Y": "CAUTION — 측면 감지",
        "R": "ALERT — 위험 / 미감지",
    }
    label = status_map.get(cmd, "UNKNOWN")
    color = color_map.get(cmd, COLOR_GRAY)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, label, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 1)

    cv2.circle(frame, (w - 24, h - 24), 14, color, -1)
    cv2.circle(frame, (w - 24, h - 24), 14, COLOR_WHITE, 1)

    return frame


# ════════════════════════════════════════════════
# 메인 함수
# ════════════════════════════════════════════════
def main():
    print("[Model] YOLOv8 감지 모델 로드 중...")
    detect_model = YOLO(MODEL_PATH)

    print("[Model] YOLOv8 Pose 모델 로드 중...")
    pose_model = YOLO(POSE_MODEL_PATH)
    print("[Model] 모델 로드 완료\n")

    cap = cv2.VideoCapture("ai/sample2.mp4")
    if not cap.isOpened():
        print(f"[Error] 영상 소스 {CAM_INDEX} 를 열 수 없습니다.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"[Camera] 영상 소스 연결 완료: {CAM_INDEX}")

    port = find_arduino_port()
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"[Serial] 연결 완료 — {port} @ {BAUD_RATE}bps")
    except serial.SerialException as e:
        print(f"[Error] 시리얼 연결 실패: {e}")
        cap.release()
        sys.exit(1)

    last_cmd       = ""
    last_send_time = 0.0
    fps            = 0.0
    prev_tick      = cv2.getTickCount()

    print("[System] 실행 중... 'q' 키로 종료")
    print("-" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Info] 영상 종료 또는 프레임 읽기 실패")
            break

        curr_tick = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (curr_tick - prev_tick)
        prev_tick = curr_tick

        roi, roi_offset = get_roi(frame)

        detect_results = detect_model(roi, conf=CONF_THRESHOLD, verbose=False)

        baby_boxes = []
        pose_label = "unknown"
        cmd        = "R"

        if len(detect_results[0].boxes) > 0:
            boxes    = detect_results[0].boxes.xyxy.cpu().numpy()
            areas    = [(x2-x1)*(y2-y1) for x1, y1, x2, y2 in boxes]
            best_idx = int(np.argmax(areas))
            baby_boxes = [boxes[best_idx]]

            pose_results = pose_model(roi, conf=CONF_THRESHOLD, verbose=False)

            has_keypoints = (
                len(pose_results[0].keypoints) > 0 and
                pose_results[0].keypoints.xy.shape[1] > 0
            )

            if has_keypoints:
                pose_label = classify_pose(pose_results[0].keypoints)
            else:
                # keypoint 없음 → 뒷모습 의심 → RED
                pose_label = "prone"
                print("[Pose] keypoint 미감지 — 뒷모습 의심 → RED")

            cmd = "R" if pose_label == "prone" else \
                  "Y" if pose_label == "side"  else "G"

        last_cmd, last_send_time = send_command(
            ser, cmd, last_cmd, last_send_time
        )

        frame = draw_overlay(
            frame, cmd, baby_boxes, pose_label, fps, roi_offset
        )

        cv2.imshow("Baby Safety Monitor v2", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[System] 종료 명령 수신")
            break

    ser.write(b"X")
    time.sleep(0.1)
    ser.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[System] 정상 종료 완료")


if __name__ == "__main__":
    main()