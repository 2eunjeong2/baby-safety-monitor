"""
================================================
 Baby Safety Monitor — YOLOv8 실시간 감지 + LED 제어
 파일명 : face_led_v2.py
 위치   : vision/face_led_v2.py
------------------------------------------------
 1회차 face_led.py (Haar Cascade) 에서
 YOLOv8 + Pose Estimation 으로 업그레이드

 감지 우선순위
   1순위 : 아기 미감지              → 'R' → RED    (자리 비움/가림)
   2순위 : 아기 감지 + 엎드린 자세  → 'R' → RED    (위험)
   3순위 : 아기 감지 + 옆으로 누움  → 'Y' → YELLOW (주의)
   4순위 : 아기 감지 + 바로 누움    → 'G' → GREEN  (정상)

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

# 웹캠 인덱스 (안 열리면 1로 변경)
CAM_INDEX = 0

# 학습된 모델 경로
MODEL_PATH = "ai/runs/baby_monitor/weights/best.pt"

# Pose Estimation 모델 (자세 판별용 - 별도 학습 불필요)
POSE_MODEL_PATH = "yolov8n-pose.pt"

# 시리얼 통신 속도
BAUD_RATE = 9600

# 감지 신뢰도 임계값 (낮추면 더 잘 잡지만 오탐 증가)
CONF_THRESHOLD = 0.5

# ROI 마스킹 비율 (모빌 장식 간섭 방지 - 가장자리 제외)
ROI_MARGIN_X = 0.10   # 좌우 10% 제외
ROI_MARGIN_Y = 0.05   # 상하 5% 제외

# LED 전송 쿨다운 (초)
SEND_COOLDOWN = 0.3

# 자세 판별 keypoint 인덱스 (COCO 기준)
# 0: nose, 5: left_shoulder, 6: right_shoulder
# 11: left_hip, 12: right_hip
KP_NOSE           = 0
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12

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
    """
    화면 가장자리를 제외한 중앙 ROI만 반환한다.
    Top view 마운트 시 모빌 암/장식이 가장자리에 걸리는 것을 방지.

    Returns:
        roi        : 마스킹된 프레임 (원본과 동일 크기)
        roi_offset : (x_offset, y_offset) 좌표 보정용
    """
    h, w = frame.shape[:2]
    x1 = int(w * ROI_MARGIN_X)
    y1 = int(h * ROI_MARGIN_Y)
    x2 = int(w * (1 - ROI_MARGIN_X))
    y2 = int(h * (1 - ROI_MARGIN_Y))

    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1)


# ════════════════════════════════════════════════
# 함수: 자세 판별 (Pose Estimation keypoint 기반)
# ════════════════════════════════════════════════
def classify_pose(keypoints):
    """
    YOLOv8 Pose keypoint 좌표로 자세를 판별한다.

    판별 기준
      - prone  (엎드림) : nose y좌표 > shoulder y좌표 평균
                          → 얼굴이 몸통보다 아래 = 엎드린 상태
      - side   (옆모습) : 좌우 shoulder x좌표 차이가 작음
                          → 어깨가 겹쳐 보임 = 옆으로 누운 상태
      - supine (바로누움): 그 외 정상 상태

    Args:
        keypoints : ultralytics keypoint 배열 (shape: [17, 3])

    Returns:
        'prone' | 'side' | 'supine'
    """
    try:
        kp = keypoints.xy[0].cpu().numpy()   # [17, 2] (x, y)
        conf = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else None

        nose           = kp[KP_NOSE]
        left_shoulder  = kp[KP_LEFT_SHOULDER]
        right_shoulder = kp[KP_RIGHT_SHOULDER]

        # 신뢰도 낮은 keypoint는 판별 제외
        if conf is not None:
            if conf[KP_NOSE] < 0.3 or conf[KP_LEFT_SHOULDER] < 0.3:
                return "supine"   # 불확실 → 안전하게 정상 처리

        shoulder_y_avg = (left_shoulder[1] + right_shoulder[1]) / 2
        shoulder_x_diff = abs(left_shoulder[0] - right_shoulder[0])

        # 엎드림 판별: nose가 어깨보다 아래 (Top view 기준)
        if nose[1] > shoulder_y_avg + 20:
            return "prone"

        # 옆모습 판별: 어깨 좌우 차이가 작음 (겹쳐 보임)
        frame_width_ref = 640
        if shoulder_x_diff < frame_width_ref * 0.08:
            return "side"

        return "supine"

    except Exception:
        return "supine"   # 오류 시 안전하게 정상 처리


# ════════════════════════════════════════════════
# 함수: 화면 오버레이 그리기
# ════════════════════════════════════════════════
def draw_overlay(frame, cmd, baby_boxes, pose_label, fps, roi_offset):
    h, w = frame.shape[:2]
    ox, oy = roi_offset

    # ── 아기 바운딩박스 ──
    color_map = {"G": COLOR_GREEN, "Y": COLOR_YELLOW, "R": COLOR_RED}
    box_color = color_map.get(cmd, COLOR_GRAY)

    for box in baby_boxes:
        x1, y1, x2, y2 = box
        # ROI 오프셋 보정
        x1, y1, x2, y2 = int(x1)+ox, int(y1)+oy, int(x2)+ox, int(y2)+oy
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, f"baby [{pose_label}]", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

    # ── ROI 영역 표시 (가장자리 점선) ──
    rx1 = int(w * ROI_MARGIN_X)
    ry1 = int(h * ROI_MARGIN_Y)
    rx2 = int(w * (1 - ROI_MARGIN_X))
    ry2 = int(h * (1 - ROI_MARGIN_Y))
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), COLOR_GRAY, 1)
    cv2.putText(frame, "ROI", (rx1 + 4, ry1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)

    # ── 상태 텍스트 ──
    status_map = {
        "G": "SAFE — 바로 누움",
        "Y": "CAUTION — 옆으로 누움",
        "R": "ALERT — 위험 감지!",
    }
    label  = status_map.get(cmd, "UNKNOWN")
    color  = color_map.get(cmd, COLOR_GRAY)

    # 반투명 상단 배경
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, label, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 1)

    # LED 상태 원 (우하단)
    cv2.circle(frame, (w - 24, h - 24), 14, color, -1)
    cv2.circle(frame, (w - 24, h - 24), 14, COLOR_WHITE, 1)

    return frame


# ════════════════════════════════════════════════
# 메인 함수
# ════════════════════════════════════════════════
def main():
    # ── 모델 로드 ──────────────────────────────
    print("[Model] YOLOv8 감지 모델 로드 중...")
    detect_model = YOLO(MODEL_PATH)

    print("[Model] YOLOv8 Pose 모델 로드 중...")
    pose_model = YOLO(POSE_MODEL_PATH)   # 자동 다운로드 (약 6MB)
    print("[Model] 모델 로드 완료\n")

    # ── 웹캠 연결 ──────────────────────────────
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[Error] 웹캠 인덱스 {CAM_INDEX} 를 열 수 없습니다.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"[Camera] 웹캠 연결 완료 (인덱스: {CAM_INDEX})")

    # ── 아두이노 시리얼 연결 ───────────────────
    port = find_arduino_port()
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"[Serial] 연결 완료 — {port} @ {BAUD_RATE}bps")
    except serial.SerialException as e:
        print(f"[Error] 시리얼 연결 실패: {e}")
        cap.release()
        sys.exit(1)

    # ── 변수 초기화 ────────────────────────────
    last_cmd       = ""
    last_send_time = 0.0
    fps            = 0.0
    prev_tick      = cv2.getTickCount()

    print("[System] 실행 중... 'q' 키로 종료")
    print("-" * 50)

    # ════════════════════════════════════════════
    # 메인 루프
    # ════════════════════════════════════════════
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] 프레임을 읽을 수 없습니다.")
            break

        # ── FPS 계산 ──────────────────────────
        curr_tick = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (curr_tick - prev_tick)
        prev_tick = curr_tick

        # ── ROI 마스킹 ────────────────────────
        roi, roi_offset = get_roi(frame)

        # ── YOLOv8 아기 감지 ──────────────────
        detect_results = detect_model(
            roi,
            conf=CONF_THRESHOLD,
            verbose=False
        )

        baby_boxes = []
        pose_label = "unknown"
        cmd        = "R"   # 기본값: 미감지 → RED

        # 감지된 객체가 있으면 자세 판별
        if len(detect_results[0].boxes) > 0:
            # 가장 큰 바운딩박스 = 가장 가까운 아기
            boxes = detect_results[0].boxes.xyxy.cpu().numpy()
            areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in boxes]
            best_idx = int(np.argmax(areas))
            baby_boxes = [boxes[best_idx]]

            # ── YOLOv8 Pose Estimation ────────
            pose_results = pose_model(
                roi,
                conf=CONF_THRESHOLD,
                verbose=False
            )

            if (len(pose_results[0].keypoints) > 0 and
                    pose_results[0].keypoints.xy.shape[1] > 0):
                pose_label = classify_pose(pose_results[0].keypoints)
            else:
                pose_label = "supine"   # pose 미감지 → 안전 처리

            # 자세 → 명령어 매핑
            if pose_label == "prone":
                cmd = "R"   # 엎드림 → RED (위험)
            elif pose_label == "side":
                cmd = "Y"   # 옆모습 → YELLOW (주의)
            else:
                cmd = "G"   # 바로누움 → GREEN (정상)

        # ── 아두이노로 명령 전송 ────────────────
        last_cmd, last_send_time = send_command(
            ser, cmd, last_cmd, last_send_time
        )

        # ── 화면 오버레이 ───────────────────────
        frame = draw_overlay(
            frame, cmd, baby_boxes, pose_label, fps, roi_offset
        )

        # ── 영상 출력 ───────────────────────────
        cv2.imshow("Baby Safety Monitor v2", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[System] 종료 명령 수신")
            break

    # ── 종료 처리 ──────────────────────────────
    ser.write(b"X")
    time.sleep(0.1)
    ser.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[System] 정상 종료 완료")


# ════════════════════════════════════════════════
if __name__ == "__main__":
    main()