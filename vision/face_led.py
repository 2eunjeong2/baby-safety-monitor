"""
================================================
 웹캠 영상에서 얼굴을 감지하여
 아두이노 LED 색상으로 상태를 표시한다.

 감지 우선순위
   1순위 : 정면 얼굴 감지  → 'G' 전송 → GREEN LED
   2순위 : 옆면 얼굴 감지  → 'Y' 전송 → YELLOW LED
   3순위 : 미감지           → 'R' 전송 → RED LED

 실행 방법
   source venv/bin/activate
   python3 vision/face_led.py

 종료
   영상 창에서 'q' 키 입력
================================================
"""

import cv2
import serial
import serial.tools.list_ports
import time
import sys

# ── 설정값 ───────────────────────────────────────

# 웹캠 인덱스 (안 열리면 1로 변경)
CAM_INDEX = 0

# 시리얼 통신 속도 (led_control.ino 와 반드시 일치)
BAUD_RATE = 9600

# 얼굴 감지 스케일 (작을수록 정확하지만 느림 / 권장: 1.1~1.3)
SCALE_FACTOR = 1.1

# 최소 이웃 수 (높을수록 오탐 감소 / 권장: 4~6)
MIN_NEIGHBORS = 5

# 감지 최소 얼굴 크기 (픽셀) — 너무 작으면 노이즈 오탐
MIN_FACE_SIZE = (60, 60)

# LED 전송 쿨다운 (초) — 같은 상태 반복 전송 방지
SEND_COOLDOWN = 0.3

# ── Haar Cascade 파일 경로 ────────────────────────
# OpenCV 설치 시 함께 포함된 기본 파일 사용
CASCADE_FRONTAL = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CASCADE_PROFILE  = cv2.data.haarcascades + "haarcascade_profileface.xml"

# ── 오버레이 색상 (BGR) ───────────────────────────
COLOR_GREEN  = (50,  200,  50)
COLOR_YELLOW = (0,   210, 210)
COLOR_RED    = (50,   50, 220)
COLOR_WHITE  = (255, 255, 255)
COLOR_GRAY   = (160, 160, 160)


# ════════════════════════════════════════════════
# 함수: 아두이노 포트 자동 탐색
# ════════════════════════════════════════════════
def find_arduino_port():
    """
    연결된 시리얼 포트 중 아두이노로 보이는 포트를 자동으로 찾는다.
    맥에서는 /dev/cu.usbmodem... 형태로 나타난다.
    """
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # 맥 기준 아두이노 포트 패턴
        if "usbmodem" in port.device or "usbserial" in port.device:
            print(f"[Serial] 아두이노 포트 자동 감지: {port.device}")
            return port.device

    # 자동 탐색 실패 시 수동 입력 안내
    print("[Serial] 아두이노 포트를 자동으로 찾지 못했습니다.")
    print("[Serial] 연결된 포트 목록:")
    for port in ports:
        print(f"         {port.device} — {port.description}")
    print("[Serial] 포트를 직접 입력하세요 (예: /dev/cu.usbmodem14101)")
    return input("포트 입력: ").strip()


# ════════════════════════════════════════════════
# 함수: 시리얼 명령 전송 (쿨다운 적용)
# ════════════════════════════════════════════════
def send_command(ser, cmd, last_cmd, last_time):
    """
    아두이노로 1바이트 명령어를 전송한다.
    같은 명령이 SEND_COOLDOWN 초 이내에 반복되면 전송하지 않는다.

    Args:
        ser       : serial.Serial 객체
        cmd       : 전송할 명령어 문자 ('G', 'Y', 'R', 'X')
        last_cmd  : 직전에 전송한 명령어
        last_time : 직전 전송 시각 (time.time())

    Returns:
        (last_cmd, last_time) — 업데이트된 값
    """
    now = time.time()

    # 같은 명령이고 쿨다운 미경과 → 전송 생략
    if cmd == last_cmd and (now - last_time) < SEND_COOLDOWN:
        return last_cmd, last_time

    # 시리얼로 명령 전송
    ser.write(cmd.encode())
    return cmd, now


# ════════════════════════════════════════════════
# 함수: 화면 오버레이 그리기
# ════════════════════════════════════════════════
def draw_overlay(frame, status, frontal_faces, profile_faces, fps):
    """
    감지 결과를 프레임 위에 시각적으로 표시한다.

    - 바운딩 박스: 정면(초록), 옆면(노랑)
    - 상태 텍스트: 좌상단
    - FPS: 우상단
    """
    h, w = frame.shape[:2]

    # ── 정면 얼굴 바운딩 박스 ──
    for (x, y, fw, fh) in frontal_faces:
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), COLOR_GREEN, 2)
        cv2.putText(frame, "Front", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GREEN, 2)

    # ── 옆면 얼굴 바운딩 박스 ──
    for (x, y, pw, ph) in profile_faces:
        cv2.rectangle(frame, (x, y), (x + pw, y + ph), COLOR_YELLOW, 2)
        cv2.putText(frame, "Profile", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_YELLOW, 2)

    # ── 상태별 텍스트 & 색상 ──
    status_map = {
        "G": ("FACE DETECTED (Front)",   COLOR_GREEN),
        "Y": ("FACE DETECTED (Profile)", COLOR_YELLOW),
        "R": ("NO FACE — CHECK BABY",    COLOR_RED),
    }
    label, color = status_map.get(status, ("UNKNOWN", COLOR_GRAY))

    # 반투명 배경 박스
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # 상태 텍스트
    cv2.putText(frame, label, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    # FPS 우상단
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 1)

    # LED 상태 표시 (우하단 작은 원)
    led_colors = {"G": COLOR_GREEN, "Y": COLOR_YELLOW, "R": COLOR_RED}
    cv2.circle(frame, (w - 24, h - 24), 14, led_colors.get(status, COLOR_GRAY), -1)
    cv2.circle(frame, (w - 24, h - 24), 14, COLOR_WHITE, 1)

    return frame


# ════════════════════════════════════════════════
# 메인 함수
# ════════════════════════════════════════════════
def main():
    # ── Haar Cascade 로드 ──────────────────────
    frontal_cascade = cv2.CascadeClassifier(CASCADE_FRONTAL)
    profile_cascade  = cv2.CascadeClassifier(CASCADE_PROFILE)

    if frontal_cascade.empty() or profile_cascade.empty():
        print("[Error] Haar Cascade XML 파일을 불러오지 못했습니다.")
        print("        OpenCV가 정상 설치되었는지 확인하세요.")
        sys.exit(1)

    print("[OpenCV] Haar Cascade 로드 완료")

    # ── 웹캠 연결 ──────────────────────────────
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"[Error] 웹캠 인덱스 {CAM_INDEX} 를 열 수 없습니다.")
        print("        CAM_INDEX 를 1 로 변경해보세요.")
        sys.exit(1)

    # 해상도 설정 (처리 속도 최적화)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"[Camera] 웹캠 연결 완료 (인덱스: {CAM_INDEX})")

    # ── 아두이노 시리얼 연결 ───────────────────
    port = find_arduino_port()
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        # 아두이노 리셋 대기 (연결 직후 약 2초 필요)
        time.sleep(2)
        print(f"[Serial] 연결 완료 — {port} @ {BAUD_RATE}bps")
    except serial.SerialException as e:
        print(f"[Error] 시리얼 연결 실패: {e}")
        cap.release()
        sys.exit(1)

    # ── 변수 초기화 ────────────────────────────
    last_cmd      = ""        # 직전 전송 명령어
    last_send_time = 0.0      # 직전 전송 시각
    fps           = 0.0       # FPS
    prev_tick     = cv2.getTickCount()

    print("[System] 실행 중... 'q' 키로 종료")
    print("-" * 44)

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

        # ── 전처리: 그레이스케일 변환 ──────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 대비 향상 (야간/저조도 환경 대비)
        gray = cv2.equalizeHist(gray)

        # ── 1순위: 정면 얼굴 감지 ──────────────
        frontal_faces = frontal_cascade.detectMultiScale(
            gray,
            scaleFactor  = SCALE_FACTOR,
            minNeighbors = MIN_NEIGHBORS,
            minSize      = MIN_FACE_SIZE,
            flags        = cv2.CASCADE_SCALE_IMAGE
        )

        # ── 2순위: 옆면 얼굴 감지 ──────────────
        # 정면이 이미 감지된 경우 옆면 감지는 건너뜀 (연산 절약)
        if len(frontal_faces) > 0:
            profile_faces = []
            cmd = "G"
        else:
            profile_faces = profile_cascade.detectMultiScale(
                gray,
                scaleFactor  = SCALE_FACTOR,
                minNeighbors = MIN_NEIGHBORS,
                minSize      = MIN_FACE_SIZE,
                flags        = cv2.CASCADE_SCALE_IMAGE
            )
            # 좌우 반전 후 한 번 더 감지 (옆면 cascade는 한쪽만 학습됨)
            gray_flip = cv2.flip(gray, 1)
            profile_faces_flip = profile_cascade.detectMultiScale(
                gray_flip,
                scaleFactor  = SCALE_FACTOR,
                minNeighbors = MIN_NEIGHBORS,
                minSize      = MIN_FACE_SIZE,
                flags        = cv2.CASCADE_SCALE_IMAGE
            )
            # 반전 좌표를 원본 좌표계로 복원
            fw = frame.shape[1]
            for (x, y, pw, ph) in profile_faces_flip:
                corrected = (fw - x - pw, y, pw, ph)
                profile_faces = list(profile_faces) + [corrected]

            cmd = "Y" if len(profile_faces) > 0 else "R"

        # ── 아두이노로 명령 전송 ────────────────
        last_cmd, last_send_time = send_command(
            ser, cmd, last_cmd, last_send_time
        )

        # ── 화면 오버레이 ───────────────────────
        frontal_list = frontal_faces if len(frontal_faces) > 0 else []
        frame = draw_overlay(frame, cmd, frontal_list, profile_faces, fps)

        # ── 영상 출력 ───────────────────────────
        cv2.imshow("Baby Safety Monitor", frame)

        # 'q' 키 입력 시 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[System] 종료 명령 수신")
            break

    # ── 종료 처리 ──────────────────────────────
    ser.write(b"X")          # LED 전체 OFF
    time.sleep(0.1)
    ser.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[System] 정상 종료 완료")


# ════════════════════════════════════════════════
if __name__ == "__main__":
    main()