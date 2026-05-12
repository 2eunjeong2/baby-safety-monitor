# 2회차 구현 내용

### 1. YOLOv8n v1 학습 — 아기 감지 (1클래스)

Haar Cascade 기반에서 딥러닝 모델로 전환하기 위해 YOLOv8n을 Google Colab에서 학습했다.

| 항목 | 내용 |
|------|------|
| 데이터셋 | Roboflow `baby-object-detection-final` (baby 1클래스) |
| 학습 환경 | Google Colab T4 GPU |
| epochs / imgsz / batch | 30 / 640 / 16 |
| mAP50 | 0.957 |
| mAP50-95 | 0.693 |
| Precision / Recall | 0.928 / 0.924 |

---

### 2. Haar Cascade → YOLOv8 교체 (`vision/face_led_v2.py`)

기존 OpenCV Haar Cascade 방식을 YOLOv8 Pose Estimation으로 교체하여 자세 판별을 시도했다.

**자세 판별 로직 (코+눈 keypoint 기반)**

```
웹캠 영상
  ├── keypoint 없음 / 코 미감지 → prone (RED)
  ├── 코 + 눈 한쪽만 감지        → side  (YELLOW)
  └── 코 + 눈 양쪽 모두 감지    → supine (GREEN)
```

**주요 구현 포인트**

- ROI 마스킹으로 모빌 장식 간섭 방지
- `CONF_THRESHOLD` 0.5 → 0.15 조정 (낮은 높이에서 keypoint 감지율 개선)
- 불확실 상황 무조건 RED 처리 (안전 우선)

---

### 3. YOLOv8n v2 학습 — 자세 4클래스

pose keypoint 방식의 한계를 보완하기 위해 자세를 직접 분류하는 4클래스 모델을 학습했다.

| 항목 | 내용 |
|------|------|
| 데이터셋 | Roboflow `baby-face-hnza2` (nose / prone / sideways / suspine) |
| 학습 환경 | Google Colab T4 GPU |
| epochs | 50 |
| 전체 mAP50 | 0.656 |
| 클래스별 mAP50 | nose: 0.346 / prone: 0.789 / sideways: 0.706 / suspine: 0.782 |

→ nose 클래스 정확도가 낮아 OpenCV 규칙 로직 보완 예정으로 남겨둠

---

### 4. Baby Safety Monitor v4 — 완전 재구성

4클래스 모델의 한계를 분석한 후, 3클래스(정면/측면/후면)로 재정의하고 새 데이터셋으로 처음부터 재구성했다.

**`vision/baby_monitor_v4.py`**

| 항목 | 내용 |
|------|------|
| 감지 클래스 | 정면(0) / 측면(1) / 후면(2) |
| 추론 주기 | `FRAME_SKIP=2` — 2프레임마다 1회 추론 |
| 시간적 평활화 | 최근 20프레임 다수결 (`SMOOTH_WIN=20`) |
| 후면 우선 규칙 | 25% 이상 + 최소 3회 감지 시 후면 확정 |
| 알림 타이머 | 후면·미감지 타이머 분리 관리 (`BACK_ALERT=5.0초`) |
| 측면 오탐 보정 | bbox W/H > 1.7 이면 측면→정면 교정 (`ASPECT_FLAT`) |

```python
# 후면 안전 우선 로직
if back_count >= BACK_MIN_COUNT and back_count >= len(history) * 0.25:
    return 2  # 후면 확정
```

**`ai/train_v4.py` — 프로덕션 학습 스크립트**

- epochs 200 / patience 20 / device MPS
- 데이터 증강: `degrees=45`, `perspective=0.0005`, `flipud=0.5`
- 후면 클래스 손실 가중치 상향 (`cls=1.5`, `box=7.5`)
- `last.pt` 존재 시 이어서 학습, `--fresh` 플래그로 완전 재학습 지원

**`ai/download_dataset.py` — 다중 데이터셋 병합 도구**

- Roboflow API로 SabiCare + VTAR + 오버헤드 3종 다운로드
- 클래스 ID 자동 재매핑 및 `data.yaml` 생성

**`colab_train.ipynb`**

- Google Colab T4 GPU 학습용 노트북
- 환경 설치 → 데이터셋 다운로드 → 학습 → best.pt 다운로드까지 원클릭

---

### 5. 감지 데모 스크린샷

| 정면 (안전) | 측면 (주의) | 후면 (위험) |
|:-----------:|:-----------:|:-----------:|
| ![정면](assets/detection_front.jpg) | ![측면](assets/detection_side.jpg) | ![후면](assets/detection_back.jpg) |

---

## 🔍 트러블슈팅

| 문제 | 원인 | 해결 |
|------|------|------|
| 측면 오탐 — 대각선 정면 자세가 측면으로 분류됨 | bbox 비율이 가로로 긴 경우 측면 레이블과 혼동 | `ASPECT_FLAT=1.7` 임계값으로 bbox W/H 검사 후 정면으로 교정 |
| 후면 초기 과민 — 1~2프레임만으로 경보 발생 | 히스토리가 적을 때 1회 감지만으로 25% 초과 | `BACK_MIN_COUNT=3` 최소 감지 횟수 조건 추가 |
| 영상 루프 재시작 시 타이머 오작동 | `cap.set()` 으로 프레임 되돌려도 상태 변수 유지 | `reset_state()` 함수로 타이머·히스토리 전체 초기화 |

---

## 📐 시스템 구성도 (v4)

```
[맥북 M1]
  baby_monitor_v4.py
  ├── YOLOv8n (best.pt)       ← 3클래스 직접 분류
  │    정면 / 측면 / 후면
  ├── 시간적 평활화 (20프레임 다수결)
  ├── 후면 타이머 (5초 초과 → 위험 경보)
  └── 화면 오버레이 (상태 텍스트 / FPS / LED 원)
              ↑
          [웹캠] — 아기 침대 오버헤드 설치
```
