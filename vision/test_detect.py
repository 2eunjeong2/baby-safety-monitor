# vision/test_detect.py
from ultralytics import YOLO
import cv2

model = YOLO("ai/runs/baby_monitor/weights/best.pt")
cap = cv2.VideoCapture("ai/sample.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25, verbose=True)
    annotated = results[0].plot()  # 바운딩박스 자동 그려줌

    cv2.imshow("Test", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()