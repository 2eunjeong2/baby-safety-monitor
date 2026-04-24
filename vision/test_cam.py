import cv2

cap = cv2.VideoCapture(0)  # 안 되면 1로 변경

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라를 찾을 수 없어요")
        break
    cv2.imshow('Camera Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()