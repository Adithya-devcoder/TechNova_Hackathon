
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit(1)

print("Webcam is working! Press Q to close.")

while True:
    success, frame = cap.read()

    if not success:
        print("Error: Could not read frame.")
        break

    cv2.imshow("SignBridge - Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
