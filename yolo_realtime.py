from ultralytics import YOLO
import cv2

# Load YOLOv5/8 model (downloads automatically)
model = YOLO("yolov8n.pt")  # you can use 'yolov5s.pt' if preferred

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model.predict(source=frame, verbose=False)

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("Real-time YOLO Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
