from ultralytics import YOLO
import cv2

# Load trained YOLOv8 model
model = YOLO("runs/detect/vest_detector_cpu/weights/best.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Error: Cannot open webcam")
    exit()

print("✅ Live vest detection started... Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run prediction (without auto GUI popups)
    results = model.predict(frame, conf=0.5, stream=True)

    # Draw bounding boxes on the frame
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f'{label} {conf:.2f}',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Show the frame
    cv2.imshow("Live Vest Detection", frame)

    # Press 'Q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
