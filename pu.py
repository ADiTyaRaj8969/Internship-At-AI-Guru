from ultralytics import YOLO
import cv2
import numpy as np
import random

# Load trained YOLOv8 model for vest detection
vest_model = YOLO("runs/detect/vest_detector_cpu/weights/best.pt")

# Load standard YOLOv8 model for person detection
person_model = YOLO('yolov8n.pt')  # Standard COCO model (includes person class)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Live vest detection started... Press 'Q' to quit.")

# Generate distinct colors for persons
def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        # Generate colors with different hues
        hue = i * 180 // n  # Spread hues across the spectrum
        color_hsv = np.uint8([[[hue, 255, 220]]])  # High saturation, medium value
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
        colors.append(tuple(int(c) for c in color_bgr[0,0]))
    return colors

# Pre-generate 20 distinct colors
distinct_colors = generate_distinct_colors(20)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Reset person counter for this frame
    person_counter = 1
    
    # Run person detection
    person_results = person_model.predict(frame, classes=[0], conf=0.5)  # Class 0 = person
    
    # Process person detections
    person_boxes = []
    for r in person_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_boxes.append((x1, y1, x2, y2))
    
    # Run vest detection
    vest_results = vest_model.predict(frame, conf=0.5)
    vest_boxes = []
    for r in vest_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vest_boxes.append((x1, y1, x2, y2))
    
    # Track vest status for each person
    person_vest_status = [False] * len(person_boxes)
    
    # Associate vests with persons
    for vest_box in vest_boxes:
        vx1, vy1, vx2, vy2 = vest_box
        vest_center = ((vx1 + vx2) // 2, (vy1 + vy2) // 2)
        
        for i, person_box in enumerate(person_boxes):
            px1, py1, px2, py2 = person_box
            # Check if vest center is inside person bounding box
            if px1 <= vest_center[0] <= px2 and py1 <= vest_center[1] <= py2:
                person_vest_status[i] = True
                break
    
    # Draw results with person numbering and unique colors
    for i, (x1, y1, x2, y2) in enumerate(person_boxes):
        status = "Vest Detected" if person_vest_status[i] else "No Vest"
        
        # Get unique color for this person (cycle through if more than 20 persons)
        color = distinct_colors[(person_counter - 1) % len(distinct_colors)]
        
        # Draw person bounding box (thicker if vest detected)
        thickness = 4 if person_vest_status[i] else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Create text label
        label = f"Person {person_counter}: {status}"
        
        # Draw background for text
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Increment person counter for this frame only
        person_counter += 1

    # Show the frame
    cv2.imshow("Live Vest Detection", frame)

    # Press 'Q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
