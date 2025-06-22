from ultralytics import YOLO
model = YOLO('yolov8n.pt')
dataset_yaml_path = r"D:\Internship\AI GURU Internship\vest-detector\vest_project_yolo_ready\dataset.yaml"
model.train(
    data=dataset_yaml_path,
    epochs=30,
    imgsz=640,
    batch=16,
    name='vest_detector_cpu',
    device='cpu' 
)
