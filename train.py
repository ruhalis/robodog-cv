from ultralytics import YOLO

# Load the pre-trained model
model = YOLO("yolo11n.pt")

# Fine-tune on your custom dataset
model.train(
    data="C:\\Users\\ruhalis\\Documents\\robodog-cv\\datasets\\general\\data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0
)
model.eval()