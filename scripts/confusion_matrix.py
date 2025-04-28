# scripts/confusion_matrix.py

from ultralytics import YOLO
import yaml
from multiprocessing import freeze_support

def main():
    with open(r'C:\Users\ruhalis\Documents\yolo-detection-tracking\datasets\final\data.yaml') as f:
        data = yaml.safe_load(f)
    print(data['names'])
    selected = [ data['names'].index(c) for c in [
        'crack','fire_extinguisher','light_off','light_on',
        'half_working_light','algae','peeling','stain','moisture'
    ] ]
    model = YOLO('runs/detect/train2/weights/best.pt')
    results = model.val(
        data=r'C:\Users\ruhalis\Documents\yolo-detection-tracking\datasets\final\data.yaml',
        conf=0.25,
        iou=0.5,
        plots=True,
        classes=selected
    )
    print("Done – see runs/val/exp*/plots/confusion_matrix.png")

if __name__ == "__main__":
    freeze_support()
    main()