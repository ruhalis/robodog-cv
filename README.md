# YOLO Detection and Tracking ready repo

## Overview
This project provides tools for object detection using YOLOv11 for recognizing defects and infrastructure elements including cracks, fire equipment, lights, and moisture. It includes scripts for dataset preparation, merging, and training.

## Dataset Structure
The project supports YOLO format datasets with the following structure:
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Installation

1. Create and activate a virtual environment:
```
python -m venv venv
.\venv\Scripts\activate
source venv/bin/activate
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Scripts

### Dataset Preparation

- **Convert Box Annotations**: Convert annotations to rectangular format
  ```
  python scripts/convert_box_annotation.py --dataset_dir cracksdataset --output_dataset_dir datasets
  ```

- **Split Dataset**: Split into train/test/val with 80/10/10 proportions
  ```
  python scripts/split_dataset.py
  ```

- **Create YOLO Annotations**: Generate annotations for defect classes
  ```
  python scripts/run_annotation.py
  # uses create_dataset_yaml.py and create_yolo_annotations.py
  ```

### Dataset Merging

- **Merge Three Datasets**:
  ```
  python scripts/merge_3_datasets.py --dataset1 datasets/cracks --dataset2 datasets/light --dataset3 datasets/fire --output datasets/general
  ```

- **Merge Four Datasets**:
  ```
  python scripts/merge_4_datasets.py --dataset1 datasets/cracks --dataset2 datasets/light --dataset3 datasets/fire --dataset4 datasets/fire --output datasets/general
  ```

- **Modify Dataset Labels**:
  ```
  python scripts/changing_labels.py --dataset_dir datasets/light --output datasets/light_modified
  ```

## Training

Train the YOLOv11 model:
```
python train.py --data datasets/light5_split/data.yaml --weights yolo11s.pt
```

Alternatively, use the ultralytics YOLO command:
```
yolo detect train model=yolo11s.pt data=datasets/endterm_final/data.yaml epochs=100 imgsz=640 batch=8 device=0
```


## Model Files

- `yolo11s.pt`: YOLOv11 small model (for training)
- `yolo11m.pt`: YOLOv11 medium model (for training)
