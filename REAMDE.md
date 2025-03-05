python scripts/convert_box_annotation.py --dataset_dir cracksdataset --output_dataset_dir datasets

Converts dataset from different shapes to  rectangular shape


python scripts/merge_3_datasets.py --dataset1 datasets/cracks --dataset2 datasets/light --dataset3 datasets/fire --output datasets/general


python scripts/changing_labels.py --dataset1 light --output datasets/light   

python scripts/merging_datasets_fire.py --dataset1 fire1 --dataset2 fire2 --output datasets/fire



.\venv\Scripts\activate


yolo detect train model=yolo11n.pt data=C:\Users\ruhalis\Documents\robodog-cv\datasets\general\data.yaml epochs=100 imgsz=640 batch=8 device=0