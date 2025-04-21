#!/usr/bin/env python3
import os
import subprocess
import sys

def main():
    """Main function to run the entire annotation and YAML generation process"""
    # Check for required Python packages
    try:
        import numpy
        import cv2
        import PIL
        import yaml
    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("Please install the required packages using:")
        print("pip install numpy opencv-python pillow pyyaml")
        return 1
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if scripts exist
    annotation_script = os.path.join(current_dir, 'create_yolo_annotations.py')
    yaml_script = os.path.join(current_dir, 'create_dataset_yaml.py')
    
    if not os.path.exists(annotation_script):
        print(f"Error: Annotation script not found at {annotation_script}")
        return 1
    
    if not os.path.exists(yaml_script):
        print(f"Error: YAML script not found at {yaml_script}")
        return 1
    
    print("Starting annotation process...")
    
    # Run the annotation script
    try:
        print("Step 1: Creating YOLO annotations...")
        os.system(f'python "{annotation_script}"')
        
        print("\nStep 2: Creating dataset YAML configuration...")
        os.system(f'python "{yaml_script}"')
        
        print("\nAnnotation process completed successfully!")
        print("The YOLO dataset is ready at: train-curat-dataset-yolo/")
        print("You can use the dataset.yaml file with YOLOv11 for training.")
        
        return 0
    except Exception as e:
        print(f"Error during annotation process: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 