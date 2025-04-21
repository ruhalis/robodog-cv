#!/usr/bin/env python3
import os
import shutil
from PIL import Image
import cv2
import numpy as np

# Define the class mapping
class_mapping = {
    'algae': 0,
    'peeling': 1,
    'stain': 2
}

# Define the source and destination paths
source_base_dir = 'train-curat-dataset'
dest_base_dir = 'train-curat-dataset-yolo'

# Create required directories if they don't exist
os.makedirs(dest_base_dir, exist_ok=True)

def process_image(image_path, class_name, dest_images_dir, dest_labels_dir):
    """Process an individual image and create YOLO annotation"""
    try:
        # Get image filename (without extension)
        image_filename = os.path.basename(image_path)
        image_basename = os.path.splitext(image_filename)[0]
        
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return
        
        height, width, _ = img.shape
        
        # Copy the image to the destination directory
        shutil.copy2(image_path, os.path.join(dest_images_dir, image_filename))
        
        # Create the YOLO annotation
        # For this script, we're treating each image as if the object fills the entire image
        # YOLO format: class_id center_x center_y width height (normalized 0-1)
        class_id = class_mapping[class_name]
        
        # Assume object fills the entire image for simplicity
        # You may need to adjust this based on your specific requirements
        center_x = 0.5  # center x coordinate (normalized)
        center_y = 0.5  # center y coordinate (normalized)
        bbox_width = 1.0  # width of bounding box (normalized)
        bbox_height = 1.0  # height of bounding box (normalized)
        
        # Create a label file for the image
        label_path = os.path.join(dest_labels_dir, f"{image_basename}.txt")
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}\n")
        
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def process_dataset():
    """Process the entire dataset structure"""
    for split in ['train', 'test', 'valid']:
        print(f"Processing {split} split...")
        
        # Define the paths
        source_split_path = os.path.join(source_base_dir, split)
        dest_split_path = os.path.join(dest_base_dir, split)
        
        # Create images and labels directories
        images_dir = os.path.join(dest_split_path, 'images')
        labels_dir = os.path.join(dest_split_path, 'labels')
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Process each class directory
        for class_name in class_mapping.keys():
            class_dir = os.path.join(source_split_path, class_name)
            
            # Check if the class directory exists
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} does not exist. Skipping.")
                continue
            
            print(f"  Processing class: {class_name}")
            
            # Process each image in the class directory
            processed_count = 0
            for image_filename in os.listdir(class_dir):
                if image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_dir, image_filename)
                    success = process_image(image_path, class_name, images_dir, labels_dir)
                    if success:
                        processed_count += 1
            
            print(f"  Processed {processed_count} images for class {class_name}")

if __name__ == "__main__":
    print("Starting YOLO annotation creation...")
    process_dataset()
    print("Finished creating YOLO annotations!")
    
    # Print summary of the created dataset
    for split in ['train', 'test', 'valid']:
        dest_split_path = os.path.join(dest_base_dir, split)
        images_dir = os.path.join(dest_split_path, 'images')
        labels_dir = os.path.join(dest_split_path, 'labels')
        
        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            image_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            label_count = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
            print(f"{split}: {image_count} images, {label_count} labels") 