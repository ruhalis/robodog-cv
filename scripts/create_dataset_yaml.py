#!/usr/bin/env python3
import os
import yaml

def create_dataset_yaml():
    """Create a YAML file for the dataset configuration"""
    # Define paths
    dest_base_dir = 'train-curat-dataset-yolo'
    yaml_path = os.path.join(dest_base_dir, 'dataset.yaml')
    
    # Define class names
    class_names = ['algae', 'peeling', 'stain']
    
    # Get absolute paths
    current_dir = os.getcwd()
    train_path = os.path.join(current_dir, dest_base_dir, 'train')
    val_path = os.path.join(current_dir, dest_base_dir, 'valid')
    test_path = os.path.join(current_dir, dest_base_dir, 'test')
    
    # Create YAML content
    yaml_content = {
        'path': os.path.join(current_dir, dest_base_dir),
        'train': os.path.join('train', 'images'),
        'val': os.path.join('valid', 'images'),
        'test': os.path.join('test', 'images'),
        'nc': len(class_names),
        'names': class_names
    }
    
    # Write to YAML file
    with open(yaml_path, 'w') as file:
        yaml.dump(yaml_content, file, default_flow_style=False)
    
    print(f"Created dataset YAML file: {yaml_path}")

if __name__ == "__main__":
    create_dataset_yaml() 