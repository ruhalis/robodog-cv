import os
import shutil
import argparse

def process_split(dataset_path, split, output_dir, prefix):
    """
    Processes one split (e.g., train, val, or test) from a dataset.
    Copies image and label files to the output dataset directory.
    Files are prefixed with the provided prefix to avoid name collisions.
    """
    input_images_dir = os.path.join(dataset_path, split, "images")
    input_labels_dir = os.path.join(dataset_path, split, "labels")
    
    # Output directories for this split.
    output_images_dir = os.path.join(output_dir, split, "images")
    output_labels_dir = os.path.join(output_dir, split, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Copy images
    if os.path.exists(input_images_dir):
        for filename in os.listdir(input_images_dir):
            # Skip non-image files if needed (you can adjust the extensions if necessary)
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            new_filename = f"{prefix}_{filename}"
            src_path = os.path.join(input_images_dir, filename)
            dst_path = os.path.join(output_images_dir, new_filename)
            shutil.copy2(src_path, dst_path)
            print(f"Copied image: {dst_path}")
    else:
        print(f"Warning: {input_images_dir} does not exist.")
    
    # Copy labels
    if os.path.exists(input_labels_dir):
        for filename in os.listdir(input_labels_dir):
            if not filename.lower().endswith(".txt"):
                continue
            new_filename = f"{prefix}_{filename}"
            src_path = os.path.join(input_labels_dir, filename)
            dst_path = os.path.join(output_labels_dir, new_filename)
            shutil.copy2(src_path, dst_path)
            print(f"Copied label: {dst_path}")
    else:
        print(f"Warning: {input_labels_dir} does not exist.")

def main():
    parser = argparse.ArgumentParser(
        description="Merge three YOLOv11 datasets (with train, val, test splits) into one dataset without modifying labels."
    )
    parser.add_argument("--dataset1", type=str, required=True,
                        help="Path to the first dataset directory.")
    parser.add_argument("--dataset2", type=str, required=True,
                        help="Path to the second dataset directory.")
    parser.add_argument("--dataset3", type=str, required=True,
                        help="Path to the third dataset directory.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output merged dataset directory.")
    args = parser.parse_args()
    
    # Define the splits to process.
    splits = ["train", "valid", "test"]
    
    # Process each dataset for each split.
    datasets = [
        (args.dataset1, "ds1"),
        (args.dataset2, "ds2"),
        (args.dataset3, "ds3")
    ]
    
    for dataset_path, prefix in datasets:
        for split in splits:
            split_input_dir = os.path.join(dataset_path, split)
            if os.path.exists(split_input_dir):
                process_split(dataset_path, split, args.output, prefix)
            else:
                print(f"Warning: Split '{split}' not found in {dataset_path}")
    
    print("Merge complete. Output dataset available at:", args.output)

if __name__ == "__main__":
    main()
