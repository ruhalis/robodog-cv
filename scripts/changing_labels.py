import os
import shutil
import argparse

# ----- Mapping Dictionaries -----
# Map original class index (from the label file) to the original class name.
original_classes = {
    0: "light_off",
    1: "light_on",
    2: "Light-Closed",
    3: "Light-Opened",
    4: "lamp1_off",
    5: "lamp1_on",
    6: "lamp2_off",
    7: "lamp2_on"
}

# Map each original class name to a unified class name.
mapping = {
    "light_off": "light_off",
    "Light-Closed": "light_off",
    "lamp1_off": "light_off",
    "lamp2_off": "light_off",
    "light_on": "light_on",
    "Light-Opened": "light_on",
    "lamp1_on": "light_on",
    "lamp2_on": "light_on"
}

# Unified class name to new index.
new_class_dict = {
    "light_off": 0,
    "light_on": 1
}

# ----- Function to Convert a Label File -----
def convert_label_file(input_path, output_path):
    """
    Reads a YOLO label file, converts each annotation from the original class index
    to the unified class index using the mapping dictionaries, and writes the updated
    annotations to the output file.
    """
    with open(input_path, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            print(f"Skipping malformed line in {input_path}: {line}")
            continue
        
        try:
            orig_index = int(parts[0])
        except ValueError:
            print(f"Skipping non-integer class index in {input_path}: {line}")
            continue

        orig_class = original_classes.get(orig_index)
        if orig_class is None:
            print(f"Unknown class index {orig_index} in {input_path}")
            continue

        unified_class = mapping.get(orig_class)
        if unified_class is None:
            print(f"No mapping defined for class {orig_class} in {input_path}")
            continue

        new_index = new_class_dict.get(unified_class)
        if new_index is None:
            print(f"No new index defined for unified class {unified_class} in {input_path}")
            continue

        # Reconstruct the annotation line with the new class index
        new_line = " ".join([str(new_index)] + parts[1:])
        new_lines.append(new_line)
    
    with open(output_path, "w") as f:
        for nl in new_lines:
            f.write(nl + "\n")

# ----- Function to Process One Split (train/val/test) -----
def process_split(split_dir, output_split_dir):
    """
    Processes a split directory (e.g., train, val, or test).
    It expects that split_dir contains "images" and "labels" subfolders.
    The function converts the labels using the mapping and copies the images
    to the output_split_dir while preserving filenames.
    """
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")
    out_images_dir = os.path.join(output_split_dir, "images")
    out_labels_dir = os.path.join(output_split_dir, "labels")
    
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    
    # Process each label file
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue
        
        base_name = os.path.splitext(label_file)[0]
        input_label_path = os.path.join(labels_dir, label_file)
        output_label_path = os.path.join(out_labels_dir, label_file)
        
        convert_label_file(input_label_path, output_label_path)
        print(f"Processed label file: {output_label_path}")
        
        # Find and copy the corresponding image
        image_found = False
        for ext in [".jpg", ".jpeg", ".png"]:
            image_file = base_name + ext
            input_image_path = os.path.join(images_dir, image_file)
            if os.path.exists(input_image_path):
                output_image_path = os.path.join(out_images_dir, image_file)
                shutil.copy2(input_image_path, output_image_path)
                print(f"Copied image: {output_image_path}")
                image_found = True
                break
        if not image_found:
            print(f"Warning: No image found for label file {label_file}")

# ----- Main Function -----
def main():
    parser = argparse.ArgumentParser(
        description="Change YOLOv11 dataset class names in a single dataset with train, val, and test splits."
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the input dataset directory (with train, val, test folders).")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output dataset directory with updated labels.")
    args = parser.parse_args()

    splits = ["train", "valid", "test"]
    for split in splits:
        split_input_dir = os.path.join(args.dataset_dir, split)
        split_output_dir = os.path.join(args.output, split)
        if not os.path.exists(split_input_dir):
            print(f"Split {split} not found in {args.dataset_dir}. Skipping.")
            continue
        process_split(split_input_dir, split_output_dir)
    
    # Create a unified classes file in the output dataset directory.
    classes_file = os.path.join(args.output, "classes.txt")
    with open(classes_file, "w") as f:
        sorted_classes = sorted(new_class_dict.items(), key=lambda x: x[1])
        for class_name, _ in sorted_classes:
            f.write(class_name + "\n")
    print("Class names conversion complete. Output dataset saved at:", args.output)

if __name__ == "__main__":
    main()
