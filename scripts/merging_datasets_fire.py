import os
import shutil
import argparse

# ----- Mapping dictionaries -----
# Maps the original class index (from the label file) to the original class name.
original_classes = {
    0: "cabinet",
    1: "Fire Extinguisher",
    2: "Fire-Extinguisher",
    3: "FireExtinguisher",
    4: "hose",
    5: "floor-kept",
    6: "wall-mounted"
}

# Mapping from original class name to unified class name.
mapping = {
    "cabinet": "cabinet",
    "Fire Extinguisher": "fire_extinguisher",
    "Fire-Extinguisher": "fire_extinguisher",
    "FireExtinguisher": "fire_extinguisher",
    "hose": "hose",
    "floor-kept": "fire_extinguisher",
    "wall-mounted": "fire_extinguisher"
}

# New unified class names to new index.
new_class_dict = {
    "fire_extinguisher": 0,
    "cabinet": 1,
    "hose": 2
}

# ----- Functions to process labels -----
def convert_label_file(input_path, output_path):
    """
    Reads a YOLO label file from input_path, converts the original class indices
    to unified indices using the mapping dictionaries, and writes the new content
    to output_path.
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

        unified_name = mapping.get(orig_class)
        if unified_name is None:
            print(f"No mapping defined for class {orig_class} in {input_path}")
            continue

        new_index = new_class_dict.get(unified_name)
        if new_index is None:
            print(f"No new index defined for unified class {unified_name} in {input_path}")
            continue

        new_line = " ".join([str(new_index)] + parts[1:])
        new_lines.append(new_line)
    
    # Write the converted annotations to the output file.
    with open(output_path, "w") as f:
        for nl in new_lines:
            f.write(nl + "\n")

# ----- Function to process one split (train/val/test) of one dataset -----
def process_split(dataset_path, split, output_dir, prefix):
    """
    Processes one split (e.g., train, val, or test) of a dataset.
    It expects that dataset_path/<split> contains "images" and "labels" folders.
    Converted label files are written to output_dir/<split>/labels,
    and corresponding image files are copied to output_dir/<split>/images.
    All filenames are prefixed with the provided prefix.
    """
    split_dir = os.path.join(dataset_path, split)
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")

    # Output directories for this split.
    out_images_dir = os.path.join(output_dir, split, "images")
    out_labels_dir = os.path.join(output_dir, split, "labels")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue

        base_name = os.path.splitext(label_file)[0]
        new_label_filename = f"{prefix}_{label_file}"
        input_label_path = os.path.join(labels_dir, label_file)
        output_label_path = os.path.join(out_labels_dir, new_label_filename)

        # Convert the label file.
        convert_label_file(input_label_path, output_label_path)
        print(f"Processed label file: {output_label_path}")

        # Copy the corresponding image file.
        image_found = False
        for ext in [".jpg", ".jpeg", ".png"]:
            image_filename = base_name + ext
            input_image_path = os.path.join(images_dir, image_filename)
            if os.path.exists(input_image_path):
                new_image_filename = f"{prefix}_{image_filename}"
                output_image_path = os.path.join(out_images_dir, new_image_filename)
                shutil.copy2(input_image_path, output_image_path)
                print(f"Copied image file: {output_image_path}")
                image_found = True
                break
        if not image_found:
            print(f"Warning: No image found for label file {label_file}")

# ----- Main function to process both datasets -----
def main():
    parser = argparse.ArgumentParser(
        description="Merge two YOLOv11 datasets (with train, val, and test splits) into one and remap classes."
    )
    parser.add_argument("--dataset1", type=str, required=True,
                        help="Path to the first dataset (with train, val, test folders).")
    parser.add_argument("--dataset2", type=str, required=True,
                        help="Path to the second dataset (with train, val, test folders).")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output merged dataset directory.")
    args = parser.parse_args()

    splits = ["train", "valid", "test"]
    for split in splits:
        # Process dataset 1 for this split.
        prefix1 = f"ds1_{split}"
        process_split(args.dataset1, split, args.output, prefix1)
        # Process dataset 2 for this split.
        prefix2 = f"ds2_{split}"
        process_split(args.dataset2, split, args.output, prefix2)

    # Create a unified classes file in the output directory.
    classes_file = os.path.join(args.output, "classes.txt")
    with open(classes_file, "w") as f:
        # Order the unified classes by new index.
        sorted_classes = sorted(new_class_dict.items(), key=lambda x: x[1])
        for class_name, _ in sorted_classes:
            f.write(class_name + "\n")
    print(f"Unified classes file written to: {classes_file}")
    print("Merge complete.")

if __name__ == "__main__":
    main()
