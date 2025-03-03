import os
import argparse
import shutil

def convert_line_to_bbox(line):
    """
    Converts a single annotation line.
      - If the line has 5 numbers, it's assumed to be a standard YOLO bbox and is returned unchanged.
      - If the line has more than 5 numbers, it's assumed to represent a polygon (class, x1, y1, x2, y2, ...).
        The function computes the axis-aligned bounding box that encloses the polygon.
    All coordinates are assumed to be normalized.
    """
    parts = line.strip().split()
    # Already a standard YOLO bbox
    if len(parts) == 5:
        return line.strip()
    
    # Otherwise, assume polygon format: class x1 y1 x2 y2 x3 y3 ...
    class_id = parts[0]
    coords = parts[1:]
    
    # Check for valid polygon (at least 2 points and even number of coordinates)
    if len(coords) < 4 or (len(coords) % 2) != 0:
        print(f"Warning: invalid polygon annotation (insufficient coordinate pairs): {line}")
        return None

    # Convert coordinates to float values
    try:
        xs = list(map(float, coords[0::2]))
        ys = list(map(float, coords[1::2]))
    except ValueError:
        print(f"Warning: unable to convert coordinates to float in line: {line}")
        return None

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    x_center = x_min + bbox_width / 2.0
    y_center = y_min + bbox_height / 2.0

    return f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"

def convert_label_file(input_path, output_path):
    """
    Reads an annotation file from input_path, converts each line as needed, and writes the new content to output_path.
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()

    converted_lines = []
    for line in lines:
        if not line.strip():
            continue
        converted = convert_line_to_bbox(line)
        if converted is not None:
            converted_lines.append(converted)
    
    if converted_lines:
        with open(output_path, 'w') as f:
            for cline in converted_lines:
                f.write(cline + '\n')

def process_dataset(input_dataset, output_dataset):
    """
    Recursively processes all files in input_dataset.
    - If the file is inside a folder named "labels" and ends with ".txt", it is processed (polygon conversion) and written to the new dataset.
    - All other files are copied as is.
    The folder structure (train, val, test, etc.) is maintained.
    """
    for root, dirs, files in os.walk(input_dataset):
        # Compute the relative path and corresponding output directory
        rel_path = os.path.relpath(root, input_dataset)
        output_dir = os.path.join(output_dataset, rel_path)
        os.makedirs(output_dir, exist_ok=True)
        
        for file in files:
            input_file = os.path.join(root, file)
            output_file = os.path.join(output_dir, file)
            
            # If we're in a "labels" folder and the file is a .txt file, convert it.
            if os.path.basename(root) == "labels" and file.endswith(".txt"):
                convert_label_file(input_file, output_file)
                print(f"Converted label file: {output_file}")
            else:
                # Otherwise, copy the file as is.
                shutil.copy2(input_file, output_file)
                print(f"Copied file: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Create a new dataset folder with the same train/val/test structure, converting polygon labels to YOLO bounding boxes."
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the original dataset directory.")
    parser.add_argument("--output_dataset_dir", type=str, required=True,
                        help="Path for the new dataset directory to be created.")
    args = parser.parse_args()

    process_dataset(args.dataset_dir, args.output_dataset_dir)
    print("New dataset created with converted labels at:", args.output_dataset_dir)

if __name__ == "__main__":
    main()

