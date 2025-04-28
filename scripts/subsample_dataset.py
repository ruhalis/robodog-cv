#!/usr/bin/env python3
"""
Subsample YOLO-formatted dataset to a maximum number of images per specified classes.
Deletes image and corresponding label files beyond the limit.
"""
import argparse
import random
from pathlib import Path
import os

def subsample(dataset_dir, class_ids, max_count, subsets):
    dataset_dir = Path(dataset_dir)
    for subset in subsets:
        images_dir = dataset_dir / subset / "images"
        labels_dir = dataset_dir / subset / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Skipping subset '{subset}': images or labels directory does not exist.")
            continue

        # Map each class ID to its list of label files
        cls_map = {cls: [] for cls in class_ids}
        for label_path in labels_dir.glob("*.txt"):
            with label_path.open('r') as f:
                lines = [l.strip() for l in f if l.strip()]
            present = set(int(l.split()[0]) for l in lines)
            for cls in class_ids:
                if cls in present:
                    cls_map[cls].append(label_path)

        # For each class, delete files exceeding the max_count
        for cls, files in cls_map.items():
            total = len(files)
            if total <= max_count:
                print(f"Class {cls} in subset '{subset}' has {total} images (<= {max_count}), skipping.")
                continue
            random.shuffle(files)
            to_delete = files[max_count:]
            print(f"Deleting {len(to_delete)} images for class {cls} in subset '{subset}'.")
            for label_path in to_delete:
                try:
                    label_path.unlink()
                except Exception as e:
                    print(f"Failed to delete label {label_path}: {e}")
                # delete corresponding image (try common extensions)
                stem = label_path.stem
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                    img_path = images_dir / f"{stem}{ext}"
                    if img_path.exists():
                        try:
                            img_path.unlink()
                        except Exception as e:
                            print(f"Failed to delete image {img_path}: {e}")
                        break


def main():
    parser = argparse.ArgumentParser(
        description="Subsample YOLO dataset to limit images per class"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to dataset root (with train/val/test folders)"
    )
    parser.add_argument(
        "--classes", required=True, nargs='+', type=int,
        help="Class IDs to subsample (e.g. 0 4 5)"
    )
    parser.add_argument(
        "--max", required=True, type=int,
        help="Maximum images to keep per class"
    )
    parser.add_argument(
        "--subsets", nargs='+', default=["train"],
        help="Dataset subsets to process (default: train)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    random.seed(args.seed)
    subsample(args.dataset, args.classes, args.max, args.subsets)

if __name__ == "__main__":
    main() 