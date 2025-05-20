from ultralytics import YOLO
import cv2
import argparse
import os

def detect_image(image_path, model_path, conf_threshold=0.25, save_output=True):
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Run inference on the image
    results = model.predict(source=image, conf=conf_threshold)
    
    # Get the annotated image
    annotated_image = results[0].plot()
    
    # Display the results
    cv2.imshow("YOLOv11 Detection", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the output if requested
    if save_output:
        output_dir = "detection_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the original filename and create output path
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_detected{ext}")
        
        cv2.imwrite(output_path, annotated_image)
        print(f"Saved detection results to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Perform YOLO detection on an image")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Path to the YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-save", action="store_true", help="Don't save the output image")
    
    args = parser.parse_args()
    
    detect_image(
        image_path=args.image,
        model_path=args.model,
        conf_threshold=args.conf,
        save_output=not args.no_save
    )

if __name__ == "__main__":
    main() 