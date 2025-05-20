import cv2
import numpy as np
import argparse
import sys

# ----------------------------------------------------------------------
# configurable defaults
CENTER_FRACTION = 0.33        # size of the centre ROI
WHITE_SAT_MAX = 50            # allow only desaturated (near-gray/white) pixels
WHITE_VAL_MIN = 180           # require brightness
COEFF_THRESH = 0.05           # if coef > this → detected
RESIZE_WIDTH = 1024           # target width for resizing
RESIZE_HEIGHT = 1024          # target height for resizing
# ----------------------------------------------------------------------


def white_mask(image, sat_max=WHITE_SAT_MAX, val_min=WHITE_VAL_MIN):
    """binary mask of 'white enough' pixels in image"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # allow any hue, sat <= sat_max, val >= val_min
    lower = np.array([0,      0,       val_min])
    upper = np.array([180, sat_max,     255])
    mask  = cv2.inRange(hsv, lower, upper)

    # clean up
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def calculate_white_coefficient(image_path, center_fraction=CENTER_FRACTION,
                               sat_max=WHITE_SAT_MAX, val_min=WHITE_VAL_MIN, 
                               display=False, use_roi=False, resize=False):
    """
    Calculate the white coefficient of an image
    
    Args:
        image_path: Path to the image
        center_fraction: If use_roi is True, fraction of image to use as ROI
        sat_max: Maximum saturation for "white" pixels
        val_min: Minimum value/brightness for "white" pixels
        display: Show visualization of detection
        use_roi: If True, uses a rectangular ROI. If False, uses entire image.
        resize: If True, resizes the image to RESIZE_WIDTH x RESIZE_HEIGHT
        
    Returns:
        float: White coefficient (0-1)
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"could not read image: {image_path}")
    
    original_size = frame.shape[:2]  # Store original size
    print(f"Original image size: {original_size[1]}x{original_size[0]} (WxH)")
    
    # Resize image if requested
    if resize:
        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)
        print(f"Resized image to {RESIZE_WIDTH}x{RESIZE_HEIGHT}")

    h, w = frame.shape[:2]
    
    if use_roi:
        # Calculate ROI dimensions (2x wider horizontally, 1.5x taller)
        roi_w = int(min(h, w) * center_fraction / 2)
        roi_h = int(min(h, w) * center_fraction * 1.5)
        roi_h = min(roi_h, h)
        
        # Extract ROI
        x0 = (w - roi_w) // 2
        y0 = (h - roi_h) // 2
        roi = frame[y0:y0 + roi_h, x0:x0 + roi_w]
        
        # Get mask for ROI
        mask = white_mask(roi, sat_max, val_min)
        white_coef = mask.mean() / 255.0
        
        # For display
        if display:
            cv2.rectangle(frame, (x0, y0), (x0 + roi_w, y0 + roi_h),
                        (0, 255, 0) if white_coef > COEFF_THRESH else (0, 0, 255), 2)
    else:
        # Process the entire image
        mask = white_mask(frame, sat_max, val_min)
        white_coef = mask.mean() / 255.0

    if display:
        # Add text with coefficient value
        cv2.putText(frame, f"white_coef={white_coef:.4f}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Create named windows with specific size control
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("White Mask", cv2.WINDOW_NORMAL)
        
        # Display the images
        cv2.imshow("Image", frame)
        cv2.imshow("White Mask", mask)
        
        # Resize windows to fit comfortably on screen
        max_display_dim = 800  # Maximum dimension for display
        
        # Calculate scaling factor
        scale = min(1.0, max_display_dim / max(h, w))
        display_w = int(w * scale)
        display_h = int(h * scale)
        
        # Resize windows
        cv2.resizeWindow("Image", display_w, display_h)
        cv2.resizeWindow("White Mask", display_w, display_h)
        
        print(f"Displaying image at {display_w}x{display_h}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return white_coef


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect white/bright areas in a photo and calculate coefficient."
    )
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--use-roi", action="store_true",
                        help="Use a region of interest instead of entire image")
    parser.add_argument("--resize", action="store_true",
                        help=f"Resize image to {RESIZE_WIDTH}x{RESIZE_HEIGHT} before processing")
    parser.add_argument("--center-fraction", type=float,
                        default=CENTER_FRACTION,
                        help=f"Fraction of the image for ROI (default: {CENTER_FRACTION})")
    parser.add_argument("--sat-max", type=int, default=WHITE_SAT_MAX,
                        help=f"Maximum saturation for white (default: {WHITE_SAT_MAX})")
    parser.add_argument("--val-min", type=int, default=WHITE_VAL_MIN,
                        help=f"Minimum brightness value (default: {WHITE_VAL_MIN})")
    parser.add_argument("--display", action="store_true",
                        help="Show visualization windows")
    args = parser.parse_args(argv)

    coef = calculate_white_coefficient(
        args.image_path,
        center_fraction=args.center_fraction,
        sat_max=args.sat_max,
        val_min=args.val_min,
        display=args.display,
        use_roi=args.use_roi,
        resize=args.resize
    )

    detected = coef > COEFF_THRESH
    print(f"white_coefficient={coef:.6f}  detected={detected}")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 