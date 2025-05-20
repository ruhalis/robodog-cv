import cv2
import numpy as np
import argparse
import sys

# ----------------------------------------------------------------------
# configurable defaults
CENTER_FRACTION = 0.33        # size of the centre ROI, as before
SAT_MIN   = 30                #   <-- was 120
VAL_MIN   = 50                #   <-- new lower-brightness floor
HUE_LOW_1,  HUE_HIGH_1  = 0, 10
HUE_LOW_2,  HUE_HIGH_2  = 170, 180
RED_DOM_RATIO = 0.40          # R / (R+G+B) must exceed this
COEFF_THRESH  = 0.05          # if coef > this → “pressed”
# ----------------------------------------------------------------------


def red_mask(roi, s_min, v_min):
    """binary mask of ‘red enough’ pixels in roi"""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    m1 = cv2.inRange(hsv, (HUE_LOW_1, s_min, v_min),
                           (HUE_HIGH_1, 255, 255))
    m2 = cv2.inRange(hsv, (HUE_LOW_2, s_min, v_min),
                           (HUE_HIGH_2, 255, 255))
    mask = m1 | m2

    # morphology to remove specks / fill holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # enforce “red channel dominates”
    b, g, r = cv2.split(roi)
    ratio   = r.astype(np.float32) / (r + g + b + 1)
    mask &= (ratio > RED_DOM_RATIO).astype(np.uint8) * 255
    return mask


def calculate_red_coefficient(image_path, center_fraction=CENTER_FRACTION,
                              s_min=SAT_MIN, v_min=VAL_MIN, display=False):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"could not read image: {image_path}")

    h, w = frame.shape[:2]

    # same asymmetric ROI you used before
    roi_w = int(min(h, w) * center_fraction / 2)
    roi_h = int(min(h, w) * center_fraction * 1.5)
    roi_h = min(roi_h, h)

    x0 = (w - roi_w) // 2
    y0 = (h - roi_h) // 2
    roi = frame[y0:y0 + roi_h, x0:x0 + roi_w]

    mask = red_mask(roi, s_min, v_min)
    red_coeff = mask.mean() / 255.0

    if display:
        cv2.rectangle(frame, (x0, y0), (x0 + roi_w, y0 + roi_h),
                      (0, 255, 0) if red_coeff > COEFF_THRESH else (0, 0, 255), 2)
        cv2.putText(frame, f"coef={red_coeff:.3f}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Image", frame)
        cv2.imshow("Red Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return red_coeff


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect pressed elevator button (red ring) in a still photo."
    )
    parser.add_argument("image_path")
    parser.add_argument("--center-fraction", type=float,
                        default=CENTER_FRACTION,
                        help="fraction of the image height to examine (default 0.33)")
    parser.add_argument("--sat-min",   type=int, default=SAT_MIN,
                        help="minimum HSV saturation to qualify as red (default 30)")
    parser.add_argument("--val-min",   type=int, default=VAL_MIN,
                        help="minimum HSV value/brightness (default 50)")
    parser.add_argument("--display", action="store_true",
                        help="show debug windows")
    args = parser.parse_args(argv)

    coef = calculate_red_coefficient(
        args.image_path,
        center_fraction=args.center_fraction,
        s_min=args.sat_min,
        v_min=args.val_min,
        display=args.display
    )

    pressed = coef > COEFF_THRESH
    print(f"red_coefficient={coef:.6f}  pressed={pressed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
