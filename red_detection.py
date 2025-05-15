import cv2
import numpy as np
import time

CAM_ID           = 0          # USB-camera index
CENTER_FRACTION  = 0.33       # side length of the square ROI as a fraction of frame size
MIN_RED_RATIO    = 0.10       # threshold for detecting a “press”
COOLDOWN_SECONDS = 1.0        # debounce time after a detection

def main():
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        raise RuntimeError("Camera not found")

    pressed        = False
    last_event_t   = 0.0

    # HSV thresholds for red (split low/high hue ranges)
    lo1 = np.array([0, 120,  70]);   hi1 = np.array([10,255,255])
    lo2 = np.array([170,120,70]);    hi2 = np.array([180,255,255])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        side = int(min(h, w) * CENTER_FRACTION)
        x0, y0 = (w - side)//2, (h - side)//2
        roi = frame[y0:y0+side, x0:x0+side]

        # detect red in ROI
        hsv      = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask1    = cv2.inRange(hsv, lo1, hi1)
        mask2    = cv2.inRange(hsv, lo2, hi2)
        red_mask = mask1 | mask2
        red_ratio = red_mask.mean() / 255.0

        now = time.time()
        if red_ratio >= MIN_RED_RATIO and not pressed and now - last_event_t > COOLDOWN_SECONDS:
            pressed = True
            last_event_t = now
            print(f"[{time.strftime('%H:%M:%S')}] PRESSED (red={red_ratio:.2f})")
        elif red_ratio < MIN_RED_RATIO * 0.5:
            pressed = False

        # draw the boundary (green if pressed, red otherwise)
        color = (0,255,0) if pressed else (0,0,255)
        cv2.rectangle(frame, (x0,y0), (x0+side, y0+side), color, 2)

        # overlay the red coefficient
        text = f"Red coef: {red_ratio:.2f}"
        cv2.putText(frame, text, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        # show
        cv2.imshow("Live Feed", frame)
        cv2.imshow("Red Mask", red_mask)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
