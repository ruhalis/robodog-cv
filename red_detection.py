import cv2
import numpy as np
import time

CAM_ID           = 0          # USB-camera index
CENTER_FRACTION  = 0.33       # side length of the square ROI as a fraction of frame size
MIN_RED_RATIO    = 0.10       # threshold for detecting a "press"
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
        
        # Calculate ROI dimensions - make it 2x narrower and 1.5x taller
        roi_width = int(min(h, w) * CENTER_FRACTION / 2)  # Half as wide
        roi_height = int(min(h, w) * CENTER_FRACTION * 1.5)  # 1.5x taller
        
        # Ensure we don't exceed image boundaries
        roi_height = min(roi_height, h)
        
        # Calculate the top-left corner of the ROI
        x0 = (w - roi_width) // 2
        y0 = (h - roi_height) // 2
        
        # Extract the ROI
        roi = frame[y0:y0+roi_height, x0:x0+roi_width]

        # detect red in ROI
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 1) two halves of red hue
        m1 = cv2.inRange(hsv, (  0, 120,   0), ( 10, 255, 240))
        m2 = cv2.inRange(hsv, (170, 120,   0), (180, 255, 240))
        raw_red = cv2.bitwise_or(m1, m2)

        # 2) morphological opening+closing to kill noise & fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        red_mask = cv2.morphologyEx(raw_red, cv2.MORPH_OPEN,  kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # 3) (optional) enforce true red channel dominance
        b,g,r = cv2.split(roi)
        ratio = r.astype(np.float32) / (r.astype(np.float32) + g + b + 1)
        red_mask &= (ratio > 0.5).astype(np.uint8) * 255

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
        cv2.rectangle(frame, (x0,y0), (x0+roi_width, y0+roi_height), color, 2)

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
