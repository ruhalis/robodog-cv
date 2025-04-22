import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- PARAMETERS ---
CAM_IDX = 0                    # which camera to open
CONF_THRESH = 0.5              # min detection confidence
STATIONARY_THRESH = 5000       # max # of changed pixels to be 'stationary'
DIFF_THRESH = 25               # per‑pixel diff threshold

# --- INITIALIZE ---
cap = cv2.VideoCapture(CAM_IDX)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {CAM_IDX}")

model = YOLO('best_12.04.pt')     
tracker = DeepSort(
    max_age=30,       # frames to keep 'dead' tracks
    n_init=3,         # frames until track is confirmed
    max_cosine_distance=0.2
)

unique_ids = set()
last_gray = None

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break



    # 2) Run YOLO → get raw xyxy boxes + scores + classes
    results = model(frame)[0]
    xyxy   = results.boxes.xyxy.cpu().numpy()    # (N,4): x1,y1,x2,y2
    scores = results.boxes.conf.cpu().numpy()    # (N,)
    classes= results.boxes.cls.cpu().numpy()     # (N,)

    # 3) Convert to DeepSORT’s ([x,y,w,h], score, cls) tuples
    raw_dets = []
    for (x1,y1,x2,y2), conf, cls in zip(xyxy, scores, classes):
        if conf < CONF_THRESH:
            continue
        w = x2 - x1
        h = y2 - y1
        bbox_xywh = [float(x1), float(y1), float(w), float(h)]
        raw_dets.append((bbox_xywh, float(conf), int(cls)))

    # 4) Update DeepSORT
    tracks = tracker.update_tracks(raw_dets, frame=frame)

    # 5) Draw + count
    for t in tracks:
        if not t.is_confirmed():
            continue
        tid = t.track_id
        unique_ids.add(tid)
        x1,y1,w,h = t.to_ltrb()  # left, top, right, bottom
        x1, y1, x2, y2 = int(x1), int(y1), int(w), int(h)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"ID:{tid}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(frame, f"Unique objects: {len(unique_ids)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
    cv2.imshow("Inspection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
