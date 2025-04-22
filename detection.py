from ultralytics import YOLO
import cv2

# Load your YOLOv11 model (update with your actual weights path)
model = YOLO('/home/ruhalis/coin/robodog-cv/runs/detect/train3/weights/best.pt')

# Open the default camera (0); change the index if needed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break

    # Run inference on the current frame; 'source' can be the frame itself
    results = model.predict(source=frame, conf=0.25)
    
    # Get the annotated frame (bounding boxes and labels are drawn on it)
    annotated_frame = results[0].plot()  # returns image in BGR format

    # Display the annotated frame
    cv2.imshow("Real-Time YOLOv11", annotated_frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close display windows
cap.release()
cv2.destroyAllWindows()
