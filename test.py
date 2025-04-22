import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
# Load your trained model
# model = YOLO("/home/ruhalis/coin/robodog-cv/runs/detect/train3/weights/best.pt")
model = YOLO("runs\\detect\\train14\\weights\\best.pt")

# Run inference on a single image
results = model.predict(source="photos\\5350635554477305320.jpg", conf=0.5, show=True)
# results = model.predict(source="datasets\\light5_split\\test\\images\\photo_19_2025-04-21_12-04-59_jpg.rf.4442fafdb82103c8fa7a7e972aef99c8.jpg", conf=0.5, show=True)


# If you want to save predictions:
# results = model.predict(source="path/to/test_image.jpg", conf=0.25, save=True)

# If you have multiple test images, you can specify a folder:
# results = model.predict(source="path/to/test_images_folder", conf=0.25, save=True)

img = results[0].plot()

# Convert BGR to RGB for correct color display in matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot the image
plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
plt.axis("off")
plt.show()