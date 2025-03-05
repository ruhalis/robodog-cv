import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
# Load your trained model
model = YOLO("runs\\detect\\train3\\weights\\best.pt")

# Run inference on a single image
results = model.predict(source="datasets\\general\\test\\images\\ds2_test_ds2_test_images_jpeg.rf.8fe4bbc7081e65f3a1d2c0c13f6830a3.jpg", conf=0.5, show=True)

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