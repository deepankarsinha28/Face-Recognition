import os
import pickle
import cv2
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(DATA_DIR, exist_ok=True)

image_data = []
labels = []

for filename in sorted(os.listdir(IMAGE_DIR)):
    path = os.path.join(IMAGE_DIR, filename)
    image = cv2.imread(path)
    if image is None:
        continue
    image = cv2.resize(image, (100, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_data.append(image)
    labels.append(filename.rsplit("_", 1)[0])

if not image_data:
    raise SystemExit("No face images found in the images/ directory.")

image_data = np.array(image_data)
labels = np.array(labels)

with open(os.path.join(DATA_DIR, "images.p"), "wb") as f:
    pickle.dump(image_data, f)
with open(os.path.join(DATA_DIR, "labels.p"), "wb") as f:
    pickle.dump(labels, f)

print(f"Saved {len(image_data)} images and labels to data/.")
