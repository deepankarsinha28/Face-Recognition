import os
import urllib.request
import cv2
import numpy as np

CASCADE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "haarcascade_frontalface_default.xml")
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
CAMERA_URL = os.getenv("CAMERA_URL", "")
SAMPLES = 100

classifier = cv2.CascadeClassifier(CASCADE_PATH)
os.makedirs(IMAGE_DIR, exist_ok=True)

if not CAMERA_URL:
    raise SystemExit("Set CAMERA_URL to your IP camera snapshot URL before running.")

data = []
while len(data) < SAMPLES:
    image_from_url = urllib.request.urlopen(CAMERA_URL, timeout=5)
    raw = np.array(bytearray(image_from_url.read()), dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_points = classifier.detectMultiScale(gray, 1.3, 5)

    if len(face_points) > 0:
        x, y, w, h = face_points[0]
        face_frame = frame[y:y + h, x:x + w]
        data.append(face_frame)
        cv2.imshow("Only face", face_frame)
        print(f"{len(data)}/{SAMPLES}")

    cv2.putText(frame, str(len(data)), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(30) == ord("q"):
        break

cv2.destroyAllWindows()

if len(data) == SAMPLES:
    name = input("Enter person's name: ").strip()
    for i, face in enumerate(data):
        cv2.imwrite(os.path.join(IMAGE_DIR, f"{name}_{i}.jpg"), face)
    print("Data collection complete.")
else:
    print("Data collection stopped before reaching the target.")
