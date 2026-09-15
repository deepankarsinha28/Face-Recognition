import os
import urllib.request
import cv2
import numpy as np
from keras.models import load_model

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
CASCADE_PATH = os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml")
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.h5")
CAMERA_URL = os.getenv("CAMERA_URL", "")
LABELS = ["akash", "chandru", "rakesh", "shiva", "siddhant"]

if not CAMERA_URL:
    raise SystemExit("Set CAMERA_URL to your IP camera snapshot URL before running.")
if not os.path.exists(MODEL_PATH):
    raise SystemExit("models/final_model.h5 is required for recognition but is not included in this repository.")

classifier = cv2.CascadeClassifier(CASCADE_PATH)
model = load_model(MODEL_PATH)

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1, 100, 100, 1).astype("float32") / 255.0
    return img

def get_pred_label(pred):
    return LABELS[pred] if 0 <= pred < len(LABELS) else "Unknown"

while True:
    try:
        raw = np.array(bytearray(urllib.request.urlopen(CAMERA_URL, timeout=5).read()), dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except Exception as exc:
        print(f"Camera error: {exc}")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5)

    for x, y, w, h in faces:
        face = frame[y:y + h, x:x + w]
        prediction = np.argmax(model.predict(preprocess(face), verbose=0))
        label = get_pred_label(prediction)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
