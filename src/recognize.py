import os
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")

CASCADE_PATH = os.path.join(
    BASE_DIR, "models", "haarcascade_frontalface_default.xml"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "final_model.h5"
)

LABELS_PATH = os.path.join(
    BASE_DIR, "models", "labels.json"
)

CAMERA_URL = os.getenv("CAMERA_URL", "0")


# Load face detector
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    raise SystemExit("Error: Haar Cascade file could not be loaded.")


# Check model
if not os.path.exists(MODEL_PATH):
    raise SystemExit(
        "Model not found. Run model_training.py first."
    )


# Check labels
if not os.path.exists(LABELS_PATH):
    raise SystemExit(
        "labels.json not found. Run model_training.py first."
    )


# Load trained model
model = load_model(MODEL_PATH)

# Load class labels
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)


# Open camera
if CAMERA_URL == "0":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    raise SystemExit("Error: Could not open camera.")


print("Face recognition started.")
print("Press 'q' to quit.")


while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = gray[y:y + h, x:x + w]

        face = cv2.resize(face, (100, 100))

        face = face.astype("float32") / 255.0

        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)

        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        name = labels[class_index]

        text = f"{name} ({confidence * 100:.1f}%)"

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()