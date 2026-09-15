import os
import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

IMAGES_PATH = os.path.join(DATA_DIR, "images.p")
LABELS_PATH = os.path.join(DATA_DIR, "labels.p")
MODEL_PATH = os.path.join(MODEL_DIR, "final_model.h5")
LABELS_JSON_PATH = os.path.join(MODEL_DIR, "labels.json")

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(IMAGES_PATH) or not os.path.exists(LABELS_PATH):
    raise SystemExit(
        "Dataset not found. Run consolidated_data.py first."
    )

with open(IMAGES_PATH, "rb") as f:
    images = pickle.load(f)

with open(LABELS_PATH, "rb") as f:
    labels = pickle.load(f)

images = np.array(images, dtype="float32") / 255.0
labels = np.array(labels)

# Add channel dimension: (samples, 100, 100, 1)
images = images.reshape(-1, 100, 100, 1)

# Create numeric labels
unique_labels = sorted(np.unique(labels))
label_to_index = {label: i for i, label in enumerate(unique_labels)}
numeric_labels = np.array([label_to_index[label] for label in labels])

# Convert labels to one-hot encoding
categorical_labels = to_categorical(
    numeric_labels,
    num_classes=len(unique_labels)
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    images,
    categorical_labels,
    test_size=0.2,
    random_state=42,
    stratify=numeric_labels
)

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(100, 100, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(len(unique_labels), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining model...\n")

model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=16,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Save trained model
model.save(MODEL_PATH)

# Save label mapping
with open(LABELS_JSON_PATH, "w") as f:
    json.dump(unique_labels.tolist(), f, indent=4)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Labels saved to: {LABELS_JSON_PATH}")