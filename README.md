# CTTC Face Recognition System

A computer vision project developed during an internship at CTTC, using **Python, OpenCV, NumPy and Keras/TensorFlow** for face-data collection, preprocessing and real-time face recognition.

## Features

- Captures face samples from an IP camera.
- Detects faces using OpenCV Haar Cascade.
- Converts and consolidates collected face images into training data.
- Supports real-time face recognition using a trained Keras model.
- Uses configurable camera URLs instead of hard-coded local network addresses.

## Project Structure

```text
CTTC-Face-Recognition/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── collect_data.py
│   ├── consolidated_data.py
│   └── recognize.py
├── models/
│   └── haarcascade_frontalface_default.xml
├── data/
└── images/
```

## Installation

```bash
git clone <YOUR-REPOSITORY-URL>
cd CTTC-Face-Recognition
python -m venv venv
```

Activate the environment and install dependencies:

**Windows:**
```bash
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/macOS:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Collect face samples

Set your IP camera snapshot URL using the `CAMERA_URL` environment variable.

Windows PowerShell:
```powershell
$env:CAMERA_URL="http://YOUR-CAMERA-IP:8080/shot.jpg"
python src/collect_data.py
```

The script captures 100 face samples and saves them in `images/`.

### 2. Consolidate the dataset

```bash
python src/consolidated_data.py
```

This creates serialized image and label data in `data/`.

### 3. Train a recognition model

The original internship project used a Keras model saved as `final_model.h5`. That trained model was not included in the available project files, so it is intentionally **not** committed to this repository.

After training your model, place it at:

```text
models/final_model.h5
```

Also update the `LABELS` list in `src/recognize.py` to match the people/classes used by your trained model.

### 4. Run recognition

```bash
python src/recognize.py
```

Press **Q** to exit.

## Technologies Used

- Python
- OpenCV
- NumPy
- TensorFlow / Keras
- Haar Cascade Classifier

## Notes

The repository does not contain personal face images, the trained model, or the original local-network camera addresses. These are environment-specific/generated assets and should be supplied locally when running the project.

## Internship Context

This project was developed as part of an internship at the **Central Tool Room & Training Centre (CTTC)** and demonstrates practical work with computer vision, image preprocessing and machine-learning-based face recognition.
