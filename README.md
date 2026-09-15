# CTTC Face Recognition System

A real-time face detection and recognition system developed during my internship at the Central Tool Room & Training Centre (CTTC), using Python, OpenCV and TensorFlow/Keras.

## Overview

This project implements a computer-vision pipeline for collecting face samples, preprocessing image data and performing real-time face recognition.

The system uses OpenCV's Haar Cascade classifier for face detection and a trained Keras/TensorFlow model for face recognition.

## Features

- Real-time face detection using OpenCV Haar Cascade
- Face sample collection from an IP camera
- Image preprocessing and dataset consolidation
- Machine-learning-based face recognition
- Configurable camera URL using an environment variable
- Modular Python source files

## Technologies Used

- Python
- OpenCV
- NumPy
- TensorFlow / Keras
- Haar Cascade Classifier

## Project Structure

```text
CTTC-Face-Recognition/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── collect_data.py
│   ├── consolidated_data.py
│   └── recognize.py
│
└── models/
    └── haarcascade_frontalface_default.xml
