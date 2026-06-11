# Real-Time Player Confidence & Mood Analysis System

## Project Overview

This repository contains a real-time facial emotion recognition system designed to estimate player confidence and mood from live webcam video. The system detects faces in the video stream, preprocesses each face region, and classifies the emotion using a trained convolutional neural network saved in `model.h5`.

The project is implemented in Python using OpenCV for video capture and face detection, and Keras / TensorFlow for emotion prediction.

## Academic Project Details

- University: Sukkur IBA University
- Department: Computer Science
- Course: Artificial Intelligence (Spring 2026)
- Project title: Real-Time Player Confidence & Mood Analysis System
- Supervisor: Dr. Muhammad Ismail Mangrio, Assistant Professor & Coordinator (CS)
- Team members:
  - Ali Tawassul [023-23-0061]
  - Muhammad Taha [023-23-0070]
  - Inamullah [023-23-0048]

## Key Features

- Real-time webcam video capture
- Face detection using Haar cascades (`haarcascade_frontalface_default.xml`)
- Emotion classification into seven categories:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise
- Live annotation of detected faces with emotion labels
- Lightweight inference pipeline for desktop use

## Repository Structure

- `face-mood-recognition.py` - Main application script for real-time mood detection using webcam input.
- `haarcascade_frontalface_default.xml` - Pre-trained OpenCV cascade file for detecting frontal faces.
- `model.h5` - Trained Keras emotion classification model.
- `train/` - Notebook and dataset organization for training and validation.
  - `train/train.ipynb` - Training notebook for model development.
  - `train/dataset/` - Image folders for training and validation data.
    - `train/` subfolders: angry, disgust, fear, happy, neutral, sad, surprise
    - `validation/` subfolders: angry, disgust, fear, happy, neutral, sad, surprise

## Prerequisites

- Python 3.8 or newer
- Webcam or video capture device connected to the system
- Installation of required Python packages

## Installation

1. Clone the repository or download the files.
2. Create a Python virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:

```bash
pip install opencv-python-headless numpy tensorflow keras
```

> Note: If you want GUI support for OpenCV windows, use `opencv-python` instead of `opencv-python-headless`.

## Running the Application

Run the main script to start the real-time emotion recognition system:

```bash
python face-mood-recognition.py
```

Controls:

- Press `q` to quit the application

The script will open a window displaying the video stream. When a face is detected, a green rectangle is drawn and the predicted emotion label is shown.

## How It Works

1. The webcam video stream is initialized via OpenCV.
2. Each frame is converted to grayscale.
3. Faces are detected using the Haar cascade classifier.
4. Each face region is resized to 48x48 pixels and normalized.
5. The pre-trained Keras model predicts the most likely emotion.
6. The predicted emotion label is overlaid on the live video.

## Dataset and Training

The `train/` folder contains the dataset and a training notebook that demonstrates how the model can be trained. The dataset is organized into standard emotion categories and separated into training and validation splits.

If you want to retrain or improve the model:

1. Open `train/train.ipynb` in Jupyter Notebook or JupyterLab.
2. Prepare the dataset under `train/dataset/train/` and `train/dataset/validation/`.
3. Train a model using the notebook, then save the resulting model as `model.h5`.

## Notes and Extensions

- The current system assumes a single webcam input and works best with good lighting.
- If you want to support multiple faces, the system already processes all detected faces in each frame.
- For improved accuracy, consider training on a larger emotion dataset or using a more advanced face detection model.
- You can extend the application with a GUI dashboard, logging, or confidence scoring for each prediction.

## Troubleshooting

- If the webcam does not open, ensure the device is connected and accessible.
- If the model fails to load, verify that `model.h5` exists and is compatible with the installed Keras/TensorFlow version.
- If face detection is unstable, check that `haarcascade_frontalface_default.xml` is present and correctly referenced.

## License

This project is provided as-is for educational and research purposes. Adjust the license text as needed for your intended use.
