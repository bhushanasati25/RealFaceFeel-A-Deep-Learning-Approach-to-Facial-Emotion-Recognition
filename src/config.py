# src/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Path to the trained Keras model (update if your filename differs)
MODEL_PATH = str(ROOT / "models" / "emotion_model.keras")

# Haar cascade path (download from OpenCV if missing)
HAAR_PATH = str(ROOT / "models" / "haarcascade_frontalface_default.xml")

# CLASSES must match the order used during training
CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# IMG_SIZE used during training (height, width)
IMG_SIZE = (48, 48)
