"""
Configuration file for DeepFER Project
Contains paths, hyperparameters, and constants
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import cv2

# =====================================================
# PATHS
# =====================================================
# Get the absolute path of the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_RAW = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'emotion_model.keras')

# Haar Cascade for face detection
HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# =====================================================
# MODEL PARAMETERS
# =====================================================
# Model input sizes
IMG_SIZE = (48, 48)           # Standard FER input size
IMG_SIZE_EFFNET = (224, 224)  # EfficientNet input size

# Emotion classes (7 basic emotions)
CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# =====================================================
# TRAINING HYPERPARAMETERS
# =====================================================
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4