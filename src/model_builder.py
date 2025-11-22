import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from src.config import IMG_SIZE_EFFNET

def build_efficientnet_model(num_classes=7):
    """Builds a Transfer Learning model using EfficientNetB0."""
    
    # 1. Base Model (Frozen)
    base_model = EfficientNetB0(
        input_shape=(*IMG_SIZE_EFFNET, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False 

    # 2. Custom Head
    inputs = layers.Input(shape=(*IMG_SIZE_EFFNET, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs, outputs, name="DeepFER_EfficientNet")
    return model