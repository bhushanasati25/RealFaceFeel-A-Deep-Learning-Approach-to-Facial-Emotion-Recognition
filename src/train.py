import tensorflow as tf
from src.config import DATA_PROCESSED, MODEL_PATH, EPOCHS, LEARNING_RATE
from src.data_manager import load_datasets
from src.model_builder import build_efficientnet_model
import os

def train():
    print("🚀 Starting Training Pipeline...")
    
    # 1. Load Data
    train_ds, val_ds = load_datasets(DATA_PROCESSED)
    
    # 2. Build Model
    model = build_efficientnet_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 3. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]

    # 4. Train
    print("Training started...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()