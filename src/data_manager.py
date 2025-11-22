import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications.efficientnet import preprocess_input
from src.config import IMG_SIZE_EFFNET, BATCH_SIZE

def load_datasets(data_dir):
    """Loads and preprocesses data specifically for EfficientNet."""
    
    # 1. Load Data
    train_ds = image_dataset_from_directory(
        data_dir + '/train',
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale', # Load as grayscale first
        image_size=(48, 48),    # Original size
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_ds = image_dataset_from_directory(
        data_dir + '/validation',
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        image_size=(48, 48),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 2. Preprocessing Function (Resize & RGB conversion)
    def preprocess(image, label):
        image = tf.image.grayscale_to_rgb(image)
        image = tf.image.resize(image, IMG_SIZE_EFFNET)
        image = preprocess_input(image) # EfficientNet specific scaling
        return image, label

    # 3. Apply & Optimize
    train_data = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_data = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    return train_data.prefetch(tf.data.AUTOTUNE), val_data.prefetch(tf.data.AUTOTUNE)