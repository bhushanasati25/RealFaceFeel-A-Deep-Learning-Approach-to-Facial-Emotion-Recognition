# 🎭 DeepFER - Deep Learning for Facial Emotion Recognition

A comprehensive deep learning project for real-time facial emotion recognition using CNN and EfficientNet architectures.

## 📋 Project Overview

DeepFER implements state-of-the-art deep learning models to recognize seven basic emotions from facial expressions:
- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😢 Sad
- 😲 Surprise
- 😐 Neutral

## 🏗️ Project Structure

```
DeepFER_Project/
├── app/
│   └── main.py               # Streamlit Web Application
├── data/
│   ├── raw/                  # Original Images (Before processing)
│   └── processed/            # Cleaned/augmented data
├── models/                   # Saved .keras models
├── src/                      # Source Code (The Brains)
│   ├── __init__.py
│   ├── config.py             # Global variables (Paths, Hyperparams)
│   ├── data_manager.py       # Data Loading & Cleaning logic
│   ├── model_builder.py      # CNN & EfficientNet Architectures
│   └── train.py              # Training Loop Script
├── Dockerfile                # Docker instructions
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd DeepFER_Project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your dataset in one of these formats:

**Option A: FER2013 CSV Format**
```
data/raw/fer2013.csv
```

**Option B: Directory Structure**
```
data/raw/
├── Angry/
├── Disgust/
├── Fear/
├── Happy/
├── Sad/
├── Surprise/
└── Neutral/
```

### 3. Configure Settings

Edit `src/config.py` to adjust:
- Model architecture (`MODEL_TYPE`: 'cnn', 'efficientnet', or 'resnet')
- Hyperparameters (learning rate, batch size, epochs)
- Data augmentation settings
- Paths and directories

### 4. Train Your Model

```bash
python src/train.py
```

This will:
- Load and preprocess your data
- Build the specified model architecture
- Train with callbacks (early stopping, learning rate reduction)
- Save the best model to `models/` directory
- Generate training logs for TensorBoard

### 5. Run the Web Application

```bash
streamlit run app/main.py
```

Visit `http://localhost:8501` to use the emotion recognition interface.

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t deepfer:latest .
```

### Run Web Application

```bash
docker run -p 8501:8501 deepfer:latest
```

### Run Training in Docker

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models deepfer:latest python src/train.py
```

## 📊 Model Architectures

### 1. Custom CNN
- 4 convolutional blocks with batch normalization
- MaxPooling and dropout for regularization
- Dense layers with 512 and 256 neurons
- ~5M parameters

### 2. EfficientNet-B0
- Transfer learning from ImageNet
- Fine-tuning capability
- ~4M parameters
- Best for higher accuracy

### 3. ResNet34 (Simplified)
- Residual connections for deeper networks
- Skip connections prevent vanishing gradients
- ~10M parameters

## 🎯 Features

### Data Processing
- Automatic data loading from CSV or directories
- Data augmentation (rotation, shifts, flips, zoom)
- Stratified train/validation/test splits
- Class distribution analysis

### Training
- Multiple architecture options
- Callbacks for optimization:
  - Early stopping
  - Learning rate reduction
  - Model checkpointing
  - TensorBoard logging
- Reproducible results with seed control

### Web Application
- Real-time emotion detection
- Multiple input modes:
  - Upload images
  - Webcam capture
  - Sample images
- Confidence scores and probability distributions
- Modern, responsive UI

## 📈 Performance Tips

1. **Data Quality**: More diverse, high-quality training data improves accuracy
2. **Augmentation**: Enable data augmentation for better generalization
3. **Model Selection**: 
   - Use CNN for faster training and inference
   - Use EfficientNet for higher accuracy
4. **Hyperparameter Tuning**: Adjust learning rate, batch size, and epochs
5. **Transfer Learning**: Use pre-trained weights for EfficientNet

## 🔧 Customization

### Add New Model Architecture

1. Edit `src/model_builder.py`
2. Add new method (e.g., `build_vit()`)
3. Update `src/config.py` with new `MODEL_TYPE`
4. Update `src/train.py` to include new model option

### Modify Data Processing

Edit `src/data_manager.py`:
- Add custom preprocessing steps
- Implement new augmentation strategies
- Support additional data formats

### Customize Web UI

Edit `app/main.py`:
- Modify CSS styling
- Add new features (face detection, batch processing)
- Implement emotion tracking over time

## 📝 Configuration Reference

Key settings in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMAGE_SIZE` | (48, 48) | Input image dimensions |
| `BATCH_SIZE` | 64 | Training batch size |
| `EPOCHS` | 50 | Maximum training epochs |
| `LEARNING_RATE` | 0.001 | Initial learning rate |
| `MODEL_TYPE` | 'efficientnet' | Architecture to use |

## 🐛 Troubleshooting

### Model Not Found Error
- Ensure you've trained a model first using `python src/train.py`
- Check that `models/best_model.keras` exists

### Memory Issues
- Reduce `BATCH_SIZE` in config.py
- Use smaller model architecture (CNN instead of EfficientNet)

### Low Accuracy
- Increase training data
- Enable data augmentation
- Try different model architectures
- Increase training epochs

## 📚 Resources

- [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- FER2013 dataset creators
- TensorFlow and Keras teams
- Streamlit community

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ using TensorFlow, Keras, and Streamlit**
