# app.py
import streamlit as st
import cv2
import numpy as np
import traceback
import sys
from pathlib import Path

# ----------------------------
# Safe root resolution & PATH
# ----------------------------
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

# ensure project root is on path so "src" imports work
sys.path.append(str(ROOT))

# ----------------------------
# Import config
# ----------------------------
try:
    from src.config import MODEL_PATH, HAAR_PATH, CLASSES, IMG_SIZE
except Exception as e:
    st.set_page_config(page_title="Config error")
    st.error("Failed to import src.config. Make sure src/config.py exists and defines MODEL_PATH, HAAR_PATH, CLASSES, IMG_SIZE.")
    st.exception(e)
    raise SystemExit

# ----------------------------
# Page config + CSS (your Metallic Dark theme)
# ----------------------------
st.set_page_config(
    page_title="RealFaceFeel: A Deep Learning Approach to Facial Emotion Recognition",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
        
        :root {
            --bg-dark: #09090b;
            --bg-card: #18181b;
            --border-color: #27272a;
            --accent-primary: #3b82f6; /* Blue-500 */
            --accent-glow: rgba(59, 130, 246, 0.5);
            --text-primary: #f4f4f5;
            --text-secondary: #a1a1aa;
        }

        /* Main Container */
        .stApp {
            background-color: var(--bg-dark);
            font-family: 'Inter', sans-serif;
        }

        /* Headers */
        h1, h2, h3 {
            color: var(--text-primary) !important;
            font-weight: 800;
            letter-spacing: -0.025em;
        }

        /* Metallic Header Card */
        .project-header {
            background: linear-gradient(180deg, #1f2937 0%, #111827 100%);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 0 40px -10px rgba(0,0,0,0.5);
            position: relative;
            overflow: hidden;
        }
        
        .project-header::before {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 2px;
            background: linear-gradient(90deg, transparent, var(--accent-primary), transparent);
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0c0c0e;
            border-right: 1px solid var(--border-color);
        }

        /* Documentation Cards (New Tab Style) */
        .doc-section {
            background: #18181b;
            border: 1px solid #27272a;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            height: 100%;
        }
        .doc-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: white;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 10px;
            border-bottom: 1px solid #27272a;
            padding-bottom: 0.5rem;
        }
        .doc-text {
            color: #a1a1aa;
            line-height: 1.6;
            font-size: 0.9rem;
        }
        
        /* Model Tags */
        .tech-tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            background: rgba(59, 130, 246, 0.1);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.2);
            margin-right: 4px;
            margin-bottom: 4px;
        }

        /* Result Cards */
        .result-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.5rem;
        }
        
        /* Custom Button */
        .stButton>button {
            background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
            border: none;
            color: white;
            font-weight: 600;
            transition: all 0.2s;
        }
        .stButton>button:hover {
            box-shadow: 0 0 15px var(--accent-glow);
            transform: translateY(-1px);
        }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ----------------------------
# Load model & cascade defensively
# ----------------------------
@st.cache_resource
def load_resources():
    from tensorflow.keras.models import load_model
    model = None
    face_cascade = None
    errors = []

    # MODEL
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        errors.append(f"Model file not found at: {model_path}")
    else:
        try:
            model = load_model(str(model_path))
        except Exception as e:
            errors.append(f"Failed to load model: {e}\n{traceback.format_exc()}")

    # HAAR
    haar_path = Path(HAAR_PATH)
    if not haar_path.exists():
        errors.append(f"Haar cascade not found at: {haar_path}")
    else:
        try:
            face_cascade = cv2.CascadeClassifier(str(haar_path))
            if face_cascade.empty():
                errors.append(f"Haar cascade loaded but is empty/corrupt: {haar_path}")
        except Exception as e:
            errors.append(f"Failed to load Haar cascade: {e}")

    return model, face_cascade, errors

model, face_cascade, load_errors = load_resources()

# ----------------------------
# Sidebar status
# ----------------------------
with st.sidebar:
    st.markdown("### 🧬 SYSTEM STATUS")
    if load_errors:
        for err in load_errors:
            st.error(err)
    if model:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #059669; color: #34d399; padding: 10px; border-radius: 6px; font-size: 0.9rem; font-weight: 600; display: flex; align-items: center; gap: 8px;">
            <span style="height: 8px; width: 8px; background: #34d399; border-radius: 50%; display: inline-block;"></span>
            MODEL ONLINE
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("MODEL OFFLINE")

    if face_cascade is not None and not face_cascade.empty():
        st.markdown("<div style='color:#60a5fa;font-weight:600;'>HAAR CASCADE LOADED</div>", unsafe_allow_html=True)
    else:
        st.warning("HAAR CASCADE NOT LOADED / INVALID")

    st.markdown("---")
    st.caption("CS583 Final Project | RealFaceFeel: A Deep Learning Approach to Facial Emotion Recognition")

# If model missing, stop with help
if model is None:
    st.warning("⚠️ CRITICAL: Model missing or failed to load. Check sidebar messages and src/config.py -> MODEL_PATH.")
    st.stop()

# ----------------------------
# Utility: prepare roi according to model input shape & IMG_SIZE
# ----------------------------
def prepare_roi_for_model(roi_gray, model):
    # ROI is grayscale numpy array (h,w)
    try:
        if isinstance(IMG_SIZE, (list, tuple)) and len(IMG_SIZE) == 2:
            target_h, target_w = int(IMG_SIZE[0]), int(IMG_SIZE[1])
        else:
            target_h = target_w = int(IMG_SIZE)
    except Exception:
        target_h = target_w = roi_gray.shape[:2]

    # resize & normalize
    roi = cv2.resize(roi_gray, (target_w, target_h)).astype("float32") / 255.0

    # determine expected channels from model.input_shape
    try:
        input_shape = model.input_shape  # e.g. (None, 48, 48, 1) or (None,224,224,3)
        expected_channels = int(input_shape[-1])
    except Exception:
        expected_channels = 1

    if expected_channels == 1:
        roi = np.expand_dims(roi, axis=-1)  # H,W,1
    elif expected_channels == 3:
        roi = np.stack([roi, roi, roi], axis=-1)  # H,W,3
    else:
        # fallback: replicate channels
        roi = np.expand_dims(roi, axis=-1)
        roi = np.tile(roi, (1, 1, expected_channels))

    roi = np.expand_dims(roi, axis=0)  # 1,H,W,C
    return roi

# ----------------------------
# Main processing + rendering
# ----------------------------
def process_and_render(image_in):
    c_img, c_data = st.columns([1.5, 1])

    if image_in is None:
        st.error("No image provided.")
        return

    # ensure BGR
    if image_in.ndim == 2:
        image_bgr = cv2.cvtColor(image_in, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image_in.copy()

    # convert to gray for face detection
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    faces = []
    try:
        if face_cascade is not None and not face_cascade.empty():
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        else:
            faces = []
    except Exception as e:
        st.error(f"Face detection failed: {e}")
        st.exception(traceback.format_exc())
        faces = []

    display_img = image_bgr.copy()

    if len(faces) == 0:
        st.info("No face detected. Improve lighting or try a different image.")
        with c_img:
            st.image(display_img, channels="BGR", use_container_width=True)
        return

    # pick largest face
    face = max(faces, key=lambda r: r[2] * r[3])
    x, y, w, h = face

    # bounding box
    cv2.rectangle(display_img, (x, y), (x + w, y + h), (59, 130, 246), 2)

    # crop with bounds safety
    y1, y2 = max(0, y), min(display_img.shape[0], y + h)
    x1, x2 = max(0, x), min(display_img.shape[1], x + w)
    roi_gray = gray[y1:y2, x1:x2]

    if roi_gray.size == 0:
        st.error("Cropped face region is empty.")
        with c_img:
            st.image(display_img, channels="BGR", use_container_width=True)
        return

    # prepare input for model
    roi_for_model = prepare_roi_for_model(roi_gray, model)

    # predict
    try:
        preds = model.predict(roi_for_model, verbose=0)[0]
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.exception(traceback.format_exc())
        with c_img:
            st.image(display_img, channels="BGR", use_container_width=True)
        return

    # defensive checks
    if not isinstance(preds, (list, np.ndarray)):
        st.error("Model returned unexpected prediction format.")
        with c_img:
            st.image(display_img, channels="BGR", use_container_width=True)
        return

    preds = np.asarray(preds)
    idx = int(np.argmax(preds))
    label = CLASSES[idx] if idx < len(CLASSES) else f"Class_{idx}"
    conf = float(preds[idx])

    # put text
    cv2.putText(display_img, f"{label} {conf*100:.0f}%", (x, max(y - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (59, 130, 246), 2)

    # display image
    with c_img:
        st.image(display_img, channels="BGR", use_container_width=True)

    # data panel
    with c_data:
        st.markdown(f"""
        <div class="result-panel" style="text-align: center;">
            <div style="color: #64748b; font-size: 0.85rem; letter-spacing: 1px; font-weight: 600;">PREDICTED CLASS</div>
            <div style="font-size: 3.5rem; font-weight: 800; color: #f4f4f5; margin: 10px 0;">{label}</div>
            <div style="background: rgba(59, 130, 246, 0.1); color: #60a5fa; display: inline-block; padding: 6px 16px; border-radius: 20px; font-weight: 600; font-size: 0.9rem; border: 1px solid rgba(59, 130, 246, 0.3);">
                CONFIDENCE: {conf*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Confidence Distribution")
        indices = np.argsort(preds)[::-1]
        for i in indices[:5]:
            val = float(preds[i])
            if val > 0.01:
                col_txt, col_bar = st.columns([1, 2])
                with col_txt:
                    st.write(f"{CLASSES[i] if i < len(CLASSES) else i}")
                with col_bar:
                    st.progress(val)

# ----------------------------
# UI: Tabs and inputs
# ----------------------------
t1, t2, t3 = st.tabs(["📤  FILE UPLOAD", "📷  WEBCAM INFERENCE", "📘  PROJECT DETAILS"])

with t1:
    f = st.file_uploader("Select Image", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    if f:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Failed to decode uploaded image.")
        else:
            process_and_render(img)

with t2:
    st.markdown("### Real-Time Inference")
    cam = st.camera_input("Capture", label_visibility="collapsed")
    if cam:
        bytes_data = cam.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.error("Failed to decode camera image.")
        else:
            process_and_render(img)

with t3:
    st.markdown("### 📄 Project Specifications")
    st.markdown("Technical details regarding the architecture, dataset, and evaluation metrics.")
    st.markdown("---")

    # Reference image (local)
    st.markdown(
        f"""
        <div style="margin-bottom: 1rem;">
            <img src="/mnt/data/c12dcdd3-203e-4725-88a3-06bfc6dd8235.png"
                 alt="Project Reference"
                 style="width:100%; height:auto; border-radius:8px; border:1px solid #27272a;" />
        </div>
        """,
        unsafe_allow_html=True
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="doc-section">
            <div class="doc-title">🎯 1. Problem Statement</div>
            <div class="doc-text">
            To develop a facial emotion recognition system classifying human expressions into 
            <b>Happy, Sad, Angry, Surprised, Fear, Disgust, and Neutral</b>.
            <br><br>
            <b>Goal:</b> Bridge the gap between human emotion and machine understanding via robust Deep Learning models and a real-time Streamlit UI.
            <br><br>
            <b>Applications:</b>
            <ul>
                <li>Affective Computing</li>
                <li>Human–Computer Interaction (HCI)</li>
                <li>Mental Health Monitoring</li>
            </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="doc-section">
            <div class="doc-title">🧠 3. Methodology</div>
            <div class="doc-text">
            We implemented a multi-tiered modeling strategy:
            <br><br>
            <b>A. Baseline CNN (Custom)</b><br>
            Lightweight 4-block ConvNet trained from scratch on 48×48 grayscale inputs.
            <br><br>
            <b>B. Transfer Learning (EfficientNetB0)</b><br>
            Fine-tuned on FER2013 using ImageNet weights. Preprocessing: resize to 224×224 and channel-stack grayscale to RGB where required.
            <br><br>
            <b>C. Additional Experiments</b><br>
            ResNet-50 & VGG16 (fine-tuning), Dlib 68-point landmarks + MLP, Vision Transformers (ViT / Swin).
            <br><br>
            <span class="tech-tag">Deep CNN</span>
            <span class="tech-tag">Transfer Learning</span>
            <span class="tech-tag">Data Augmentation</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="doc-section">
            <div class="doc-title">🔮 5. Quantitative Results</div>
            <div class="doc-text">
            • <b>Baseline CNN:</b> Achieved ~65% accuracy on FER2013.<br>
            • <b>EfficientNetB0 (Fine-tuned):</b> Achieved ~73% accuracy (significant improvement).<br>
            • <b>Inference:</b> Real-time capable (~25 ms per frame on target hardware).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="doc-section">
            <div class="doc-title">💾 2. Dataset: FER2013</div>
            <div class="doc-text">
            <b>ICML 2013 / Kaggle Benchmark</b>
            <br><br>
            • <b>Volume:</b> 35,887 grayscale images (48×48 px).<br>
            • <b>Splits:</b> Train (28,709), Val (3,589), Test (3,589).<br>
            • <b>Challenges:</b> In-the-wild faces, occlusion, varying illumination, and class imbalance (e.g., Disgust).<br>
            • <b>Preprocessing:</b> Removed 1,800+ duplicate images to prevent leakage; face detection → crop → normalize; augmentation (flip, shift, brightness).
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="doc-section">
            <div class="doc-title">📈 4. Evaluation Metrics</div>
            <div class="doc-text">
            Due to class imbalance, we used comprehensive metrics:
            <ul>
                <li><b>Categorical Accuracy:</b> Overall correctness.</li>
                <li><b>Macro F1-Score:</b> Critical for minority classes.</li>
                <li><b>Confusion Matrix:</b> Analyze inter-class errors (e.g., Fear vs Surprise).</li>
                <li><b>Precision / Recall:</b> Per-class performance details.</li>
            </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="doc-section">
            <div class="doc-title">🤖 6. AI as a Teammate</div>
            <div class="doc-text">
            AI tools acted as a \"force multiplier\" across the project:
            <ul>
                <li><b>Refactoring:</b> Modularized code for training, inference, and deployment.</li>
                <li><b>Debugging:</b> Resolved input-shape mismatches and preprocessing bugs.</li>
                <li><b>Design:</b> Generated the Metallic Dark UI theme and CSS.</li>
                <li><b>Prototyping:</b> Rapid iteration for model architectures and scripts.</li>
            </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
