import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import os
import traceback

# Add project root to system path (guard if __file__ isn't available)
try:
    ROOT = os.path.join(os.path.dirname(__file__), "..")
    sys.path.append(ROOT)
except Exception:
    # Running in an environment without __file__ (e.g., notebook); do nothing
    pass

# Import config (wrap in try/except for clearer error messages)
try:
    from src.config import MODEL_PATH, HAAR_PATH, CLASSES, IMG_SIZE
except Exception as e:
    st.warning("Could not import `src.config`. Make sure src/config.py exists and defines MODEL_PATH, HAAR_PATH, CLASSES, IMG_SIZE.")
    st.write("Import error:", e)
    # Provide sensible defaults so the app still loads for UI/debugging:
    MODEL_PATH = "models/emotion_model.keras"
    HAAR_PATH = "models/haarcascade_frontalface_default.xml"
    CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    IMG_SIZE = (48, 48)

# ========================================================
# 1. PAGE CONFIGURATION
# ========================================================
st.set_page_config(
    page_title="RealFaceFeel: A Deep Learning Approach to Facial Emotion Recognition",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================================
# 2. METALLIC DARK THEME CSS (unchanged)
# ========================================================
def load_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
        
        :root {
            --bg-dark: #09090b;
            --bg-card: #18181b;
            --border-color: #27272a;
            --accent-primary: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.5);
            --text-primary: #f4f4f5;
            --text-secondary: #a1a1aa;
        }

        .stApp {
            background-color: var(--bg-dark);
            font-family: 'Inter', sans-serif;
        }

        h1, h2, h3 {
            color: var(--text-primary) !important;
            font-weight: 800;
            letter-spacing: -0.025em;
        }

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

        [data-testid="stSidebar"] {
            background-color: #0c0c0e;
            border-right: 1px solid var(--border-color);
        }

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

        .result-panel {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.5rem;
        }

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

# ========================================================
# 3. RESOURCE LOADING (safer)
# ========================================================
@st.cache_resource
def load_resources(model_path, haar_path):
    """Load Keras model and Haar cascade. Return (model, face_cascade, load_errors)"""
    load_errors = []
    model = None
    face_cascade = None

    # Load model
    if not os.path.exists(model_path):
        load_errors.append(f"Model file not found at: {model_path}")
    else:
        try:
            model = load_model(model_path)
        except Exception as e:
            load_errors.append(f"Failed to load model: {e}")

    # Load haar cascade
    if not os.path.exists(haar_path):
        load_errors.append(f"Haar cascade not found at: {haar_path}")
    else:
        try:
            face_cascade = cv2.CascadeClassifier(haar_path)
            # verify it loaded
            if face_cascade.empty():
                load_errors.append("CascadeClassifier loaded but is empty - invalid xml or path.")
        except Exception as e:
            load_errors.append(f"Failed to load Haar cascade: {e}")

    return model, face_cascade, load_errors

model, face_cascade, load_errors = load_resources(MODEL_PATH, HAAR_PATH)

# ========================================================
# 4. SIDEBAR (Minimal Status)
# ========================================================
with st.sidebar:
    st.markdown("### 🧬 SYSTEM STATUS")
    if load_errors:
        for err in load_errors:
            st.error(err)
    if model is not None and face_cascade is not None and not load_errors:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #059669; color: #34d399; padding: 10px; border-radius: 6px; font-size: 0.9rem; font-weight: 600; display: flex; align-items: center; gap: 8px;">
            <span style="height: 8px; width: 8px; background: #34d399; border-radius: 50%; display: inline-block;"></span>
            MODEL ONLINE
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Model or Haar cascade is not fully loaded. Some features will be disabled.")

    st.markdown("---")
    st.caption("CS583 Final Project | RealFaceFeel: A Deep Learning Approach to Facial Emotion Recognition")

# ========================================================
# 5. MAIN CONTENT
# ========================================================
st.markdown("""
<div class="project-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 2.5rem;">RealFaceFeel: A Deep Learning Approach to Facial Emotion Recognition</h1>
            <p style="color: #94a3b8; margin: 5px 0 0 0; font-size: 1.1rem;">
                A Deep Learning Approach to Facial Emotion Recognition
            </p>
        </div>
        <div style="text-align: right;">
            <span style="background: #2563eb; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">CS583 PROJECT</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if model is None or face_cascade is None:
    st.warning("⚠️ CRITICAL: Model or face detector missing. Upload model and Haar cascade and restart the app.")
    # We don't `st.stop()` so the UI still loads for debugging. In production you may want to stop.
    # st.stop()

# --- TABS CONFIGURATION ---
t1, t2, t3 = st.tabs(["📤  FILE UPLOAD", "📷  WEBCAM INFERENCE", "📘  PROJECT DETAILS"])

def safe_resize(img, target_size):
    """Resize image while guarding against invalid target_size."""
    try:
        return cv2.resize(img, target_size)
    except Exception:
        # fallback: use cv2.INTER_AREA and convert to correct dtype
        h, w = target_size
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def process_and_render(image_in):
    """Run face detection, inference and render UI."""
    c_img, c_data = st.columns([1.5, 1])

    # Preprocessing
    try:
        gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
    except Exception:
        st.error("Input image must be BGR color np.array.")
        return

    # detect faces (guard if cascade not loaded)
    if face_cascade is None:
        st.error("Face cascade not loaded. Cannot run detection.")
        with c_img:
            st.image(image_in, channels="BGR", use_column_width=True)
        return

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    display_img = image_in.copy()

    if len(faces) == 0:
        st.info("No face detected. Please improve lighting or try a different image.")
        with c_img:
            st.image(image_in, channels="BGR", use_column_width=True)
        return

    # Process largest face
    face = max(faces, key=lambda r: r[2] * r[3])
    x, y, w, h = face

    # sanity check for box size
    if w <= 2 or h <= 2:
        st.warning("Detected face bounding box too small for reliable inference.")
        with c_img:
            st.image(display_img, channels="BGR", use_column_width=True)
        return

    # UI: Bounding Box & label space
    cv2.rectangle(display_img, (x, y), (x + w, y + h), (59, 130, 246), 2)

    # Extract ROI / preprocess for model
    try:
        roi = gray[y:y + h, x:x + w]
        # ensure ROI size not zero
        if roi.size == 0:
            st.error("Empty ROI — face coordinates invalid.")
            with c_img:
                st.image(display_img, channels="BGR", use_column_width=True)
            return

        roi_resized = safe_resize(roi, IMG_SIZE).astype("float32") / 255.0

        # If IMG_SIZE expected by model is (224,224) and grayscale input is single channel,
        # some models expect 3 channels. We'll stack if needed.
        if roi_resized.ndim == 2:
            roi_resized = np.expand_dims(roi_resized, axis=-1)  # H,W,1

        # If model expects 3 channels but we have 1:
        if model is not None:
            # Model input shape inference
            try:
                model_input_shape = model.input_shape  # e.g. (None, H, W, C)
                # detect channel dim
                if len(model_input_shape) == 4:
                    expected_channels = model_input_shape[-1]
                    if expected_channels == 3 and roi_resized.shape[-1] == 1:
                        roi_resized = np.repeat(roi_resized, 3, axis=-1)
            except Exception:
                # ignore and continue
                pass

        roi_batch = np.expand_dims(roi_resized, axis=0)  # 1,H,W,C

    except Exception as e:
        st.error("Error preprocessing ROI: " + str(e))
        traceback.print_exc()
        return

    # Inference
    if model is None:
        st.error("Model not loaded; cannot perform inference.")
        with c_img:
            st.image(display_img, channels="BGR", use_column_width=True)
        return

    try:
        preds = model.predict(roi_batch, verbose=0)[0]
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        traceback.print_exc()
        with c_img:
            st.image(display_img, channels="BGR", use_column_width=True)
        return

    idx = int(np.argmax(preds))
    label = CLASSES[idx] if idx < len(CLASSES) else f"cls_{idx}"
    conf = float(preds[idx])

    # UI: Label
    cv2.putText(display_img, f"{label} {conf*100:.0f}%", (x, max(y - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (59, 130, 246), 2)

    with c_img:
        st.image(display_img, channels="BGR", use_column_width=True)

    # UI: Data Panel
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
                    st.write(f"{CLASSES[i]}")
                with col_bar:
                    # `st.progress` expects 0.0-1.0
                    st.progress(min(max(val, 0.0), 1.0))

# --- TAB 1: UPLOAD ---
with t1:
    f = st.file_uploader("Select Image", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    if f:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Failed to decode uploaded image. Please try another file.")
        else:
            process_and_render(img)

# --- TAB 2: LIVE ---
with t2:
    st.markdown("### Real-Time Inference")
    cam = st.camera_input("Capture", label_visibility="collapsed")
    if cam:
        bytes_data = cam.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.error("Failed to decode camera capture.")
        else:
            process_and_render(img)

# --- TAB 3: PROJECT DETAILS (Updated) ---
with t3:
    st.markdown("### 📄 Project Specifications")
    st.markdown("Technical details regarding the architecture, dataset, and evaluation metrics.")
    st.markdown("---")

    # PROJECT REFERENCE IMAGE — use the local path you supplied
    local_ref_path = "/mnt/data/09d44fb9-da69-4d93-9bf9-55416baf68d0.png"
    if os.path.exists(local_ref_path):
        # Render using streamlit's native st.image (reliable)
        st.markdown("<div style='display:flex; gap:12px; align-items:center;'>", unsafe_allow_html=True)
        st.image(local_ref_path, caption="Project Reference", use_column_width=False, width=420)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning(f"Project reference image not found at: {local_ref_path}")

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
