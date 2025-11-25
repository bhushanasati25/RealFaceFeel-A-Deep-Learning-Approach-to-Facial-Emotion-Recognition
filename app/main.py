import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import os

# Add project root to system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import MODEL_PATH, HAAR_PATH, CLASSES, IMG_SIZE

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
# 2. METALLIC DARK THEME CSS
# ========================================================
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

# ========================================================
# 3. RESOURCE LOADING
# ========================================================
@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        model = load_model(MODEL_PATH)
        face_cascade = cv2.CascadeClassifier(HAAR_PATH)
        return model, face_cascade
    except:
        return None, None

model, face_cascade = load_resources()

# ========================================================
# 4. SIDEBAR (Minimal Status)
# ========================================================
with st.sidebar:
    st.markdown("### 🧬 SYSTEM STATUS")
    
    if model:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #059669; color: #34d399; padding: 10px; border-radius: 6px; font-size: 0.9rem; font-weight: 600; display: flex; align-items: center; gap: 8px;">
            <span style="height: 8px; width: 8px; background: #34d399; border-radius: 50%; display: inline-block;"></span>
            MODEL ONLINE
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("MODEL OFFLINE")

    st.markdown("---")
    st.caption("CS583 Final Project | RealFaceFeel: A Deep Learning Approach to Facial Emotion Recognition")

# ========================================================
# 5. MAIN CONTENT
# ========================================================

# PROJECT HEADER
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

if not model:
    st.warning("⚠️ CRITICAL: Model file missing. Upload `emotion_model.keras` to `models/` directory.")
    st.stop()

# --- TABS CONFIGURATION ---
t1, t2, t3 = st.tabs(["📤  FILE UPLOAD", "📷  WEBCAM INFERENCE", "📘  PROJECT DETAILS"])

def process_and_render(image_in):
    c_img, c_data = st.columns([1.5, 1])

    # Preprocessing
    gray = cv2.cvtColor(image_in, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    display_img = image_in.copy()
    
    if len(faces) == 0:
        st.info("No biometric data detected. Please improve lighting.")
        with c_img:
            st.image(image_in, channels="BGR", use_container_width=True)
        return

    # Process Largest Face
    face = max(faces, key=lambda r: r[2] * r[3])
    x, y, w, h = face
    
    # UI: Bounding Box
    cv2.rectangle(display_img, (x, y), (x+w, y+h), (59, 130, 246), 2)
    
    # Inference
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, IMG_SIZE).astype("float") / 255.0
    roi = np.expand_dims(np.expand_dims(roi, axis=-1), axis=0)
    
    preds = model.predict(roi, verbose=0)[0]
    idx = np.argmax(preds)
    label = CLASSES[idx]
    conf = preds[idx]

    # UI: Label
    cv2.putText(display_img, f"{label} {conf*100:.0f}%", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (59, 130, 246), 2)
    
    with c_img:
        st.image(display_img, channels="BGR", use_container_width=True)

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
            val = preds[i]
            if val > 0.01:
                col_txt, col_bar = st.columns([1, 2])
                with col_txt:
                    st.write(f"{CLASSES[i]}")
                with col_bar:
                    st.progress(float(val))

# --- TAB 1: UPLOAD ---
with t1:
    f = st.file_uploader("Select Image", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    if f:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        process_and_render(img)

# --- TAB 2: LIVE ---
with t2:
    st.markdown("### Real-Time Inference")
    cam = st.camera_input("Capture", label_visibility="collapsed")
    if cam:
        bytes_data = cam.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        process_and_render(img)

# --- TAB 3: PROJECT DETAILS (Updated) ---
with t3:
    st.markdown("### 📄 Project Specifications")
    st.markdown("Technical details regarding the architecture, dataset, and evaluation metrics.")
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # 1. Problem Statement
        st.markdown("""
        <div class="doc-section">
            <div class="doc-title">🎯 1. Problem Statement</div>
            <div class="doc-text">
            We aim to develop a facial emotion recognition system that can classify human facial expressions into 
            categories such as <b>Happy, Sad, Angry, Surprised, Fear, Disgust, and Neutral</b>.
            <br><br>
            <b>Goal:</b> Build a deep learning model to interpret visual cues and map them to emotional states 
            via a real-time Streamlit web application.
            <br><br>
            <b>Applications:</b>
            <ul>
                <li>Affective Computing</li>
                <li>Human-Computer Interaction (HCI)</li>
                <li>Mental Health Monitoring</li>
            </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 3. Models Tried
        st.markdown("""
        <div class="doc-section">
            <div class="doc-title">🧠 3. Model Experiments</div>
            <div class="doc-text">
            We experimented with the following architectures:
            <ul>
                <li><b>Baseline CNN:</b> Custom CNN trained from scratch (48x48 input).</li>
                <li><b>Transfer Learning:</b> ResNet-50 & VGG16 fine-tuned on FER2013.</li>
                <li><b>Dlib + MLP:</b> 68-point landmark extraction.</li>
                <li><b>Transformers:</b> Evaluation of ViT and Swin Transformers.</li>
            </ul>
            <br>
            <span class="tech-tag">Deep CNN</span>
            <span class="tech-tag">Transfer Learning</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 5. Performance Expectations
        st.markdown("""
        <div class="doc-section">
            <div class="doc-title">🔮 5. Performance Expectations</div>
            <div class="doc-text">
            • <b>Baseline CNN:</b> Expected ~70–75% accuracy.<br>
            • <b>Target Accuracy:</b> Aiming for <b>85–90%</b> with Transfer Learning & Augmentation.<br>
            • <b>Challenges:</b> Performance bounded by label noise and dataset difficulty (occlusions/illumination).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        # 2. Dataset Info
        st.markdown("""
        <div class="doc-section">
            <div class="doc-title">💾 2. Dataset: FER2013</div>
            <div class="doc-text">
            <b>ICML 2013 Kaggle Challenge Benchmark</b>
            <br><br>
            • <b>Volume:</b> 35,887 Grayscale Images (48x48 pixels).<br>
            • <b>Diversity:</b> "In-the-wild" faces with noise & varying illumination.<br>
            • <b>Splits:</b>
            <ul>
                <li>Training: 28,709</li>
                <li>Validation: 3,589</li>
                <li>Test: 3,589</li>
            </ul>
            • <b>Note:</b> Imbalanced classes (e.g., Disgust is underrepresented).
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4. Evaluation Metrics
        st.markdown("""
        <div class="doc-section">
            <div class="doc-title">📈 4. Evaluation Metrics</div>
            <div class="doc-text">
            Performance is measured using:
            <ul>
                <li><b>Accuracy:</b> Standard metric on FER2013 test set.</li>
                <li><b>Macro F1-Score:</b> To account for class imbalance.</li>
                <li><b>Confusion Matrix:</b> Identify inter-class misclassifications (e.g., Fear vs Surprise).</li>
                <li><b>Precision/Recall:</b> Granular per-class insight.</li>
            </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 6. AI Collaboration
        st.markdown("""
        <div class="doc-section">
            <div class="doc-title">🤖 6. AI as a Teammate</div>
            <div class="doc-text">
            We utilized AI tools throughout the project lifecycle:
            <ul>
                <li><b>Preprocessing:</b> Dlib/MediaPipe code generation.</li>
                <li><b>Modeling:</b> PyTorch/TensorFlow architectural support.</li>
                <li><b>Debugging:</b> AI assistants for prototyping.</li>
                <li><b>Deployment:</b> Streamlit integration support.</li>
            </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)