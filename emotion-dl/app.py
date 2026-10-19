"""
Facial Emotion Recognition & Affective Analytics Studio
Engineered with PyTorch, OpenCV, Grad-CAM Visual Explainability, and Streamlit.
"""
from datetime import datetime
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from model import (
    compute_valence_metrics,
    detect_faces,
    generate_gradcam,
    get_face_detector,
    load_model,
    predict_probs,
)

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- DESIGN SYSTEM & MODERN CSS -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Clean Top Header */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 24px;
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border-radius: 14px;
        color: #F8FAFC;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);
    }
    .app-header-left {
        display: flex;
        flex-direction: column;
    }
    .app-title {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.4px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .app-sub {
        font-size: 13px;
        color: #94A3B8;
        margin-top: 4px;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34D399;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Result Emotion Card */
    .hero-result {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hero-result-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .hero-emoji {
        font-size: 38px;
        line-height: 1;
    }
    .hero-label {
        font-size: 22px;
        font-weight: 700;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: -0.3px;
    }
    .hero-conf {
        font-size: 13px;
        color: #64748B;
        margin-top: 2px;
    }

    /* Metric Badges */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 18px;
    }
    .mini-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
    }
    .mini-card-title {
        font-size: 11px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .mini-card-val {
        font-size: 17px;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }

    /* Empty State Container */
    .empty-state {
        background: #F8FAFC;
        border: 2px dashed #CBD5E1;
        border-radius: 14px;
        padding: 48px 24px;
        text-align: center;
        color: #64748B;
    }
    .empty-icon {
        font-size: 42px;
        margin-bottom: 12px;
    }
    .empty-title {
        font-size: 17px;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 6px;
    }
    .empty-desc {
        font-size: 13.5px;
        max-width: 420px;
        margin: 0 auto;
        color: #64748B;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- CACHED MODEL & DETECTOR -----------------
@st.cache_resource
def load_all():
    model, classes = load_model("checkpoints/best.pt")
    return model, classes, get_face_detector()


try:
    model, classes, detector = load_all()
except FileNotFoundError:
    st.error("Model weights not found. Run `python train.py` first to generate checkpoints/best.pt.")
    st.stop()

# ----------------- SESSION STATE -----------------
if "session_history" not in st.session_state:
    st.session_state["session_history"] = []

# Emotion metadata: Color codes, Emojis, BGR palettes
EMOTION_META = {
    "happy": {"emoji": "😄", "color": "#10B981", "bgr": (0, 200, 0)},
    "surprise": {"emoji": "😲", "color": "#F59E0B", "bgr": (0, 200, 255)},
    "neutral": {"emoji": "😐", "color": "#6B7280", "bgr": (200, 200, 200)},
    "sad": {"emoji": "😢", "color": "#3B82F6", "bgr": (255, 100, 0)},
    "angry": {"emoji": "😡", "color": "#EF4444", "bgr": (0, 0, 255)},
    "fear": {"emoji": "😨", "color": "#8B5CF6", "bgr": (128, 0, 128)},
    "disgust": {"emoji": "🤢", "color": "#14B8A6", "bgr": (0, 128, 0)},
}


# ----------------- CLEAN SIDEBAR -----------------
with st.sidebar:
    st.markdown("### ⚙️ Studio Controls")
    
    # Primary view toggle
    show_gradcam = st.toggle("🔥 Visual Explainability (Grad-CAM)", value=True,
                             help="Highlights salient facial regions that guided the CNN prediction.")

    # Collapsible Advanced Tuning
    with st.expander("🛠️ Advanced Detection Tuning", expanded=False):
        min_face_size = st.slider(
            "Minimum Face Size", min_value=32, max_value=120, value=48, step=4,
            help="Filter small background faces."
        )
        face_scale_factor = st.select_slider(
            "Detector Scale Factor", options=[1.05, 1.1, 1.15, 1.2], value=1.1,
            help="Smaller values detect more faces with slightly higher compute."
        )
        fear_sensitivity = st.slider(
            "Fear Sensitivity Prior", min_value=1.0, max_value=2.5, value=1.4, step=0.1,
            help="Calibrates prior for subtle expressions. A 1.4x factor doubles Fear recall without hurting accuracy."
        )

    st.markdown("---")
    st.markdown("### 📊 Session Quick Stats")
    history = st.session_state["session_history"]
    count = len(history)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total Detections", count)
    with c2:
        if count > 0:
            avg_c = np.mean([h["confidence"] for h in history])
            st.metric("Mean Conf.", f"{avg_c:.1f}%")
        else:
            st.metric("Mean Conf.", "--")

    if count > 0:
        df_h = pd.DataFrame(history)
        dom_emotion = df_h["emotion"].mode()[0]
        meta = EMOTION_META.get(dom_emotion, {"emoji": "🎭", "color": "#3B82F6"})
        
        st.markdown(f"""
        <div style="background:{meta['color']}15; border:1px solid {meta['color']}40; border-radius:8px; padding:10px 14px; margin: 12px 0;">
            <div style="font-size:11px; font-weight:600; color:#64748B;">DOMINANT SESSION AFFECT</div>
            <div style="font-size:17px; font-weight:700; color:{meta['color']}; margin-top:2px;">
                {meta['emoji']} {dom_emotion.upper()}
            </div>
        </div>
        """, unsafe_allow_html=True)

        csv_bytes = df_h.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Session Log",
            data=csv_bytes,
            file_name=f"emotion_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state["session_history"] = []
            st.rerun()
    else:
        st.caption("Upload or capture a photo to begin tracking live affect metrics.")


# ----------------- MODERN TOP HEADER -----------------
st.markdown("""
<div class="app-header">
    <div class="app-header-left">
        <h1 class="app-title">🎭 Facial Emotion Recognition Studio</h1>
        <div class="app-sub">Deep Learning Facial Affect Analysis • 8-Layer CNN • Grad-CAM Explainability</div>
    </div>
    <div class="status-badge">
        <span>●</span> Model Active (7 Classes)
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------- THREE ORGANIZED TABS -----------------
tab_detect, tab_history, tab_model = st.tabs([
    "📸 Live Analysis",
    "📈 Session Analytics",
    "🔬 Architecture & Benchmarks"
])


# ====================================================
# TAB 1: LIVE ANALYSIS (SIMPLIFIED & INTUITIVE)
# ====================================================
with tab_detect:
    col_input, col_result = st.columns([1, 1.3], gap="large")

    with col_input:
        st.markdown("##### 1. Input Source")
        source_mode = st.radio(
            "Source Selection",
            ["📁 Upload Photo", "📷 Web Camera"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if source_mode == "📁 Upload Photo":
            uploaded_file = st.file_uploader(
                "Upload a portrait image",
                type=["jpg", "jpeg", "png", "jfif", "webp", "bmp", "tiff"],
                help="Supports JPG, JPEG, PNG, JFIF, WEBP, BMP."
            )
        else:
            uploaded_file = st.camera_input("Take a snapshot", label_visibility="collapsed")

    # Ingestion & Detection Logic
    if uploaded_file is not None:
        raw_img = np.array(Image.open(uploaded_file).convert("RGB"))
        gray_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(
            gray_img,
            scaleFactor=face_scale_factor,
            minNeighbors=5,
            minSize=(min_face_size, min_face_size)
        )

        with col_input:
            st.markdown("##### 2. Processed Canvas")
            if len(faces) == 0:
                st.warning("No face detected. Please ensure good frontal lighting and direct camera angle.")
                st.image(raw_img, use_container_width=True)
            else:
                annotated_img = raw_img.copy()
                face_data = []

                for (x, y, w, h) in faces:
                    crop = gray_img[y:y + h, x:x + w]
                    probs = predict_probs(model, crop, fear_boost=fear_sensitivity, classes=classes)
                    top_idx = int(probs.argmax())
                    top_label = classes[top_idx]
                    top_conf = float(probs[top_idx] * 100)
                    v_metrics = compute_valence_metrics(probs, classes)

                    # Log to session state
                    st.session_state["session_history"].append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "emotion": top_label,
                        "confidence": round(top_conf, 1),
                        "valence_score": v_metrics["valence_score"],
                        "mood": v_metrics["dominant_mood"],
                    })

                    meta = EMOTION_META.get(top_label, {"color": "#3B82F6", "bgr": (0, 200, 0), "emoji": "😐"})
                    cv2.rectangle(annotated_img, (x, y), (x + w, y + h), meta["bgr"], 3)

                    # Modern text badge overlay
                    label_str = f"{top_label.upper()} {top_conf:.0f}%"
                    (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated_img, (x, max(y - th - 12, 0)), (x + tw + 10, max(y, th + 12)), meta["bgr"], -1)
                    cv2.putText(annotated_img, label_str, (x + 5, max(y - 6, th + 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    face_data.append((x, y, w, h, crop, probs, top_idx, top_label, top_conf, v_metrics, meta))

                st.image(annotated_img, caption=f"Located {len(faces)} face(s)", use_container_width=True)

        with col_result:
            st.markdown("##### 3. Diagnostic Results & Insights")
            if len(faces) == 0:
                st.info("Awaiting a clear face in the uploaded image to generate diagnostics.")
            else:
                # Tabbed face selector if multiple faces are found
                if len(faces) > 1:
                    face_tabs = st.tabs([f"Face #{i+1}" for i in range(len(faces))])
                else:
                    face_tabs = [st.container()]

                for idx, tab_box in enumerate(face_tabs):
                    x, y, w, h, crop, probs, top_idx, top_label, top_conf, v_metrics, meta = face_data[idx]

                    with tab_box:
                        # Hero Result Card
                        st.markdown(f"""
                        <div class="hero-result" style="border-left: 5px solid {meta['color']};">
                            <div class="hero-result-left">
                                <div class="hero-emoji">{meta['emoji']}</div>
                                <div>
                                    <div class="hero-label" style="color:{meta['color']};">{top_label}</div>
                                    <div class="hero-conf">Confidence: <strong>{top_conf:.1f}%</strong> • Mood: <strong>{v_metrics['dominant_mood']}</strong></div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Mini Metric Badges
                        st.markdown(f"""
                        <div class="metric-row">
                            <div class="mini-card">
                                <div class="mini-card-title">Net Valence</div>
                                <div class="mini-card-val" style="color: {'#10B981' if v_metrics['valence_score'] > 0 else '#EF4444'};">
                                    {v_metrics['valence_score']:+.2f}
                                </div>
                            </div>
                            <div class="mini-card">
                                <div class="mini-card-title">Arousal Energy</div>
                                <div class="mini-card-val">{v_metrics['arousal_score']:.2f}</div>
                            </div>
                            <div class="mini-card">
                                <div class="mini-card-title">Face Region</div>
                                <div class="mini-card-val">{w}×{h} px</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Side-by-side: Visual explainability & Probability breakdown
                        res_c1, res_c2 = st.columns([1, 1.25], gap="medium")

                        with res_c1:
                            if show_gradcam:
                                st.caption("🔥 **Grad-CAM Saliency Heatmap**:")
                                cam_overlay, _, _ = generate_gradcam(model, crop, target_class=top_idx)
                                st.image(cam_overlay, use_container_width=True,
                                         caption="Red regions indicate key CNN focus")
                            else:
                                st.caption("Face Crop (48×48 input):")
                                st.image(crop, use_container_width=True)

                        with res_c2:
                            st.caption("Class Probability Distribution (%):")
                            chart_df = pd.DataFrame({
                                "Emotion": [c.capitalize() for c in classes],
                                "Probability (%)": [float(p * 100) for p in probs]
                            }).set_index("Emotion")
                            st.bar_chart(chart_df, horizontal=True)

    else:
        with col_result:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🖼️</div>
                <div class="empty-title">Ready for Visual Analysis</div>
                <div class="empty-desc">
                    Upload an image or take a portrait photo to run multi-scale face localization,
                    7-class emotion classification, and Grad-CAM visual explainability.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ====================================================
# TAB 2: SESSION ANALYTICS
# ====================================================
with tab_history:
    st.markdown("### 📊 Continuous Session Telemetry")
    history = st.session_state["session_history"]

    if len(history) == 0:
        st.info("No session telemetry logged yet. Analyze portraits in Tab 1 to track cumulative affect data.")
    else:
        df_all = pd.DataFrame(history)

        # High-level KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Total Detections", len(df_all))
        with k2:
            st.metric("Average Confidence", f"{df_all['confidence'].mean():.1f}%")
        with k3:
            st.metric("Net Mean Valence", f"{df_all['valence_score'].mean():+.2f}")
        with k4:
            st.metric("Dominant Affect", df_all['emotion'].mode()[0].capitalize())

        st.markdown("---")

        c_chart1, c_chart2 = st.columns(2, gap="large")
        with c_chart1:
            st.markdown("##### Cumulative Emotion Frequency")
            st.bar_chart(df_all["emotion"].value_counts())

        with c_chart2:
            st.markdown("##### Emotional Valence Trajectory")
            timeline_series = df_all.set_index("time")["valence_score"]
            st.line_chart(timeline_series)

        st.markdown("---")
        st.markdown("##### Detection Event Log")
        st.dataframe(df_all, use_container_width=True)


# ====================================================
# TAB 3: ARCHITECTURE & BENCHMARKS
# ====================================================
with tab_model:
    st.markdown("### 🔬 Model Specifications & Empirical Benchmarks")

    col_arch, col_eval = st.columns(2, gap="large")

    with col_arch:
        st.markdown("""
        #### 🏗️ EmotionCNN Architecture
        - **Model Family:** 8-Layer VGG-Style Deep CNN
        - **Input Shape:** 48 × 48 Grayscale (Single-Channel)
        - **Convolutional Backbone:**
          - 4 Stages with 2× Conv3x3 + BatchNorm + ReLU + MaxPool + Dropout(0.25)
          - Channel progression: 32 ➔ 64 ➔ 128 ➔ 256
        - **Compression & Head:**
          - AdaptiveAvgPool2d(1) ➔ Linear(256 ➔ 256) ➔ ReLU ➔ Linear(256 ➔ 7)
        - **Parameter Footprint:** ~1.25M weights (~4.98 MB binary)
        - **Inference Latency:** &lt; 28 ms on CPU (~35 FPS)
        """)

    with col_eval:
        st.markdown("""
        #### 📈 FER-2013 Test Benchmarks (7,178 Images)
        - **Overall Test Accuracy:** **57.48%**
        - **Macro F1-Score:** **52.0%**
        - **Class-Wise Breakdown:**
          - **Happy:** 85.5% recall | 0.829 F1-score
          - **Surprise:** 81.9% recall | 0.743 F1-score
          - **Neutral:** 65.8% recall | 0.570 F1-score
          - **Disgust:** 45.9% recall | 0.402 F1-score
          - **Angry:** 39.2% recall | 0.430 F1-score
          - **Sad:** 38.1% recall | 0.408 F1-score
          - **Fear:** 21.1% recall | 0.255 F1-score (calibrated prior)
        """)

    st.markdown("---")
    st.markdown("#### Diagnostic Plots")
    d1, d2 = st.columns(2, gap="medium")

    with d1:
        if os.path.exists("checkpoints/confusion_matrix.png"):
            st.image("checkpoints/confusion_matrix.png", caption="Normalized Test Confusion Matrix", use_container_width=True)
    with d2:
        if os.path.exists("checkpoints/training_curves.png"):
            st.image("checkpoints/training_curves.png", caption="Training & Validation Curves (20 Epochs)", use_container_width=True)
