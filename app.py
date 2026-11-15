"""
Streamlit Web Dashboard for Multimodal Deepfake Detection & Explainable Content Verification.
Professional AI SaaS Interface supporting Image, Video, Audio, and Multimodal Forensic Analysis.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe cache directories
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")

import json
import time
import cv2
import numpy as np
from PIL import Image
import streamlit as st

from configs.config import config
from inference.image_inference import ImageInferencePipeline
from inference.video_inference import VideoInferencePipeline
from inference.audio_inference import AudioInferencePipeline
from models.multimodal_model import MultimodalVerifier
from utils.device import get_device, get_device_name
from utils.helpers import format_file_size, load_json

# Page configuration
st.set_page_config(
    page_title="DeepGuard AI • Enterprise Multimodal Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SaaS Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }

    /* SaaS Header Banner */
    .saas-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .saas-header h1 {
        margin: 0;
        font-size: 26px;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .saas-header p {
        margin: 4px 0 0 0;
        font-size: 13px;
        color: #94a3b8;
    }

    /* Status Telemetry Pills */
    .telemetry-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 12px;
        color: #cbd5e1;
        font-weight: 500;
    }
    .status-dot-green {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
    }

    /* Elevated Surface Cards */
    .saas-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .saas-card h3 {
        margin-top: 0;
        font-size: 17px;
        font-weight: 600;
        color: #e2e8f0;
    }

    /* Prediction Verdict Cards */
    .verdict-hero {
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .verdict-real {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.4) 0%, rgba(6, 95, 70, 0.2) 100%);
        border: 1px solid #10b981;
    }
    .verdict-fake {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.4) 0%, rgba(153, 27, 27, 0.2) 100%);
        border: 1px solid #ef4444;
    }
    .verdict-warning {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.4) 0%, rgba(146, 64, 14, 0.2) 100%);
        border: 1px solid #f59e0b;
    }
    .verdict-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .text-real { color: #34d399; }
    .text-fake { color: #f87171; }
    .text-warning { color: #fbbf24; }

    /* Confidence Bars */
    .confidence-meter-container {
        background: #1f2937;
        border-radius: 9999px;
        height: 12px;
        overflow: hidden;
        margin: 12px 0 6px 0;
        display: flex;
    }
    .meter-fill-real {
        background: linear-gradient(90deg, #059669, #10b981);
        height: 100%;
        transition: width 0.5s ease;
    }
    .meter-fill-fake {
        background: linear-gradient(90deg, #dc2626, #ef4444);
        height: 100%;
        transition: width 0.5s ease;
    }

    /* Metric Tiles Grid */
    .metric-chip {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
    }
    .metric-chip-label {
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .metric-chip-value {
        font-size: 16px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 2px;
    }

    /* Navigation Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0d121f;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        color: #94a3b8;
        font-weight: 600;
    }

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        background-color: #111827;
        border-radius: 8px;
        color: #94a3b8;
        border: 1px solid #1f2937;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
        font-weight: 600;
    }

    /* Frame Gallery Card */
    .frame-card {
        background: #1e2433;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 8px;
        text-align: center;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# Cached pipeline loaders with automatic device selection
@st.cache_resource
def get_image_pipeline():
    return ImageInferencePipeline()

@st.cache_resource
def get_video_pipeline():
    return VideoInferencePipeline()

@st.cache_resource
def get_audio_pipeline():
    return AudioInferencePipeline()

@st.cache_resource
def get_multimodal_verifier():
    return MultimodalVerifier()


# Hardware context
device = get_device()
hardware_name = get_device_name(device)

# Sidebar Navigation & Telemetry
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 14px;">
        <span style="font-size: 30px;">🛡️</span>
        <div>
            <div style="font-weight: 800; font-size: 18px; color: #f8fafc; letter-spacing: 0.5px;">DeepGuard AI</div>
            <div style="font-size: 11px; color: #64748b; font-weight: 500;">Enterprise Forensics SaaS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="telemetry-pill" style="width: 100%; justify-content: center; margin-bottom: 16px;">
        <span class="status-dot-green"></span>
        <span>Engine Active • {device.type.upper()}</span>
    </div>
    """, unsafe_allow_html=True)

    navigation = st.radio(
        "FORENSIC SUITE",
        [
            "Overview & Telemetry",
            "Image Verification",
            "Video Verification",
            "Audio Verification",
            "Audio-Visual Consistency",
            "Explainability & XAI",
            "Enterprise Benchmarks",
            "System Architecture",
        ],
        label_visibility="visible"
    )

    st.markdown("---")
    st.caption("**System Specifications**")
    st.markdown(f"""
    - **Compute Engine:** `{hardware_name}`
    - **PyTorch Hub Cache:** `Active (/tmp)`
    - **Image Backbone:** `ResNet18 (Fine-Tuned)`
    - **Audio Backbone:** `4-Stage 2D CNN`
    - **Multimodal Engine:** `Late Fusion (wv=0.55)`
    """)

    if st.button("🧹 Clear Session & Cache"):
        st.session_state.clear()
        st.cache_resource.clear()
        st.rerun()


# Helper: Downloadable Structured Reports
def make_download_report(results: dict, modality: str) -> str:
    report_dict = {
        "platform": "DeepGuard AI Enterprise Forensics Suite",
        "modality": modality.upper(),
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "hardware_compute": hardware_name,
        "forensic_audit": {
            k: v for k, v in results.items()
            if not isinstance(v, (np.ndarray, Image.Image))
        }
    }
    return json.dumps(report_dict, indent=4)


def make_markdown_certificate(results: dict, modality: str) -> str:
    verdict = results.get("verdict", "N/A")
    confidence = results.get("confidence", 0.0)
    latency = results.get("processing_time_sec", results.get("total_processing_time_sec", 0.0))

    lines = [
        f"# DEEPGUARD AI — FORENSIC AUDIT CERTIFICATE",
        f"**Modality:** {modality.upper()} | **Audit Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Hardware Environment:** {hardware_name}",
        f"",
        f"---",
        f"## Executive Forensic Verdict: **{verdict}**",
        f"- **Model Confidence Score:** {confidence * 100:.2f}%",
        f"- **Analysis Latency:** {latency:.3f} seconds",
        f"- **Authenticity Likelihood:** {(results.get('real_probability', 1.0 - confidence)) * 100:.2f}%",
        f"- **Manipulation Likelihood:** {(results.get('fake_probability', confidence)) * 100:.2f}%",
        f"",
        f"---",
        f"## Forensic Details & Methodology",
        f"{results.get('explanation', 'No detailed notes logged.')}",
        f"",
        f"---",
        f"*Scientific Disclosure: This certificate represents statistical feature extraction from deep neural networks. It does not constitute a legal guarantee of media origin.*"
    ]
    return "\n".join(lines)


# =====================================================================
# 1. OVERVIEW & TELEMETRY
# =====================================================================
if navigation == "Overview & Telemetry":
    st.markdown("""
    <div class="saas-header">
        <div>
            <h1>DeepGuard AI Forensics Suite</h1>
            <p>Next-Generation Explainable Artificial Intelligence for Synthetic Media Verification</p>
        </div>
        <div class="telemetry-pill">
            <span class="status-dot-green"></span>
            <span>v1.0.0 Enterprise • Production Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Welcome to the enterprise verification console. DeepGuard AI analyzes digital assets across **visual, acoustic, 
    and cross-modal channels** to detect synthetic manipulation (GANs, Diffusion, voice cloning, and deepfakes) 
    with transparent mathematical explainability.
    """)

    # Modality Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="saas-card">
            <h3>🖼️ Visual Forensics</h3>
            <p style="font-size: 13px; color: #94a3b8;">Fine-tuned ResNet18 CNN detecting localized facial warping, boundary seams, and generative artifacts.</p>
            <div style="font-size: 11px; color: #3b82f6; font-weight: 600;">GRAD-CAM SALIENCY ENABLED</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="saas-card">
            <h3>🎬 Temporal Video</h3>
            <p style="font-size: 13px; color: #94a3b8;">Uniform frame sampling with <code>top_k_mean</code> temporal aggregation to catch intermittent splicing.</p>
            <div style="font-size: 11px; color: #3b82f6; font-weight: 600;">TIMELINE TRAJECTORY ANALYSIS</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="saas-card">
            <h3>🎙️ Acoustic Forensics</h3>
            <p style="font-size: 13px; color: #94a3b8;">4-stage 2D SpectrogramCNN identifying TTS vocoder distortions, robotic harmonics, and phase jumps.</p>
            <div style="font-size: 11px; color: #3b82f6; font-weight: 600;">128-BAND LOG-MEL STFT</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="saas-card">
            <h3>🔗 Audio-Visual Fusion</h3>
            <p style="font-size: 13px; color: #94a3b8;">Cross-modal integrity metric (|P_video - P_audio|) detecting voice dubbing or facial replacement.</p>
            <div style="font-size: 11px; color: #3b82f6; font-weight: 600;">DUAL-STREAM CONGRUENCE</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("System Pipeline & Data Flow")
    st.markdown("""
    ```mermaid
    flowchart LR
        Media[Media Ingestion] --> Router{Container Router}
        Router -->|Image| PrepImg[Face Isolation] --> ResNet[ResNet18 CNN] --> GradCAM[Grad-CAM Saliency Engine]
        Router -->|Video| PrepVid[Uniform Sampling] --> VideoModel[Spatial-Temporal Aggregator] --> Timeline[Suspicion Timeline]
        Router -->|Audio| PrepAud[Mel Spectrogram] --> AudioModel[SpectrogramCNN] --> SpecVis[Acoustic Visualizer]
        
        VideoModel --> Fusion[Cross-Modal Consistency Engine]
        AudioModel --> Fusion
        
        GradCAM --> SaaSReport[Audit Dashboard & Downloadable Certificate]
        Timeline --> SaaSReport
        SpecVis --> SaaSReport
        Fusion --> SaaSReport
    ```
    """)


# =====================================================================
# 2. IMAGE VERIFICATION
# =====================================================================
elif navigation == "Image Verification":
    st.markdown("""
    <div class="saas-header">
        <div>
            <h1>Visual Deepfake Verification</h1>
            <p>Spatial artifact inspection, facial landmark isolation, and Grad-CAM saliency mapping</p>
        </div>
        <div class="telemetry-pill">
            <span class="status-dot-green"></span>
            <span>ResNet18 Backbone Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pipeline = get_image_pipeline()
    sample_img_dir = config.sample_data_dir / "images"

    if "target_image_path" not in st.session_state:
        st.session_state.target_image_path = None

    # Staged Sample Selector Cards
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📁 Load Authentic Sample Image", use_container_width=True):
            st.session_state.target_image_path = sample_img_dir / "sample_real.jpg"
    with col_btn2:
        if st.button("⚠️ Load Manipulated Sample Image", use_container_width=True):
            st.session_state.target_image_path = sample_img_dir / "sample_fake.jpg"

    uploaded_file = st.file_uploader(
        "Upload target portrait (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
        help="Accepts high-resolution portraits and facial imagery."
    )

    if uploaded_file is not None:
        temp_path = config.outputs_dir / f"upload_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.target_image_path = temp_path

    target_image_path = st.session_state.target_image_path

    if target_image_path and Path(target_image_path).exists():
        p_card1, p_card2 = st.columns([1, 2])
        with p_card1:
            st.markdown(f"**Target Asset:** `{Path(target_image_path).name}`")
            st.image(str(target_image_path), use_container_width=True)
            crop_face = st.checkbox("Automated Facial Bounding Box Crop", value=True)
            run_btn = st.button("⚡ Execute Forensic Audit", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Extracting spatial features and generating Grad-CAM activation hooks..."):
                results = pipeline.analyze(target_image_path, crop_face=crop_face)

            verdict = results["verdict"]
            confidence = results["confidence"]
            real_prob = results["real_probability"]
            fake_prob = results["fake_probability"]

            with p_card2:
                # Prediction Hero Card
                v_class = "verdict-fake" if verdict == "FAKE" else "verdict-real"
                t_class = "text-fake" if verdict == "FAKE" else "text-real"
                status_label = "MANIPULATED / SYNTHETIC" if verdict == "FAKE" else "AUTHENTIC / UNMANIPULATED"

                st.markdown(f"""
                <div class="verdict-hero {v_class}">
                    <div style="font-size: 12px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Forensic Determination</div>
                    <div class="verdict-title {t_class}">{status_label}</div>
                    <div style="font-size: 16px; color: #cbd5e1;">Model Confidence: <strong>{confidence * 100:.2f}%</strong></div>
                    <div class="confidence-meter-container">
                        <div class="meter-fill-real" style="width: {real_prob * 100}%;"></div>
                        <div class="meter-fill-fake" style="width: {fake_prob * 100}%;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #94a3b8;">
                        <span>Real: {real_prob * 100:.1f}%</span>
                        <span>Fake: {fake_prob * 100:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Telemetry Metric Chips
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="metric-chip">
                        <div class="metric-chip-label">Latency</div>
                        <div class="metric-chip-value">{results['processing_time_sec']:.3f}s</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-chip">
                        <div class="metric-chip-label">Hardware</div>
                        <div class="metric-chip-value">{device.type.upper()}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    chk_status = "Trained" if results['checkpoint_loaded'] else "Backbone"
                    st.markdown(f"""
                    <div class="metric-chip">
                        <div class="metric-chip-label">Checkpoint</div>
                        <div class="metric-chip-value">{chk_status}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                st.info(results["explanation"])

                # Report Download Buttons
                rep_c1, rep_c2 = st.columns(2)
                with rep_c1:
                    json_str = make_download_report(results, "image")
                    st.download_button(
                        "📥 Export JSON Audit Log",
                        data=json_str,
                        file_name=f"audit_{Path(target_image_path).stem}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                with rep_c2:
                    cert_str = make_markdown_certificate(results, "image")
                    st.download_button(
                        "📄 Download Forensic Certificate",
                        data=cert_str,
                        file_name=f"certificate_{Path(target_image_path).stem}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )

            # Explainability Tabs
            st.markdown("---")
            st.subheader("🔍 Explainable AI (XAI) Saliency Inspection")
            x_tab1, x_tab2, x_tab3 = st.tabs(["Side-by-Side Saliency", "Activation Heatmap", "Technical Interpretation"])

            with x_tab1:
                col_x1, col_x2 = st.columns(2)
                with col_x1:
                    st.caption("**1. Preprocessed Primary Face Region (224x224)**")
                    st.image(results["original_image"], use_container_width=True)
                with col_x2:
                    st.caption("**2. Grad-CAM Saliency Overlay (Layer4[-1] Activation)**")
                    st.image(results["gradcam_overlay"], use_container_width=True)

            with x_tab2:
                col_h1, col_h2 = st.columns([2, 1])
                with col_h1:
                    st.image(results["gradcam_heatmap"], caption="Normalized Gradient Activation Map", clamp=True, use_container_width=True)
                with col_h2:
                    st.markdown("""
                    **Heatmap Colormap Guide:**
                    - 🔴 **Warm / Red:** High positive gradient contribution towards deepfake prediction (boundary seams, chromatic shifts, texture blur).
                    - 🔵 **Cool / Blue:** Natural frequency regions consistent with genuine photography.
                    """)

            with x_tab3:
                st.markdown("""
                **Mathematical Saliency Derivation:**
                $$\\alpha_k^c = \\frac{1}{Z} \\sum_{i} \\sum_{j} \\frac{\\partial y^c}{\\partial A_{ij}^k}, \\quad L_{\\text{Grad-CAM}}^c = \\text{ReLU}\\left( \\sum_k \\alpha_k^c A^k \\right)$$
                Grad-CAM captures the exact spatial coordinates driving the classification. When synthetic manipulation occurs, 
                activations concentrate around the facial boundary perimeter where blending discontinuities appear.
                """)


# =====================================================================
# 3. VIDEO VERIFICATION
# =====================================================================
elif navigation == "Video Verification":
    st.markdown("""
    <div class="saas-header">
        <div>
            <h1>Temporal Video Verification</h1>
            <p>Uniform frame sampling, temporal aggregation heuristics, and suspicious frame timelines</p>
        </div>
        <div class="telemetry-pill">
            <span class="status-dot-green"></span>
            <span>Spatial-Temporal Aggregator</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pipeline = get_video_pipeline()
    sample_vid_dir = config.sample_data_dir / "videos"

    if "target_video_path" not in st.session_state:
        st.session_state.target_video_path = None

    v_btn1, v_btn2 = st.columns(2)
    with v_btn1:
        if st.button("📁 Load Authentic Video Sample", use_container_width=True):
            st.session_state.target_video_path = sample_vid_dir / "sample_real.mp4"
    with v_btn2:
        if st.button("⚠️ Load Manipulated Video Sample", use_container_width=True):
            st.session_state.target_video_path = sample_vid_dir / "sample_fake.mp4"

    uploaded_video = st.file_uploader(
        "Upload target video container (.mp4, .mov, .avi)",
        type=["mp4", "mov", "avi"],
        help="Maximum file size: 100 MB"
    )

    if uploaded_video is not None:
        temp_path = config.outputs_dir / f"upload_{uploaded_video.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.session_state.target_video_path = temp_path

    target_video_path = st.session_state.target_video_path

    if target_video_path and Path(target_video_path).exists():
        st.video(str(target_video_path))
        st.markdown("---")

        c_opt1, c_opt2, c_opt3 = st.columns(3)
        with c_opt1:
            agg_method = st.selectbox(
                "Aggregation Strategy",
                ["top_k_mean", "mean", "max"],
                help="top_k_mean averages the top 25% most suspicious frames to prevent temporal dilution."
            )
        with c_opt2:
            frame_sample_count = st.slider("Frame Sample Count", min_value=8, max_value=32, value=16, step=4)
        with c_opt3:
            crop_face = st.checkbox("Facial Isolation Mode", value=True)

        if st.button("⚡ Execute Video Forensic Audit", type="primary", use_container_width=True):
            pipeline.preprocessor.frame_sample_count = frame_sample_count
            pipeline.video_detector.aggregation_method = agg_method

            with st.spinner(f"Extracting {frame_sample_count} uniform frames and computing temporal suspicion scores..."):
                results = pipeline.analyze(target_video_path, crop_face=crop_face)

            verdict = results["verdict"]
            confidence = results["confidence"]
            real_prob = results["real_probability"]
            fake_prob = results["fake_probability"]

            # Video Verdict Card
            v_class = "verdict-fake" if verdict == "FAKE" else "verdict-real"
            t_class = "text-fake" if verdict == "FAKE" else "text-real"
            status_label = "MANIPULATED VIDEO DETECTED" if verdict == "FAKE" else "AUTHENTIC VIDEO STREAM"

            st.markdown(f"""
            <div class="verdict-hero {v_class}">
                <div style="font-size: 12px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Video Forensic Verdict</div>
                <div class="verdict-title {t_class}">{status_label}</div>
                <div style="font-size: 16px; color: #cbd5e1;">Aggregated Manipulation Confidence: <strong>{confidence * 100:.2f}%</strong></div>
                <div class="confidence-meter-container">
                    <div class="meter-fill-real" style="width: {real_prob * 100}%;"></div>
                    <div class="meter-fill-fake" style="width: {fake_prob * 100}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #94a3b8;">
                    <span>Real: {real_prob * 100:.1f}%</span>
                    <span>Fake: {fake_prob * 100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Metadata Chips
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""
                <div class="metric-chip">
                    <div class="metric-chip-label">Duration</div>
                    <div class="metric-chip-value">{results['metadata']['duration_sec']}s</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-chip">
                    <div class="metric-chip-label">Sampled Frames</div>
                    <div class="metric-chip-value">{len(results['frame_probabilities'])}</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-chip">
                    <div class="metric-chip-label">Total Latency</div>
                    <div class="metric-chip-value">{results['total_processing_time_sec']:.2f}s</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-chip">
                    <div class="metric-chip-label">Throughput</div>
                    <div class="metric-chip-value">{results['throughput_fps']} FPS</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.info(results["explanation"])

            # Temporal Timeline Chart
            st.markdown("### 📈 Temporal Suspicion Trajectory")
            st.image(results["timeline_image"], use_container_width=True)

            # Peak Suspicious Frame
            st.markdown("### 🎯 Peak Suspicious Frame Breakdown")
            pf1, pf2 = st.columns(2)
            with pf1:
                st.caption(f"Peak Frame Index: #{results['peak_frame_index']} | Timestamp: {results['peak_frame_timestamp']}s")
                st.image(results["peak_raw_frame"], use_container_width=True)
            with pf2:
                st.caption("Grad-CAM Saliency Overlay (Peak Frame)")
                st.image(results["peak_cam_overlay"], use_container_width=True)

            # Flagged Frames Grid
            if results["suspicious_frames"]:
                st.markdown("### ⚠️ Flagged Suspicious Frames Gallery")
                flag_cols = st.columns(min(4, len(results["suspicious_frames"])))
                for idx, item in enumerate(results["suspicious_frames"][:8]):
                    with flag_cols[idx % 4]:
                        st.markdown(f"""
                        <div class="frame-card">
                            <div style="font-size: 11px; font-weight: 700; color: #ef4444;">T={item['timestamp_sec']}s (Frame #{item['frame_index']})</div>
                            <div style="font-size: 12px; color: #cbd5e1;">Score: {item['fake_probability'] * 100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(item["image"], use_container_width=True)

            # Download Reports
            st.markdown("---")
            d_c1, d_c2 = st.columns(2)
            with d_c1:
                json_str = make_download_report(results, "video")
                st.download_button(
                    "📥 Export Video JSON Audit Log",
                    data=json_str,
                    file_name=f"audit_video_{Path(target_video_path).stem}.json",
                    mime="application/json",
                    use_container_width=True
                )
            with d_c2:
                cert_str = make_markdown_certificate(results, "video")
                st.download_button(
                    "📄 Download Forensic Certificate",
                    data=cert_str,
                    file_name=f"certificate_video_{Path(target_video_path).stem}.md",
                    mime="text/markdown",
                    use_container_width=True
                )


# =====================================================================
# 4. AUDIO VERIFICATION
# =====================================================================
elif navigation == "Audio Verification":
    st.markdown("""
    <div class="saas-header">
        <div>
            <h1>Acoustic Deepfake & Speech Verification</h1>
            <p>Log-Mel spectrogram feature extraction and 2D SpectrogramCNN voice spoofing detection</p>
        </div>
        <div class="telemetry-pill">
            <span class="status-dot-green"></span>
            <span>SpectrogramCNN Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pipeline = get_audio_pipeline()
    sample_aud_dir = config.sample_data_dir / "audio"

    if "target_audio_path" not in st.session_state:
        st.session_state.target_audio_path = None

    a_btn1, a_btn2 = st.columns(2)
    with a_btn1:
        if st.button("📁 Load Authentic Audio Sample", use_container_width=True):
            st.session_state.target_audio_path = sample_aud_dir / "sample_real.wav"
    with a_btn2:
        if st.button("⚠️ Load Synthetic Audio Sample", use_container_width=True):
            st.session_state.target_audio_path = sample_aud_dir / "sample_fake.wav"

    uploaded_audio = st.file_uploader(
        "Upload audio recording (.wav, .mp3, .flac)",
        type=["wav", "mp3", "flac"]
    )

    if uploaded_audio is not None:
        temp_path = config.outputs_dir / f"upload_{uploaded_audio.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_audio.getbuffer())
        st.session_state.target_audio_path = temp_path

    target_audio_path = st.session_state.target_audio_path

    if target_audio_path and Path(target_audio_path).exists():
        st.audio(str(target_audio_path))
        st.markdown("---")

        if st.button("⚡ Execute Acoustic Forensic Audit", type="primary", use_container_width=True):
            with st.spinner("Computing 128-band log-Mel spectrogram and classifying spectral patterns..."):
                results = pipeline.analyze(target_audio_path)

            verdict = results["verdict"]
            confidence = results["confidence"]
            real_prob = results["real_probability"]
            fake_prob = results["fake_probability"]

            v_class = "verdict-fake" if verdict == "FAKE" else "verdict-real"
            t_class = "text-fake" if verdict == "FAKE" else "text-real"
            status_label = "SYNTHETIC SPEECH / CLONED VOICE" if verdict == "FAKE" else "NATURAL HUMAN SPEECH"

            st.markdown(f"""
            <div class="verdict-hero {v_class}">
                <div style="font-size: 12px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Acoustic Forensic Verdict</div>
                <div class="verdict-title {t_class}">{status_label}</div>
                <div style="font-size: 16px; color: #cbd5e1;">Acoustic Likelihood: <strong>{confidence * 100:.2f}%</strong></div>
                <div class="confidence-meter-container">
                    <div class="meter-fill-real" style="width: {real_prob * 100}%;"></div>
                    <div class="meter-fill-fake" style="width: {fake_prob * 100}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 12px; color: #94a3b8;">
                    <span>Authentic: {real_prob * 100:.1f}%</span>
                    <span>Synthetic: {fake_prob * 100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-chip">
                    <div class="metric-chip-label">Sample Rate</div>
                    <div class="metric-chip-value">{results['metadata']['sample_rate']} Hz</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-chip">
                    <div class="metric-chip-label">Duration</div>
                    <div class="metric-chip-value">{results['metadata']['raw_duration_sec']}s</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-chip">
                    <div class="metric-chip-label">Analysis Latency</div>
                    <div class="metric-chip-value">{results['processing_time_sec']:.3f}s</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.info(results["explanation"])

            st.markdown("### 📊 Log-Mel Spectrogram (Decibels)")
            st.image(results["spectrogram_image"], use_container_width=True)

            # Download Reports
            st.markdown("---")
            d_c1, d_c2 = st.columns(2)
            with d_c1:
                json_str = make_download_report(results, "audio")
                st.download_button(
                    "📥 Export Audio JSON Audit Log",
                    data=json_str,
                    file_name=f"audit_audio_{Path(target_audio_path).stem}.json",
                    mime="application/json",
                    use_container_width=True
                )
            with d_c2:
                cert_str = make_markdown_certificate(results, "audio")
                st.download_button(
                    "📄 Download Forensic Certificate",
                    data=cert_str,
                    file_name=f"certificate_audio_{Path(target_audio_path).stem}.md",
                    mime="text/markdown",
                    use_container_width=True
                )


# =====================================================================
# 5. AUDIO-VISUAL CONSISTENCY
# =====================================================================
elif navigation == "Audio-Visual Consistency":
    st.markdown("""
    <div class="saas-header">
        <div>
            <h1>Cross-Modal Consistency Engine</h1>
            <p>Dual-stream fusion detecting voice dubbing on authentic video or face-swaps on real audio</p>
        </div>
        <div class="telemetry-pill">
            <span class="status-dot-green"></span>
            <span>Multimodal Fusion Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    vid_pipeline = get_video_pipeline()
    aud_pipeline = get_audio_pipeline()
    verifier = get_multimodal_verifier()
    sample_vid_dir = config.sample_data_dir / "videos"

    if "target_av_path" not in st.session_state:
        st.session_state.target_av_path = None

    av_btn1, av_btn2 = st.columns(2)
    with av_btn1:
        if st.button("📁 Load Authentic Video Container", key="av_real", use_container_width=True):
            st.session_state.target_av_path = sample_vid_dir / "sample_real.mp4"
    with av_btn2:
        if st.button("⚠️ Load Spliced/Manipulated Video Container", key="av_fake", use_container_width=True):
            st.session_state.target_av_path = sample_vid_dir / "sample_fake.mp4"

    uploaded_av = st.file_uploader(
        "Upload container with audio track (.mp4, .mov, .avi)",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_av is not None:
        temp_video = config.outputs_dir / f"av_{uploaded_av.name}"
        with open(temp_video, "wb") as f:
            f.write(uploaded_av.getbuffer())
        st.session_state.target_av_path = temp_video

    target_av_path = st.session_state.target_av_path

    if target_av_path and Path(target_av_path).exists():
        st.video(str(target_av_path))

        if st.button("⚡ Analyze Cross-Modal Consistency", type="primary", use_container_width=True):
            with st.spinner("Performing dual-stream feature extraction (video frames & audio track)..."):
                # 1. Video analysis
                vid_res = vid_pipeline.analyze(target_av_path)

                # 2. Extract audio and analyze
                temp_wav = config.outputs_dir / f"extracted_{Path(target_av_path).stem}.wav"
                try:
                    aud_pipeline.preprocessor.extract_audio_from_video(target_av_path, temp_wav)
                    aud_res = aud_pipeline.analyze(temp_wav)
                    audio_available = True
                except Exception as e:
                    audio_available = False
                    st.warning(f"Audio stream extraction notice: {e}. Analyzing visual stream only.")

            if audio_available:
                eval_res = verifier.evaluate_consistency(
                    video_fake_prob=vid_res["fake_probability"],
                    audio_fake_prob=aud_res["fake_probability"]
                )

                st.markdown("---")
                v_class = "verdict-fake" if "FAKE" in eval_res["overall_verdict"] or "SUSPICIOUS" in eval_res["overall_verdict"] else "verdict-real"
                t_class = "text-fake" if "FAKE" in eval_res["overall_verdict"] or "SUSPICIOUS" in eval_res["overall_verdict"] else "text-real"

                st.markdown(f"""
                <div class="verdict-hero {v_class}">
                    <div style="font-size: 12px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Multimodal Forensic Determination</div>
                    <div class="verdict-title {t_class}">VERDICT: {eval_res['overall_verdict']}</div>
                    <div style="font-size: 16px; color: #cbd5e1;">Consistency Score: <strong>{eval_res['consistency_score'] * 100:.1f}%</strong> (Delta: {eval_res['inconsistency_delta']:.2f})</div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""
                    <div class="metric-chip">
                        <div class="metric-chip-label">Visual Fake Score</div>
                        <div class="metric-chip-value">{eval_res['video_fake_probability'] * 100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-chip">
                        <div class="metric-chip-label">Audio Fake Score</div>
                        <div class="metric-chip-value">{eval_res['audio_fake_probability'] * 100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="metric-chip">
                        <div class="metric-chip-label">Cross-Modal Congruence</div>
                        <div class="metric-chip-value">{eval_res['consistency_score'] * 100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                st.info(f"**Diagnostic Assessment:** {eval_res['diagnosis']}")
                st.caption(eval_res["limitations_notice"])


# =====================================================================
# 6. EXPLAINABILITY & XAI
# =====================================================================
elif navigation == "Explainability & XAI":
    st.markdown("""
    <div class="saas-header">
        <div>
            <h1>Explainable AI (XAI) Scientific Framework</h1>
            <p>Mathematical formulations and interpretability mechanisms across forensic models</p>
        </div>
        <div class="telemetry-pill">
            <span class="status-dot-green"></span>
            <span>Transparent Diagnostics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["Grad-CAM Saliency", "Acoustic Spectrograms", "Temporal Heuristics"])

    with t1:
        st.subheader("Grad-CAM: Gradient-Weighted Class Activation Mapping")
        st.markdown(r"""
        Grad-CAM computes the gradient of the unnormalized score for class $c$ ($y^c$) with respect to feature activation map 
        $A^k$ of the convolutional layer:
        $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{ij}^k}$$
        The neuron importance weights $\alpha_k^c$ perform a weighted linear combination of feature maps, followed by a ReLU operation:
        $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
        This identifies whether the classifier focuses on genuine facial boundary blending seams or spurious background cues.
        """)

    with t2:
        st.subheader("Mel-Spectrogram Acoustic Interpretation")
        st.markdown("""
        Human vocal tracts naturally generate smooth formant transitions and exponential harmonic decay.
        In contrast, neural vocoders (e.g. WaveGlow, HiFi-GAN) exhibit:
        - **Robotic harmonic frequency repetition** in the upper frequency bands.
        - **Phase discontinuities** across phonetic viseme transitions.
        - **High-frequency spectral buzz** visible in the 3 kHz – 8 kHz region of log-Mel spectrograms.
        """)

    with t3:
        st.subheader("Temporal Frame Trajectory & Top-k Aggregation")
        st.markdown(r"""
        Deepfake manipulation in videos often degrades intermittently when a subject rotates their head.
        Standard average pooling dilutes detection confidence. We implement **top-$k$ mean aggregation**:
        $$P_{\text{video}} = \frac{1}{k} \sum_{i=1}^k \text{sort\_descending}\left(\{p_1, p_2, \dots, p_N\}\right)_i$$
        where $k = \max(1, \lceil N \times 0.25 \rceil)$. This ensures localized temporal manipulations are robustly captured.
        """)


# =====================================================================
# 7. ENTERPRISE BENCHMARKS
# =====================================================================
elif navigation == "Enterprise Benchmarks":
    st.markdown("""
    <div class="saas-header">
        <div>
            <h1>Empirical Performance Dashboard</h1>
            <p>Measured test split performance versus portfolio target benchmarks</p>
        </div>
        <div class="telemetry-pill">
            <span class="status-dot-green"></span>
            <span>Audited Metrics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    report_path = config.reports_dir / "full_evaluation_report.json"
    if report_path.exists():
        report = load_json(report_path)

        st.caption(f"Last Evaluation: {report.get('timestamp', 'N/A')} | Compute: {report.get('hardware_device', 'CPU')}")

        st.subheader("Target Benchmark Comparison")
        comp_plot = config.plots_dir / "benchmark_comparison.png"
        if comp_plot.exists():
            st.image(str(comp_plot), use_container_width=True)

        st.markdown("---")
        st.subheader("Modality Performance Summary")
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.markdown("#### Image Classifier (Test Partition)")
            img_m = report.get("image_modality", {})
            st.write(f"- **Accuracy:** {img_m.get('accuracy', 0.0) * 100:.1f}%")
            st.write(f"- **F1-Score:** {img_m.get('f1', 0.0):.3f}")
            st.write(f"- **Precision:** {img_m.get('precision', 0.0):.3f}")
            st.write(f"- **Recall:** {img_m.get('recall', 0.0):.3f}")
            st.write(f"- **ROC-AUC:** {img_m.get('roc_auc', 0.0):.3f}")
            st.write(f"- **Inference Latency:** {img_m.get('avg_inference_latency_ms', 0.0)} ms/sample")

            cm_img = config.plots_dir / "confusion_matrix_image.png"
            if cm_img.exists():
                st.image(str(cm_img), caption="Image Confusion Matrix", width=340)

        with p_c2:
            st.markdown("#### Audio Classifier (Test Partition)")
            aud_m = report.get("audio_modality", {})
            st.write(f"- **Accuracy:** {aud_m.get('accuracy', 0.0) * 100:.1f}%")
            st.write(f"- **F1-Score:** {aud_m.get('f1', 0.0):.3f}")
            st.write(f"- **Precision:** {aud_m.get('precision', 0.0):.3f}")
            st.write(f"- **Recall:** {aud_m.get('recall', 0.0):.3f}")
            st.write(f"- **ROC-AUC:** {aud_m.get('roc_auc', 0.0):.3f}")
            st.write(f"- **Inference Latency:** {aud_m.get('avg_inference_latency_ms', 0.0)} ms/sample")

            roc_img = config.plots_dir / "roc_curve_image.png"
            if roc_img.exists():
                st.image(str(roc_img), caption="Audio ROC Curve", width=340)

        st.markdown("---")
        st.subheader("Video Processing Throughput")
        vb = report.get("video_benchmark", {})
        vb_c1, vb_c2, vb_c3, vb_c4 = st.columns(4)
        with vb_c1:
            st.metric("Video Duration", f"{vb.get('video_duration_sec', 0)}s")
        with vb_c2:
            st.metric("Sampled Frames", f"{vb.get('frames_sampled', 0)}")
        with vb_c3:
            st.metric("Total Latency", f"{vb.get('total_processing_time_sec', 0)}s")
        with vb_c4:
            st.metric("Throughput FPS", f"{vb.get('throughput_fps', 0)} FPS")
    else:
        st.warning("No evaluation report found. Execute `python evaluation/evaluate.py` to record live metrics.")


# =====================================================================
# 8. SYSTEM ARCHITECTURE
# =====================================================================
elif navigation == "System Architecture":
    st.markdown("""
    <div class="saas-header">
        <div>
            <h1>System Architecture & Academic Citations</h1>
            <p>Technology stack, mathematical foundation, and benchmark dataset citations</p>
        </div>
        <div class="telemetry-pill">
            <span class="status-dot-green"></span>
            <span>B.Tech AI/ML Capstone</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Engineering Standards
    1. **Identity Disjointness**: Training and validation/test partitions enforce strict subject separation to prevent optimistic data leakage.
    2. **Multi-Hardware Portability**: Automatic hardware acceleration across Apple Silicon MPS (`arm64`), NVIDIA CUDA, and standard CPU.
    3. **Responsible AI Disclosure**: Model confidence represents empirical posterior probabilities rather than guaranteed real-world truth.

    ### Benchmark Citations
    - **FaceForensics++**: Rössler et al., *Learning to Detect Manipulated Facial Images*, ICCV 2019.
    - **Celeb-DF**: Li et al., *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics*, CVPR 2020.
    - **ASVspoof**: Todisco et al., *ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection*, Interspeech 2019.
    """)
