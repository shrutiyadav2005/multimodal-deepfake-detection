"""
Streamlit Web Dashboard for Multimodal Deepfake Detection & Explainable Content Verification.
Modern dark-themed interface supporting Image, Video, Audio, and Multimodal forensics.
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
    page_title="Multimodal Deepfake Detection & XAI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Dark Theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #f3f4f6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        background-color: #1a1f2c;
        border-radius: 6px;
        color: #9ca3af;
        border: 1px solid #2d3748;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        font-weight: 600;
    }
    .metric-card {
        background: #1e2433;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-real {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: bold;
        font-size: 14px;
        display: inline-block;
    }
    .badge-fake {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: bold;
        font-size: 14px;
        display: inline-block;
    }
    .badge-warning {
        background-color: #78350f;
        color: #fbbf24;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: bold;
        font-size: 14px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# Cached pipeline loaders
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


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("Deepfake Forensics")
st.sidebar.caption("Multimodal & Explainable AI Engine")

hardware_name = get_device_name(get_device())
st.sidebar.info(f"⚡ **Compute Hardware:**\n{hardware_name}")

navigation = st.sidebar.radio(
    "Navigation",
    [
        "Home & Overview",
        "Image Verification",
        "Video Verification",
        "Audio Verification",
        "Audio-Visual Consistency",
        "Explainability & XAI",
        "Model Performance",
        "About & Architecture",
    ]
)


# Helper: Downloadable Report
def make_download_report(results: dict, modality: str) -> str:
    report_dict = {
        "report_type": f"{modality.upper()} Forensics Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "hardware_device": hardware_name,
        "results": {k: v for k, v in results.items() if not isinstance(v, (np.ndarray, Image.Image))}
    }
    return json.dumps(report_dict, indent=4)


# =====================================================================
# 1. HOME & OVERVIEW
# =====================================================================
if navigation == "Home & Overview":
    st.title("🛡️ Multimodal Deepfake Detection & Explainable Content Verification")
    st.markdown("""
    Welcome to the **Multimodal Deepfake Forensics Platform**. This system provides automated integrity 
    verification and explainable artificial intelligence (XAI) diagnostics across **Images, Videos, and Audio**, 
    along with **Cross-Modal Audio-Visual Consistency Analysis**.
    """)

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🖼️ Image Forensics</h4>
            <p>ResNet18 CNN backbone trained to detect synthetic facial manipulation with Grad-CAM saliency heatmaps.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🎬 Video Forensics</h4>
            <p>Intelligent uniform frame sampling with temporal aggregation heuristics and suspicious frame timelines.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>🎙️ Audio Forensics</h4>
            <p>Spectrogram CNN extracting log-Mel features to detect voice cloning, synthetic TTS, and vocoder artifacts.</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>🔗 Multimodal Fusion</h4>
            <p>Evaluates cross-modal consistency between visual and audio streams to detect dubbing and face swaps.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("System Architecture")
    st.markdown("""
    ```mermaid
    flowchart LR
        Input[Uploaded Content] --> Router{Media Modality}
        
        Router -->|Image| ImgPrep[Face Detection & Crop] --> ResNet[ResNet18 Classifier] --> GradCAM[Grad-CAM Saliency]
        Router -->|Video| VidPrep[Uniform Frame Sampling] --> FrameInfer[Frame-Level CNN] --> Aggregation[Temporal Aggregation]
        Router -->|Audio| AudPrep[Log-Mel Spectrogram] --> SpecCNN[Spectrogram CNN] --> AudExplain[Spectral Saliency]
        
        Aggregation --> FusionEngine[Audio-Visual Consistency Engine]
        SpecCNN --> FusionEngine
        
        GradCAM --> Dashboard[Verification Dashboard & Report]
        Aggregation --> Dashboard
        AudExplain --> Dashboard
        FusionEngine --> Dashboard
    ```
    """)


# =====================================================================
# 2. IMAGE VERIFICATION
# =====================================================================
elif navigation == "Image Verification":
    st.title("🖼️ Image Deepfake Verification")
    st.markdown("Analyze uploaded portraits and images for synthetic generation, facial warping, or blending seams.")

    pipeline = get_image_pipeline()
    sample_img_dir = config.sample_data_dir / "images"

    col_btn1, col_btn2 = st.columns(2)
    selected_sample = None
    with col_btn1:
        if st.button("📁 Load Sample Authentic Image"):
            selected_sample = sample_img_dir / "sample_real.jpg"
    with col_btn2:
        if st.button("⚠️ Load Sample Manipulated Image"):
            selected_sample = sample_img_dir / "sample_fake.jpg"

    uploaded_file = st.file_uploader(
        "Choose an image file (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"]
    )

    target_image_path = None
    if uploaded_file is not None:
        temp_path = config.outputs_dir / f"upload_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        target_image_path = temp_path
    elif selected_sample and selected_sample.exists():
        target_image_path = selected_sample

    if target_image_path:
        st.markdown("---")
        crop_face = st.checkbox("Enable Automatic Facial Landmark Cropping", value=True)

        if st.button("🚀 Run Image Forensic Verification", type="primary"):
            with st.spinner("Analyzing image features and computing Grad-CAM saliency..."):
                results = pipeline.analyze(target_image_path, crop_face=crop_face)

            verdict = results["verdict"]
            confidence = results["confidence"]
            fake_prob = results["fake_probability"]

            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                badge = "badge-fake" if verdict == "FAKE" else "badge-real"
                st.markdown(f'<div class="{badge}">{verdict} ({confidence * 100:.1f}%)</div>', unsafe_allow_html=True)
                st.markdown(f"**Fake Probability:** {fake_prob * 100:.2f}%")
                st.markdown(f"**Real Probability:** {(1.0 - fake_prob) * 100:.2f}%")
                st.markdown(f"**Latency:** {results['processing_time_sec']:.3f} s")
                st.markdown(f"**Hardware:** {results['hardware_device']}")

                # Download report
                rep_json = make_download_report(results, "image")
                st.download_button(
                    "📥 Download JSON Forensic Report",
                    data=rep_json,
                    file_name="image_verification_report.json",
                    mime="application/json"
                )

            with res_col2:
                st.info(results["explanation"])

            st.markdown("### Visual Explainability (Grad-CAM Saliency)")
            img_c1, img_c2, img_c3 = st.columns(3)
            with img_c1:
                st.caption("1. Preprocessed Input Image")
                st.image(results["original_image"], use_container_width=True)
            with img_c2:
                st.caption("2. Raw Activation Heatmap")
                st.image(results["gradcam_heatmap"], clamp=True, use_container_width=True)
            with img_c3:
                st.caption("3. Grad-CAM Blended Overlay")
                st.image(results["gradcam_overlay"], use_container_width=True)


# =====================================================================
# 3. VIDEO VERIFICATION
# =====================================================================
elif navigation == "Video Verification":
    st.title("🎬 Video Deepfake Verification")
    st.markdown("Inspect video sequences for facial manipulation, temporal flickering, and spliced synthetic frames.")

    pipeline = get_video_pipeline()
    sample_vid_dir = config.sample_data_dir / "videos"

    col_btn1, col_btn2 = st.columns(2)
    selected_sample = None
    with col_btn1:
        if st.button("📁 Load Sample Authentic Video"):
            selected_sample = sample_vid_dir / "sample_real.mp4"
    with col_btn2:
        if st.button("⚠️ Load Sample Manipulated Video"):
            selected_sample = sample_vid_dir / "sample_fake.mp4"

    uploaded_video = st.file_uploader(
        "Choose a video file (.mp4, .mov, .avi)",
        type=["mp4", "mov", "avi"]
    )

    target_video_path = None
    if uploaded_video is not None:
        temp_path = config.outputs_dir / f"upload_{uploaded_video.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        target_video_path = temp_path
    elif selected_sample and selected_sample.exists():
        target_video_path = selected_sample

    if target_video_path:
        st.video(str(target_video_path))
        st.markdown("---")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            agg_method = st.selectbox(
                "Temporal Aggregation Method",
                ["top_k_mean", "mean", "max"],
                help="top_k_mean averages the top 25% most suspicious frames."
            )
        with col_opt2:
            crop_face = st.checkbox("Facial Region Isolation", value=True)

        if st.button("🚀 Run Video Forensic Analysis", type="primary"):
            pipeline.video_detector.aggregation_method = agg_method
            with st.spinner("Extracting frames, evaluating CNN features, and aggregating timeline..."):
                results = pipeline.analyze(target_video_path, crop_face=crop_face)

            verdict = results["verdict"]
            confidence = results["confidence"]

            v_col1, v_col2 = st.columns([1, 2])
            with v_col1:
                badge = "badge-fake" if verdict == "FAKE" else "badge-real"
                st.markdown(f'<div class="{badge}">{verdict} ({confidence * 100:.1f}%)</div>', unsafe_allow_html=True)
                st.markdown(f"**Duration:** {results['metadata']['duration_sec']} s")
                st.markdown(f"**Frames Sampled:** {len(results['frame_probabilities'])}")
                st.markdown(f"**Processing Time:** {results['total_processing_time_sec']:.2f} s ({results['throughput_fps']} FPS)")

                rep_json = make_download_report(results, "video")
                st.download_button(
                    "📥 Download Video Forensic Report",
                    data=rep_json,
                    file_name="video_verification_report.json",
                    mime="application/json"
                )

            with v_col2:
                st.info(results["explanation"])

            st.markdown("### Temporal Suspicion Timeline")
            st.image(results["timeline_image"], use_container_width=True)

            st.markdown("### Peak Suspicious Frame & Grad-CAM Overlay")
            pk1, pk2 = st.columns(2)
            with pk1:
                st.caption(f"Peak Frame #{results['peak_frame_index']} (Time: {results['peak_frame_timestamp']}s)")
                st.image(results["peak_raw_frame"], use_container_width=True)
            with pk2:
                st.caption("Grad-CAM Saliency Overlay")
                st.image(results["peak_cam_overlay"], use_container_width=True)


# =====================================================================
# 4. AUDIO VERIFICATION
# =====================================================================
elif navigation == "Audio Verification":
    st.title("🎙️ Audio Deepfake & Spoofing Verification")
    st.markdown("Detect synthetic speech, voice cloning, and text-to-speech (TTS) vocoder artifacts from audio files.")

    pipeline = get_audio_pipeline()
    sample_aud_dir = config.sample_data_dir / "audio"

    col_btn1, col_btn2 = st.columns(2)
    selected_sample = None
    with col_btn1:
        if st.button("📁 Load Sample Authentic Audio"):
            selected_sample = sample_aud_dir / "sample_real.wav"
    with col_btn2:
        if st.button("⚠️ Load Sample Synthetic Audio"):
            selected_sample = sample_aud_dir / "sample_fake.wav"

    uploaded_audio = st.file_uploader(
        "Choose an audio file (.wav, .mp3, .flac)",
        type=["wav", "mp3", "flac"]
    )

    target_audio_path = None
    if uploaded_audio is not None:
        temp_path = config.outputs_dir / f"upload_{uploaded_audio.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_audio.getbuffer())
        target_audio_path = temp_path
    elif selected_sample and selected_sample.exists():
        target_audio_path = selected_sample

    if target_audio_path:
        st.audio(str(target_audio_path))
        st.markdown("---")

        if st.button("🚀 Run Audio Forensic Analysis", type="primary"):
            with st.spinner("Extracting log-Mel spectrogram and classifying acoustic patterns..."):
                results = pipeline.analyze(target_audio_path)

            verdict = results["verdict"]
            confidence = results["confidence"]

            a_col1, a_col2 = st.columns([1, 2])
            with a_col1:
                badge = "badge-fake" if verdict == "FAKE" else "badge-real"
                st.markdown(f'<div class="{badge}">{verdict} ({confidence * 100:.1f}%)</div>', unsafe_allow_html=True)
                st.markdown(f"**Duration:** {results['metadata']['raw_duration_sec']} s")
                st.markdown(f"**Sample Rate:** {results['metadata']['sample_rate']} Hz")
                st.markdown(f"**Processing Time:** {results['processing_time_sec']:.3f} s")

                rep_json = make_download_report(results, "audio")
                st.download_button(
                    "📥 Download Audio Forensic Report",
                    data=rep_json,
                    file_name="audio_verification_report.json",
                    mime="application/json"
                )

            with a_col2:
                st.info(results["explanation"])

            st.markdown("### Acoustic Log-Mel Spectrogram")
            st.image(results["spectrogram_image"], use_container_width=True)


# =====================================================================
# 5. AUDIO-VISUAL CONSISTENCY
# =====================================================================
elif navigation == "Audio-Visual Consistency":
    st.title("🔗 Audio-Visual Consistency Analysis")
    st.markdown("""
    Evaluates independent visual and acoustic integrity streams to detect cross-modal inconsistencies,
    such as **voice dubbing over real video** or **face-swapping paired with authentic audio**.
    """)

    vid_pipeline = get_video_pipeline()
    aud_pipeline = get_audio_pipeline()
    verifier = get_multimodal_verifier()

    uploaded_av = st.file_uploader(
        "Choose a video file (.mp4, .mov, .avi) with an audio track",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_av is not None:
        temp_video = config.outputs_dir / f"av_{uploaded_av.name}"
        with open(temp_video, "wb") as f:
            f.write(uploaded_av.getbuffer())

        st.video(str(temp_video))

        if st.button("🚀 Analyze Cross-Modal Consistency", type="primary"):
            with st.spinner("Extracting visual frames, extracting audio track, and evaluating congruence..."):
                # 1. Video analysis
                vid_res = vid_pipeline.analyze(temp_video)

                # 2. Extract audio and analyze
                temp_wav = config.outputs_dir / f"extracted_{Path(temp_video).stem}.wav"
                try:
                    aud_pipeline.preprocessor.extract_audio_from_video(temp_video, temp_wav)
                    aud_res = aud_pipeline.analyze(temp_wav)
                    audio_available = True
                except Exception as e:
                    audio_available = False
                    st.warning(f"Audio stream extraction warning: {e}. Analyzing video-only.")

            if audio_available:
                eval_res = verifier.evaluate_consistency(
                    video_fake_prob=vid_res["fake_probability"],
                    audio_fake_prob=aud_res["fake_probability"]
                )

                st.markdown("---")
                st.subheader("Multimodal Integrity Assessment")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Visual Fake Probability", f"{eval_res['video_fake_probability'] * 100:.1f}%")
                with c2:
                    st.metric("Audio Fake Probability", f"{eval_res['audio_fake_probability'] * 100:.1f}%")
                with c3:
                    st.metric("Consistency Score", f"{eval_res['consistency_score'] * 100:.1f}%")

                verdict_badge = "badge-fake" if "FAKE" in eval_res["overall_verdict"] or "SUSPICIOUS" in eval_res["overall_verdict"] else "badge-real"
                st.markdown(f'<div class="{verdict_badge}">Verdict: {eval_res["overall_verdict"]}</div>', unsafe_allow_html=True)

                st.markdown(f"**Diagnostic Analysis:** {eval_res['diagnosis']}")
                st.caption(eval_res["limitations_notice"])


# =====================================================================
# 6. EXPLAINABILITY & XAI
# =====================================================================
elif navigation == "Explainability & XAI":
    st.title("🔍 Explainable AI (XAI) in Forensic Verification")
    st.markdown("""
    Explainability is a core pillar of trustworthy AI. In forensic deepfake detection, black-box decisions 
    are insufficient for legal, journalistic, or security contexts. This system provides interpretability across all modalities:
    """)

    t1, t2, t3 = st.tabs(["Grad-CAM Saliency", "Acoustic Spectrograms", "Temporal Video Timelines"])

    with t1:
        st.subheader("Grad-CAM: Gradient-Weighted Class Activation Mapping")
        st.markdown(r"""
        Grad-CAM calculates the gradients of the target class score $y^c$ with respect to feature activation maps $A^k$
        of the final convolutional layer:
        $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{ij}^k}$$
        The neuron importance weights $\alpha_k^c$ scale the feature maps, followed by a ReLU non-linearity:
        $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
        This reveals whether the classifier is genuinely focusing on boundary blending seams, facial warping artifacts, 
        or background noise.
        """)

    with t2:
        st.subheader("Mel-Spectrogram Acoustic Interpretation")
        st.markdown("""
        Human speech naturally exhibits smooth formant envelopes and continuous harmonic decay.
        In contrast, synthetic speech generated by vocoders (e.g. WaveGlow, HiFi-GAN) often introduces:
        - **Robotic harmonic frequency repetition** in higher frequency bands.
        - **Phase discontinuities** at phonetic boundaries.
        - **High-frequency spectral buzz** visible in the 3 kHz – 8 kHz region of log-Mel spectrograms.
        """)

    with t3:
        st.subheader("Temporal Frame Trajectory")
        st.markdown("""
        Deepfakes frequently suffer from temporal coherence degradation (flickering, sudden boundary shifts 
        when the subject rotates their head). By uniformly sampling frames and plotting per-frame suspicion scores, 
        investigators can pinpoint the exact sub-second timestamps where manipulation occurs.
        """)


# =====================================================================
# 7. MODEL PERFORMANCE
# =====================================================================
elif navigation == "Model Performance":
    st.title("📊 Empirical Performance Dashboard")
    st.markdown("Real measured performance metrics on benchmark test sets compared against portfolio targets.")

    report_path = config.reports_dir / "full_evaluation_report.json"
    if report_path.exists():
        report = load_json(report_path)

        st.info(f"Report Generated: {report.get('timestamp', 'N/A')} | Hardware: {report.get('hardware_device', 'CPU')}")

        st.subheader("Target Benchmark Comparison")
        comp_plot = config.plots_dir / "benchmark_comparison.png"
        if comp_plot.exists():
            st.image(str(comp_plot), use_container_width=True)

        st.subheader("Modality Performance Summaries")
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.markdown("#### Image Classifier (Test Split)")
            img_m = report.get("image_modality", {})
            st.write(f"- **Accuracy:** {img_m.get('accuracy', 0.0) * 100:.1f}%")
            st.write(f"- **F1-Score:** {img_m.get('f1', 0.0):.3f}")
            st.write(f"- **Precision:** {img_m.get('precision', 0.0):.3f}")
            st.write(f"- **Recall:** {img_m.get('recall', 0.0):.3f}")
            st.write(f"- **ROC-AUC:** {img_m.get('roc_auc', 0.0):.3f}")
            st.write(f"- **Inference Latency:** {img_m.get('avg_inference_latency_ms', 0.0)} ms/sample")

            cm_img = config.plots_dir / "confusion_matrix_image.png"
            if cm_img.exists():
                st.image(str(cm_img), caption="Confusion Matrix", width=320)

        with p_c2:
            st.markdown("#### Audio Classifier (Test Split)")
            aud_m = report.get("audio_modality", {})
            st.write(f"- **Accuracy:** {aud_m.get('accuracy', 0.0) * 100:.1f}%")
            st.write(f"- **F1-Score:** {aud_m.get('f1', 0.0):.3f}")
            st.write(f"- **Precision:** {aud_m.get('precision', 0.0):.3f}")
            st.write(f"- **Recall:** {aud_m.get('recall', 0.0):.3f}")
            st.write(f"- **ROC-AUC:** {aud_m.get('roc_auc', 0.0):.3f}")
            st.write(f"- **Inference Latency:** {aud_m.get('avg_inference_latency_ms', 0.0)} ms/sample")

            roc_img = config.plots_dir / "roc_curve_image.png"
            if roc_img.exists():
                st.image(str(roc_img), caption="ROC Curve", width=320)

        st.markdown("---")
        st.subheader("Video Throughput Benchmark")
        vb = report.get("video_benchmark", {})
        vb_c1, vb_c2, vb_c3, vb_c4 = st.columns(4)
        with vb_c1:
            st.metric("Video Duration", f"{vb.get('video_duration_sec', 0)} s")
        with vb_c2:
            st.metric("Sampled Frames", f"{vb.get('frames_sampled', 0)}")
        with vb_c3:
            st.metric("Total Latency", f"{vb.get('total_processing_time_sec', 0)} s")
        with vb_c4:
            st.metric("Throughput FPS", f"{vb.get('throughput_fps', 0)} FPS")

    else:
        st.warning("No evaluation report found. Run `python evaluation/evaluate.py` to generate live metrics.")


# =====================================================================
# 8. ABOUT & ARCHITECTURE
# =====================================================================
elif navigation == "About & Architecture":
    st.title("ℹ️ About the Project & Engineering Details")
    st.markdown("""
    ### Project Background
    This project was engineered as a high-standard **B.Tech CSE (AI & ML) Capstone Portfolio System** 
    demonstrating genuine proficiency in deep learning, computer vision, acoustic signal processing, 
    multimodal machine learning, and explainable AI (XAI).

    ### Key Engineering Design Decisions
    1. **Identity-Disjoint Splitting**: Preprocessing and dataset generators enforce strict identity separation 
       between training and validation/test splits to eliminate optimistic data leakage.
    2. **Graceful Fallbacks & Portability**: Automatically handles hardware switching between Apple Silicon (MPS), 
       NVIDIA CUDA, and standard CPU without code alteration.
    3. **Honest Confidence & Limitations**: Does not misrepresent neural network softmax confidence as absolute real-world 
       authenticity, and explicitly clarifies the distinction between cross-modal consistency and phoneme-level lip-synchrony.

    ### Technology Stack
    - **Deep Learning Framework**: PyTorch 2.x, Torchvision, Torchaudio
    - **Vision & Video**: OpenCV, PIL
    - **Audio Processing**: Librosa, SoundFile
    - **Metrics & Science**: Scikit-learn, NumPy, Pandas, SciPy
    - **Explainability**: Custom Grad-CAM, Matplotlib, Seaborn
    - **Web Framework**: Streamlit
    """)
