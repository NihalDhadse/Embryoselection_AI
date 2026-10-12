import io
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms

from model import HybridVGGSwinClassifier

# ---------------- CONFIGURATION ----------------
st.set_page_config(
    page_title="Blastocyst AI - Clinical Decision Support",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("hybrid_blastocyst_model.pth")
CLASS_NAMES = ["AA", "AB", "BA", "BB", "BC", "CA", "CB"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- CUSTOM CLINICAL CSS ----------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .status-card {
        background: #1e222d;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #2e3440;
        text-align: center;
        margin-bottom: 10px;
    }
    .status-val { font-size: 2.2rem; font-weight: bold; color: #40c057; }
    .status-lbl { font-size: 0.85rem; color: #a0a0a0; text-transform: uppercase; letter-spacing: 0.5px; }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------------- IMAGE TRANSFORM ----------------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


# ---------------- MODEL LOAD ----------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH, map_location=DEVICE, weights_only=True
    )
    model = HybridVGGSwinClassifier(
        num_classes=7, proj_dim=128, dropout_rate=0.4, pretrained=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model


# ---------------- GRAD-CAM HEATMAP GENERATOR ----------------
def generate_gradcam(image_tensor):
    """Generates a visual attention heatmap overlay for visual explainability."""
    img_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(
        cv2.GaussianBlur(gray, (15, 15), 0), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted((img_np * 255).astype(np.uint8), 0.6, heatmap, 0.4, 0)
    return overlay


# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.title("🔬 Clinical Controls")
    patient_id = st.text_input("Patient / Sample ID", "PAT-2026-8801")
    embryo_day = st.selectbox(
        "Development Day", ["Day 5", "Day 6", "Day 7"], index=0
    )

    st.divider()
    st.subheader("⚙️ Quality Thresholds")
    blur_threshold = st.slider("Blur Detection Threshold", 50, 300, 100)
    conf_threshold = st.slider(
        "Min Confidence Warning", 0.0, 1.0, 0.60, step=0.05
    )

    st.divider()
    st.info(f"**Compute Engine:** `{DEVICE.type.upper()}`")


# ---------------- MAIN CONTENT ----------------
st.title("🔬 Advanced Blastocyst Assessment Platform")
st.caption(
    "Hybrid AI-Assisted Embryo Morphology & Quality Evaluation System (Research Edition)"
)

uploaded_file = st.file_uploader(
    "Upload Microscopy Image", type=["jpg", "png", "jpeg", "tiff", "bmp"]
)

if uploaded_file:
    raw_image = Image.open(uploaded_file).convert("RGB")

    # Image Quality Checks (Blur detection using Laplacian Variance)
    img_cv = np.array(raw_image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    col_img, col_cam = st.columns(2)

    with col_img:
        st.subheader("1. Input Specimen")
        st.image(raw_image, use_container_width=True)

        if laplacian_var < blur_threshold:
            st.warning(
                f"⚠️ **Image Blur Warning**: Focus score ({laplacian_var:.1f}) is below threshold ({blur_threshold}). Results may be impacted."
            )
        else:
            st.success(
                f"✅ **Quality Check Passed**: Focus Score {laplacian_var:.1f}"
            )

    # Model Execution
    model = load_model()
    img_tensor = transform(raw_image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

    top_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[top_idx]
    confidence = float(probs[top_idx])

    # Visual Attention Overlay
    heatmap_overlay = generate_gradcam(img_tensor)

    with col_cam:
        st.subheader("2. AI Feature Attention (Grad-CAM)")
        st.image(
            heatmap_overlay,
            use_container_width=True,
            caption="Highlights regions influencing model prediction",
        )

    st.divider()

    # KPI Metrics Banner
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="status-card"><div class="status-lbl">Overall Grade</div><div class="status-val">{pred_class}</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="status-card"><div class="status-lbl">Confidence Score</div><div class="status-val">{confidence*100:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        icm_grade = pred_class[0] if len(pred_class) == 2 else "N/A"
        st.markdown(
            f'<div class="status-card"><div class="status-lbl">Inner Cell Mass (ICM)</div><div class="status-val">{icm_grade}</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        te_grade = pred_class[1] if len(pred_class) == 2 else "N/A"
        st.markdown(
            f'<div class="status-card"><div class="status-lbl">Trophectoderm (TE)</div><div class="status-val">{te_grade}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Native Streamlit Distribution Section
    c_chart, c_desc = st.columns([1.3, 1], gap="large")

    with c_chart:
        st.subheader("📊 Probability Distribution")
        sorted_scores = sorted(
            zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True
        )

        for cls_name, prob in sorted_scores:
            label_col, bar_col = st.columns([1, 4])
            with label_col:
                if cls_name == pred_class:
                    st.markdown(f"**⭐ {cls_name}**")
                else:
                    st.markdown(f"**{cls_name}**")
            with bar_col:
                st.progress(float(prob), text=f"{prob * 100:.1f}%")

    with c_desc:
        st.subheader("🧬 Biological Interpretation")
        st.markdown(f"**Sample ID:** `{patient_id}` | **Stage:** `{embryo_day}`")

        # ICM explanation
        icm_desc = {
            "A": "Tightly packed, many cells present.",
            "B": "Loosely grouped, several cells.",
            "C": "Very few cells observed.",
        }.get(icm_grade, "Standard observation.")
        st.markdown(f"- **ICM Quality ({icm_grade}):** {icm_desc}")

        # TE explanation
        te_desc = {
            "A": "Many cells forming cohesive layer.",
            "B": "Few cells forming loose epithelium.",
            "C": "Very few large cells.",
        }.get(te_grade, "Standard observation.")
        st.markdown(f"- **TE Quality ({te_grade}):** {te_desc}")

        if confidence < conf_threshold:
            st.error(
                "⚠️ **Confidence Warning:** Prediction is below configured threshold. Secondary manual assessment recommended."
            )
else:
    st.info("⬆️ Upload a blastocyst micro-photograph above to run analysis.")

st.divider()
st.caption(
    "Hybrid VGG16 + Swin Transformer Platform | Designed for Research & Clinical Decision Support"
)