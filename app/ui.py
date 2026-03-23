import sys
import os

# ---------------------------------------------------------------------------
# Path setup -- make sure the app/ directory is on sys.path so that our
# sub-packages (models, utils) can be imported no matter which directory
# Streamlit is launched from.
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import streamlit as st
from PIL import Image
import torch
import pickle as pkl
import numpy as np

from config import MODEL_REGISTRY, VOCAB_PATH, DEFAULT_BEAM_SIZE, DEFAULT_MAX_LEN
from utils.transform import inference_transform, visual_transform
from utils.vocab import Vocab

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LaTeX OCR",
    page_icon="\u03a3",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1222 0%, #131b2e 50%, #0f1729 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #c8d6e5 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label {
        font-weight: 600 !important;
        letter-spacing: 0.02em;
    }

    /* Main header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.75rem 0;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #faf5ff 100%);
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        margin-top: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    /* Model info badge */
    .model-badge {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.2rem;
        letter-spacing: 0.02em;
    }
    .badge-baseline { background: #ecfeff; color: #0e7490; border: 1px solid #a5f3fc; }
    .badge-enhanced { background: #f5f3ff; color: #7c3aed; border: 1px solid #c4b5fd; }
    .badge-transformer { background: #fdf2f8; color: #be185d; border: 1px solid #f9a8d4; }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper: load model + producer ────────────────────────────────────────────
@st.cache_resource
def load_producer(model_key: str, beam_size: int, max_len: int):
    """Instantiate the correct model, load weights, and wrap in a LatexProducer."""
    entry = MODEL_REGISTRY[model_key]

    # Load vocab
    with open(VOCAB_PATH, "rb") as f:
        vocab = pkl.load(f)
    vocab_size = len(vocab)

    # Build model
    model = entry["build_fn"](vocab_size)

    # Load weights
    weight_path = entry["weight"]
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))

    # Producer
    ProducerClass = entry["producer_cls"]
    use_cuda = torch.cuda.is_available()
    producer = ProducerClass(
        model, vocab, max_len=max_len, use_cuda=use_cuda, beam_size=beam_size
    )
    return producer, entry


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")

    # Model selection grouped by category
    model_names = list(MODEL_REGISTRY.keys())
    selected_model = st.selectbox(
        "Model",
        model_names,
        index=0,
        help="Select the encoder-decoder architecture to use for prediction.",
    )

    entry_info = MODEL_REGISTRY[selected_model]
    cat = entry_info["category"]
    badge_cls = {
        "Baseline": "badge-baseline",
        "Enhanced": "badge-enhanced",
        "Transformer": "badge-transformer",
    }[cat]
    st.markdown(
        f'<span class="model-badge {badge_cls}">{cat}</span>',
        unsafe_allow_html=True,
    )
    st.caption(entry_info["description"])

    st.markdown("---")
    beam_size = st.slider("Beam Size", 1, 10, DEFAULT_BEAM_SIZE)
    max_len = st.number_input(
        "Max Sequence Length", value=DEFAULT_MAX_LEN, min_value=50, max_value=500
    )

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image of a mathematical formula.",
    )


# ── Main content ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>LaTeX OCR</h1>
        <p>Convert images of mathematical formulas into LaTeX code</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_left, col_right = st.columns(2, gap="large")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with col_left:
        st.markdown("#### Input Image")
        st.image(image, use_container_width=True)

    # Predict button
    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner("Running inference..."):
            producer, entry = load_producer(selected_model, beam_size, max_len)

            img_np = np.array(image)
            img_dict = inference_transform(image=img_np)
            img_tensor = img_dict["image"].unsqueeze(0)  # [1, C, H, W]

            # Inference
            output = producer(img_tensor)

            # Some producers return (results, alphas), others just results
            has_attention = entry.get("has_attention", False)
            if has_attention and isinstance(output, tuple) and len(output) == 2:
                result, alphas = output
            else:
                result = output[0] if isinstance(output, tuple) else output
                alphas = None

            st.session_state["latex_code"] = result[0]
            st.session_state["alphas"] = alphas
            st.session_state["image_np"] = img_np
            st.session_state["img_dict"] = img_dict
            st.session_state["has_attention"] = has_attention

    # ── Display results ──────────────────────────────────────────────────
    if "latex_code" in st.session_state:
        with col_right:
            st.markdown("#### LaTeX Output")
            st.markdown('<div class="result-card">', unsafe_allow_html=True)

            edited_latex = st.text_area(
                "Edit LaTeX code below:",
                value=st.session_state["latex_code"],
                height=120,
                label_visibility="collapsed",
            )

            st.code(edited_latex, language="latex")

            st.markdown("**Rendered Preview**")
            try:
                st.latex(edited_latex)
            except Exception as e:
                st.error(f"Render error: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Attention visualization (for LSTM-based models) ──────────────
        if (
            st.session_state.get("has_attention")
            and st.session_state.get("alphas") is not None
        ):
            with st.expander("Attention Visualization", expanded=False):
                from utils.visualize import visualize_attention

                alphas = st.session_state["alphas"]
                img_np = st.session_state["image_np"]
                img_dict = st.session_state["img_dict"]
                latex_code = st.session_state["latex_code"]

                _, H_feat, W_feat = img_dict["image"].shape
                H_prime, W_prime = H_feat // 8, W_feat // 8

                vis_img = visual_transform(image=img_np)["image"]

                fig = visualize_attention(
                    vis_img,
                    latex_code.split(),
                    alphas[0],
                    H_prime,
                    W_prime,
                )
                if fig is not None:
                    st.pyplot(fig)

else:
    # Empty state
    with col_left:
        st.markdown("#### Input Image")
        st.info("Upload an image from the sidebar to get started.")
    with col_right:
        st.markdown("#### LaTeX Output")
        st.info("Prediction results will appear here.")
