"""Streamlit app: take or upload a photo and show the predicted lesion type.

Run from the project root:
    streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Make src/ importable no matter which directory streamlit is launched from
# (absolute path built from this file's location; works on Windows and Linux).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import config  # noqa: E402
from predict import Predictor, ensure_model_files  # noqa: E402

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🩺", layout="centered")


@st.cache_resource(show_spinner="Loading model (first start downloads it, ~110 MB)...")
def load_predictor() -> Predictor:
    """Load the model once per server process, on CPU.

    If models/best_model.pth or models/classes.json are missing (e.g. on
    Streamlit Community Cloud), they are downloaded from Hugging Face first.
    Because this function is cached, that happens only once.
    """
    model_path, classes_path = ensure_model_files()
    return Predictor(model_path, classes_path, device="cpu")


# One-line description per class + whether it warrants a doctor's visit soon.
CLASS_INFO = {
    "Melanocytic Nevus": ("A common mole. Usually harmless.", False),
    "Melanoma": ("A serious type of skin cancer.", True),
    "Benign Keratosis": ("A non-cancerous skin growth (e.g. seborrheic keratosis).", False),
    "Basal Cell Carcinoma": ("A common, slow-growing skin cancer.", True),
    "Actinic Keratosis": ("A rough, sun-damaged patch that can become cancerous.", True),
    "Vascular Lesion": ("A blood-vessel mark such as a cherry angioma. Usually harmless.", False),
    "Dermatofibroma": ("A small, firm, harmless skin bump.", False),
}

# Semi-transparent colors so the card looks right in both light and dark themes.
CARD_CSS = """
<style>
.result-card {
    margin: 1.5rem auto 0.5rem; padding: 1.8rem 1.5rem; max-width: 560px;
    text-align: center; border-radius: 18px; border: 1px solid;
}
.result-card.ok     { background: rgba(46, 160, 67, 0.10);  border-color: rgba(46, 160, 67, 0.45); }
.result-card.alert  { background: rgba(229, 72, 77, 0.10);  border-color: rgba(229, 72, 77, 0.50); }
.result-card.unsure { background: rgba(245, 166, 35, 0.10); border-color: rgba(245, 166, 35, 0.50); }
.result-label {
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; opacity: 0.7; margin-bottom: 0.4rem;
}
.result-name { font-size: 2.4rem; font-weight: 800; line-height: 1.15; }
.result-desc { font-size: 1.05rem; opacity: 0.85; margin-top: 0.6rem; }
.result-advice {
    display: inline-block; margin-top: 1rem; padding: 0.35rem 0.9rem;
    border-radius: 999px; font-size: 0.9rem; font-weight: 600;
    background: rgba(229, 72, 77, 0.18);
}
</style>
"""


def show_result(file):
    """Classify the image and show the result as a styled card."""
    with st.spinner("Analyzing..."):
        result = load_predictor().predict(Image.open(file))

    if result == config.UNSURE_MESSAGE:
        html = f"""
        <div class="result-card unsure">
            <div class="result-label">🤔 Couldn't recognise clearly</div>
            <div class="result-desc">{result}</div>
        </div>"""
    else:
        desc, serious = CLASS_INFO.get(result, ("", False))
        advice = ("<div class='result-advice'>⚠️ Please consult a dermatologist soon</div>"
                  if serious else "")
        html = f"""
        <div class="result-card {'alert' if serious else 'ok'}">
            <div class="result-label">🩺 Disease recognised</div>
            <div class="result-name">{result}</div>
            <div class="result-desc">{desc}</div>
            {advice}
        </div>"""

    # Collapse to a single line: Markdown treats indented lines as a code block,
    # which would print the raw HTML instead of rendering it.
    html = "".join(line.strip() for line in html.splitlines())
    st.markdown(CARD_CSS + html, unsafe_allow_html=True)


st.markdown("<h2 style='text-align:center;margin-bottom:0;'>Skin Lesion Classifier</h2>"
            "<p style='text-align:center;opacity:0.7;margin-top:0.3rem;'>"
            "Take or upload a clear, close-up photo of the skin spot.</p>",
            unsafe_allow_html=True)

# Always-visible disclaimer.
st.markdown(
    "<div style='text-align:center;font-size:0.85rem;padding:0.5rem 0.8rem;margin:0 auto 1rem;"
    "max-width:560px;border-radius:10px;background:rgba(245,166,35,0.12);"
    "border:1px solid rgba(245,166,35,0.45);'>"
    "⚠️ <b>Not a medical diagnosis tool.</b> This is an AI demo for educational use. "
    "Always see a doctor about any skin concern.</div>",
    unsafe_allow_html=True,
)

# Load (and if needed download) the model up front so the first prediction is fast.
try:
    load_predictor()
except Exception as exc:  # e.g. no internet / Hugging Face unavailable
    st.error(f"Could not load the model: {exc}")
    st.stop()

tab_cam, tab_upload = st.tabs(["Take Photo", "Upload Image"])
with tab_cam:
    photo = st.camera_input("Take a close-up photo of the lesion")
    if photo is not None:
        show_result(photo)
with tab_upload:
    upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if upload is not None:
        show_result(upload)
