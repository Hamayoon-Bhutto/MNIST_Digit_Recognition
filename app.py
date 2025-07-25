import os
import io
import time
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import Dict, Any, Tuple, Optional

# Third‑party Streamlit component for drawing
try:
    from streamlit_drawable_canvas import st_canvas  # type: ignore
    _HAS_CANVAS = True
except Exception:  # pragma: no cover - safe fallback
    _HAS_CANVAS = False

# ------------------------------------------------------------------
# Streamlit Page Config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide",
)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.title("MNIST Digit Recognition")
    st.markdown("Draw a digit (0–9) on the canvas and select a model to classify it.")

    model_choice = st.radio("Select Model", ["SGD Classifier", "Random Forest"], index=0)

    st.markdown("---")
    st.markdown("**About**")
    st.markdown("This demo uses scikit-learn models trained on MNIST 28×28 grayscale digits.")
    st.markdown("Built with ❤️ using Streamlit.")

# ------------------------------------------------------------------
# Model + Metrics Loading
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models_and_metrics() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray]]:
    """Load models and optional metrics/confusion matrices from disk.

    Returns
    -------
    models: dict
        {model_name: estimator or None}
    metrics: dict
        {model_name: {"accuracy": float, "precision": float, "recall": float, "f1": float}}
    cms: dict
        {model_name: np.ndarray shape (10,10)}
    """
    models: Dict[str, Any] = {"SGD Classifier": None, "Random Forest": None}
    metrics_out: Dict[str, Any] = {}
    cms_out: Dict[str, np.ndarray] = {}

    model_dir = "models"
    if not os.path.isdir(model_dir):
        st.warning("`models/` directory not found. Upload models to enable predictions.")
        return models, metrics_out, cms_out

    # Load SGD model
    sgd_path = os.path.join(model_dir, "sgd_mnist.pkl")
    if os.path.isfile(sgd_path):
        try:
            models["SGD Classifier"] = joblib.load(sgd_path)
        except Exception as e:  # pragma: no cover
            st.error(f"Failed to load SGD model: {e}")

    # Load RF model
    rf_path = os.path.join(model_dir, "rf_mnist.pkl")
    if os.path.isfile(rf_path):
        try:
            models["Random Forest"] = joblib.load(rf_path)
        except Exception as e:  # pragma: no cover
            st.error(f"Failed to load Random Forest model: {e}")

    # Load metrics artifact (optional)
    metrics_path = os.path.join(model_dir, "metrics.pkl")
    if os.path.isfile(metrics_path):
        try:
            artifact = joblib.load(metrics_path)
            # expected structure described in header comment
            if isinstance(artifact, dict):
                for mname in ["SGD Classifier", "Random Forest"]:
                    if mname in artifact:
                        metrics_out[mname] = artifact[mname]
                # confusion matrices
                cm_art = artifact.get("cm") or {}
                for mname in ["SGD Classifier", "Random Forest"]:
                    if isinstance(cm_art, dict) and mname in cm_art:
                        cms_out[mname] = np.array(cm_art[mname])
        except Exception as e:  # pragma: no cover
            st.warning(f"Could not load metrics artifact: {e}")

    return models, metrics_out, cms_out


MODELS, METRICS, CMS = load_models_and_metrics()


# ------------------------------------------------------------------
# Image Preprocessing Utilities
# ------------------------------------------------------------------

def _to_grayscale(img_arr: np.ndarray) -> Image.Image:
    """Convert RGBA/RGB/Gray array to grayscale PIL Image (L)."""
    if img_arr.dtype != np.uint8:
        img_arr = img_arr.astype("uint8")
    if img_arr.ndim == 2:  # already grayscale
        pil_img = Image.fromarray(img_arr, mode="L")
    elif img_arr.shape[2] == 4:
        pil_img = Image.fromarray(img_arr, mode="RGBA").convert("L")
    else:
        pil_img = Image.fromarray(img_arr, mode="RGB").convert("L")
    return pil_img


def _crop_to_content(img: Image.Image, threshold: int = 10) -> Image.Image:
    """Crop image to bounding box of pixels brighter than threshold.

    Parameters
    ----------
    img : PIL Image (L mode)
    threshold : int
        Pixel > threshold considered part of digit.
    """
    arr = np.array(img)
    mask = arr > threshold
    if not mask.any():
        return img  # nothing drawn; return as-is
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # slice end
    cropped = img.crop((x0, y0, x1, y1))
    return cropped


def _resize_and_center(cropped: Image.Image, final_size: int = 28, digit_box: int = 20) -> Image.Image:
    """Resize cropped digit to digit_box (max dimension) and center in a final_size square.

    This mimics the classic MNIST centering heuristic.
    """
    # Maintain aspect ratio
    w, h = cropped.size
    scale = float(digit_box) / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cropped.resize((new_w, new_h), Image.LANCZOS)

    # Paste into 28x28 canvas
    canvas = Image.new("L", (final_size, final_size), color=0)
    left = (final_size - new_w) // 2
    top = (final_size - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def preprocess_canvas_image(img_arr: np.ndarray, invert: bool = False) -> np.ndarray:
    """Full preprocessing pipeline -> flattened normalized 784 float array.

    Steps:
    1. Convert to grayscale.
    2. Crop to drawn content.
    3. Resize & center to 28×28 (digit scaled to 20px max dim).
    4. (Optional) invert colors.
    5. Normalize to [0,1].
    6. Flatten to shape (1,784).
    """
    pil_img = _to_grayscale(img_arr)
    cropped = _crop_to_content(pil_img)
    centered = _resize_and_center(cropped)
    arr = np.array(centered).astype("float32")

    if invert:
        arr = 255.0 - arr

    arr /= 255.0
    return arr.reshape(1, -1)


# ------------------------------------------------------------------
# Prediction Helper
# ------------------------------------------------------------------

def predict_digit(model_name: str, X: np.ndarray) -> Tuple[Optional[int], Optional[np.ndarray]]:
    model = MODELS.get(model_name)
    if model is None:
        return None, None
    try:
        pred = model.predict(X)
        # Not all sklearn estimators support predict_proba; guard
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
        else:
            # fallback: one-hot at predicted class
            proba = np.zeros(10, dtype=float)
            proba[int(pred[0])] = 1.0
        return int(pred[0]), proba
    except Exception as e:  # pragma: no cover
        st.error(f"Prediction error: {e}")
        return None, None


# ------------------------------------------------------------------
# Metrics Display Helper
# ------------------------------------------------------------------

def show_model_metrics(model_name: str) -> None:
    st.subheader("Performance Metrics")
    m = METRICS.get(model_name)
    if m is None:
        # Fallback heuristics
        if model_name == "SGD Classifier":
            m = {"accuracy": 0.95, "precision": 0.94, "recall": 0.95, "f1": 0.95}
        else:
            m = {"accuracy": 0.97, "precision": 0.96, "recall": 0.97, "f1": 0.96}
        st.info("Showing placeholder metrics. Provide a metrics.pkl artifact for real scores.")

    st.metric("Accuracy", f"{m.get('accuracy', float('nan')):.2f}")
    st.metric("Precision", f"{m.get('precision', float('nan')):.2f}")
    st.metric("Recall", f"{m.get('recall', float('nan')):.2f}")
    st.metric("F1 Score", f"{m.get('f1', float('nan')):.2f}")


def show_confusion_matrix(model_name: str) -> None:
    st.subheader("Confusion Matrix")
    cm = CMS.get(model_name)
    if cm is None:
        # fallback synthetic random matrix
        rng = np.random.default_rng(seed=42)
        cm = rng.integers(0, 100, size=(10, 10))
        st.info("Showing synthetic confusion matrix. Provide real evaluation in metrics.pkl artifact.")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)


# ------------------------------------------------------------------
# Example Digits Gallery (optional random noise fallback)
# ------------------------------------------------------------------

def show_example_digits(n: int = 5) -> None:
    st.subheader("Example Digits")
    cols = st.columns(n)
    rng = np.random.default_rng(seed=0)
    for i in range(n):
        img = rng.random((28, 28))
        cols[i].image(img, width=50, clamp=True)


# ------------------------------------------------------------------
# Main App Layout
# ------------------------------------------------------------------

def main():
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Draw (or Upload) a Digit")
        canvas_size = 280

        use_invert = st.checkbox("Invert Colors (if your digit appears black on white)", value=False)

        # Drawing canvas if available ------------------------------------------------
        if _HAS_CANVAS:
            canvas_result = st_canvas(
                fill_color="rgba(0,0,0,0)",  # transparent fill
                stroke_width=20,
                stroke_color="#FFFFFF",
                background_color="#000000",
                width=canvas_size,
                height=canvas_size,
                drawing_mode="freedraw",
                key="drawable_canvas",
            )
            img_data = canvas_result.image_data
        else:
            st.warning("`streamlit-drawable-canvas` not installed. Using upload fallback.")
            uploaded_file = st.file_uploader("Upload a 28×28 (or larger) digit image", type=["png", "jpg", "jpeg"])
            img_data = None
            if uploaded_file is not None:
                pil = Image.open(uploaded_file).convert("RGBA")
                img_data = np.array(pil)

        # If user drew/uploaded ------------------------------------------------------
        processed_arr = None
        if img_data is not None:
            processed_arr = preprocess_canvas_image(img_data, invert=use_invert)
            # show processed 28x28 image
            st.caption("Processed (centered) 28×28 image")
            st.image(processed_arr.reshape(28, 28), width=100, clamp=True)

        # Predict button -------------------------------------------------------------
        if st.button("Predict Digit", type="primary", disabled=(processed_arr is None)):
            if processed_arr is None:
                st.error("Please draw or upload an image first.")
            else:
                with st.spinner('Predicting...'):
                    pred, proba = predict_digit(model_choice, processed_arr)
                if pred is None:
                    st.error("Model not loaded. Please add model file(s) under models/.")
                else:
                    st.success(f"Predicted Digit: **{pred}**")
                    if proba is not None:
                        st.subheader("Confidence Scores")
                        fig, ax = plt.subplots()
                        ax.bar(range(10), proba)
                        ax.set_xticks(range(10))
                        ax.set_xlabel('Digit')
                        ax.set_ylabel('Probability')
                        ax.set_ylim([0, 1])
                        st.pyplot(fig)

    # ------------------------------------------------------------------
    # Right column: metrics, confusion matrix, examples
    # ------------------------------------------------------------------
    with col2:
        st.header("Model Information")
        show_model_metrics(model_choice)
        show_confusion_matrix(model_name=model_choice)
        show_example_digits()


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
