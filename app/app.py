"""
Retail Product Recognition System — Streamlit app.

Run with:
    streamlit run app/app.py

Three tabs:
  1. Single Product Scan   — upload one image, get top-3 predictions
                              with a confidence check.
  2. Checkout Simulation    — upload several product images at once and
                              get an itemized mock receipt, the way a
                              self-checkout camera would.
  3. Model Info             — shows the confusion matrix / training
                              curves produced by src/evaluate.py, so
                              the app doubles as a small model report.
"""

import os
import sys
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

st.set_page_config(page_title="Retail Product Recognition System", layout="wide")


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(config.MODEL_PATH)
    with open(config.CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
    array = np.array(image).astype("float32") / 255.0
    return np.expand_dims(array, axis=0)


def predict(model, class_names, image: Image.Image):
    batch = preprocess(image)
    probabilities = model.predict(batch, verbose=0)[0]
    order = np.argsort(probabilities)[::-1]
    top3 = [(class_names[i], float(probabilities[i])) for i in order[:3]]
    return top3


st.title("Retail Product Recognition System")
st.caption(
    "CNN-based product recognition, demonstrated on a fruit catalog — "
    "the same pipeline generalizes to any retail shelf category."
)

model, class_names = load_model_and_classes()

tab_scan, tab_checkout, tab_info = st.tabs(
    ["Single Product Scan", "Checkout Simulation", "Model Info"]
)

# --------------------------------------------------------------------
# Tab 1 — single scan with top-3 + confidence check
# --------------------------------------------------------------------
with tab_scan:
    st.subheader("Scan a single product")
    uploaded_file = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], key="single"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            top3 = predict(model, class_names, image)
            best_label, best_conf = top3[0]

            if best_conf < config.CONFIDENCE_THRESHOLD:
                st.warning(
                    f"Low confidence ({best_conf * 100:.1f}%) — this item "
                    "may not be in the current catalog. Best guess below."
                )
            else:
                st.success(f"Recognized: {best_label}")

            st.metric("Confidence", f"{best_conf * 100:.2f}%")
            price = config.PRODUCT_PRICES.get(best_label)
            if price is not None:
                st.metric("Shelf Price", f"{price:.2f} EGP")

            st.write("Top-3 candidates:")
            for label, conf in top3:
                st.write(f"- {label}: {conf * 100:.1f}%")

            fig, ax = plt.subplots()
            labels = [label for label, _ in top3]
            values = [conf for _, conf in top3]
            ax.barh(labels, values)
            ax.set_xlabel("Probability")
            ax.set_xlim(0, 1)
            st.pyplot(fig)

# --------------------------------------------------------------------
# Tab 2 — checkout simulation: multiple images -> itemized receipt
# --------------------------------------------------------------------
with tab_checkout:
    st.subheader("Simulate a self-checkout basket")
    st.caption(
        "Upload photos of several items as if they were scanned one by "
        "one at checkout. The app recognizes each one and totals the bill."
    )

    uploaded_files = st.file_uploader(
        "Choose product images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="basket",
    )

    if uploaded_files:
        receipt = []
        cols = st.columns(min(4, len(uploaded_files)))

        for idx, file in enumerate(uploaded_files):
            image = Image.open(file)
            top3 = predict(model, class_names, image)
            label, confidence = top3[0]
            price = config.PRODUCT_PRICES.get(label, 0.0)
            receipt.append({"item": label, "confidence": confidence, "price": price})

            with cols[idx % len(cols)]:
                st.image(image, use_container_width=True)
                st.caption(f"{label}\n{confidence * 100:.0f}%")

        st.divider()
        st.subheader("Receipt")
        total = 0.0
        for row in receipt:
            flag = "" if row["confidence"] >= config.CONFIDENCE_THRESHOLD else " (unsure)"
            st.write(f"{row['item']}{flag} — {row['price']:.2f} EGP")
            total += row["price"]

        st.metric("Total", f"{total:.2f} EGP")

# --------------------------------------------------------------------
# Tab 3 — model report (confusion matrix / training curves, if present)
# --------------------------------------------------------------------
with tab_info:
    st.subheader("Model report")
    st.write("Catalog classes:", ", ".join(class_names))

    if os.path.exists(config.CONFUSION_MATRIX_PATH):
        st.image(config.CONFUSION_MATRIX_PATH, caption="Confusion Matrix")
    else:
        st.info("Run `python -m src.evaluate` to generate the confusion matrix.")

    if os.path.exists(config.TRAINING_CURVES_PATH):
        st.image(config.TRAINING_CURVES_PATH, caption="Training Curves")
    else:
        st.info("Run `python -m src.evaluate` to generate training curves.")

    if os.path.exists(config.CLASSIFICATION_REPORT_PATH):
        with open(config.CLASSIFICATION_REPORT_PATH) as f:
            st.text(f.read())
