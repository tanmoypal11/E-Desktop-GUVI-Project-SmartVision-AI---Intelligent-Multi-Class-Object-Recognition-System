import streamlit as st
import pandas as pd
from PIL import Image
import os

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Model Performance | SmartVision AI",
    layout="wide"
)

st.title("📊 Model Performance Dashboard")
st.write("Comprehensive evaluation of CNN models used in **SmartVision AI – Intelligent Multi-Class Object Recognition System**")

# =================================================
# FINAL METRICS TABLE
# =================================================
data = {
    "Model": [
        "EfficientNet-B0 (Fully Unlocked)",
        "EfficientNet-B0 (Partial)",
        "MobileNet",
        "ResNet18",
        "VGG16"
    ],
    "Accuracy (%)": [
        81.65,
        65.38,
        64.26,
        64.82,
        55.38
    ],
    "Precision (%)": [
        81.88,
        64.95,
        64.07,
        64.92,
        59.73
    ],
    "Recall (%)": [
        81.90,
        65.38,
        64.26,
        64.82,
        55.38
    ],
    "F1 Score (%)": [
        81.86,
        64.73,
        63.91,
        64.55,
        53.11
    ],
    "Inference Time": [
        "0.49 ms",
        "84.10 sec",
        "57.52 sec",
        "65.48 sec",
        "64.14 sec"
    ],
    "Model Size (MB)": [
        41.49,
        15.70,
        9.28,
        42.76,
        512.58
    ]
}

df = pd.DataFrame(data)

st.subheader("📌 Final Test Metrics Comparison")
st.dataframe(df, use_container_width=True)

# =================================================
# EfficientNet Fully Unlocked (BEST MODEL)
# =================================================
st.markdown("---")
st.subheader("🏆 EfficientNet-B0 (Fully Unlocked – Colab Trained)")

st.markdown("""
**📊 FINAL EVALUATION**
- Accuracy: **81.65%**
- Precision: **81.88%**
- Recall: **81.90%**
- F1 Score: **81.86%**
- Inference Time: **0.49 ms**
- Model Size: **41.49 MB**
""")

img = Image.open(
    "images/EfficientNet_Colab/Efficientnet_confusion_matrix_colab.png"
)
st.image(img, caption="Confusion Matrix – EfficientNet Fully Unlocked", use_column_width=True)

# =================================================
# EfficientNet Partial
# =================================================
st.markdown("---")
st.subheader("🔵 EfficientNet-B0 (Partially Frozen)")

col1, col2 = st.columns(2)

with col1:
    st.write("**Confusion Matrix**")
    img = Image.open("images/EfficientNet/EfficientNet_confusion_matrix.png")
    st.image(img, use_column_width=True)

with col2:
    st.write("**Loss & Accuracy Curve**")
    img = Image.open("images/EfficientNet/EfficientNet_loss_accuracy_curve.png")
    st.image(img, use_column_width=True)

# =================================================
# MobileNet
# =================================================
st.markdown("---")
st.subheader("🟢 MobileNet")

col1, col2 = st.columns(2)

with col1:
    st.write("**Confusion Matrix**")
    img = Image.open("images/MobileNet/MobileNet_confusion_matrix.png")
    st.image(img, use_column_width=True)

with col2:
    st.write("**Loss & Accuracy Curve**")
    img = Image.open("images/MobileNet/MobileNet_loss_accuracy.png")
    st.image(img, use_column_width=True)

# =================================================
# ResNet18
# =================================================
st.markdown("---")
st.subheader("🟣 ResNet18")

col1, col2 = st.columns(2)

with col1:
    st.write("**Confusion Matrix**")
    img = Image.open("images/Resnet18/Confusion_matrix_Resnet18.png")
    st.image(img, use_column_width=True)

with col2:
    st.write("**Loss & Accuracy Curve**")
    img = Image.open("images/Resnet18/loss_accuracy_Resnet18.png")
    st.image(img, use_column_width=True)

# =================================================
# VGG16
# =================================================
st.markdown("---")
st.subheader("🔴 VGG16")

col1, col2 = st.columns(2)

with col1:
    st.write("**Confusion Matrix**")
    img = Image.open("images/VGG16/VGG16_confusion_matrix.png")
    st.image(img, use_column_width=True)

with col2:
    st.write("**Loss & Accuracy Curve**")
    img = Image.open("images/VGG16/VGG16_accuracy_loss_curve.png")
    st.image(img, use_column_width=True)

# =================================================
# FINAL OBSERVATIONS
# =================================================
st.markdown("---")
st.subheader("📈 Final Observations")

st.markdown("""
- 🏆 **EfficientNet-B0 (Fully Unlocked)** is the **best performing model**, achieving **81.65% accuracy** with ultra-fast inference.
- 🔵 **Partially frozen EfficientNet** shows moderate performance but lower generalization.
- 🟢 **MobileNet** is optimal for **lightweight and edge deployment**.
- 🟣 **ResNet18** offers balanced accuracy with higher computational cost.
- 🔴 **VGG16**, despite large size, provides a classical baseline for CNN comparison.
""")
