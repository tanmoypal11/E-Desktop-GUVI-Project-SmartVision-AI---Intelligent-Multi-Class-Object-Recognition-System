import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(
    page_title="SmartVision AI | Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<h1 style="text-align:center;">🎯 Object Detection</h1>
<p style="text-align:center;">
YOLO-based Object Detection (Demo View)
</p>
""", unsafe_allow_html=True)

st.divider()

st.warning(
    "⚠️ Live YOLO inference is disabled on Streamlit Cloud "
    "due to PyTorch compatibility limits (Python 3.13)."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width='stretch')

    draw = ImageDraw.Draw(image)
    draw.rectangle([50, 50, 300, 300], outline="green", width=3)
    draw.text((55, 35), "person (0.92)", fill="green")

    st.subheader("📦 Detection Result (Sample)")
    st.image(image, width='stretch')

    st.info("Sample bounding box shown for UI demonstration.")
