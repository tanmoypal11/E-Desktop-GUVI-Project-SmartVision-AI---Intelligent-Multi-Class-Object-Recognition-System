import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="SmartVision AI | Object Detection",
    page_icon="🎯",
    layout="wide"
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🎯 Object Detection</h1>
    <p style="text-align:center; font-size:18px;">
    Multi-object detection using YOLO
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# LOAD YOLO MODEL (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_yolo_model():
    # Use pretrained YOLOv8 model
    model = YOLO("yolov8n.pt")  # nano → fastest & HF-friendly
    return model

model = load_yolo_model()

# -------------------------------------------------
# CONFIDENCE THRESHOLD
# -------------------------------------------------
conf_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# -------------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image for object detection",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.subheader("📷 Original Image")
    st.image(image, use_column_width=True)

    # -------------------------------------------------
    # YOLO INFERENCE
    # -------------------------------------------------
    with st.spinner("Running YOLO object detection..."):
        results = model.predict(
            source=image_np,
            conf=conf_threshold,
            save=False
        )

    # -------------------------------------------------
    # DRAW BOUNDING BOXES
    # -------------------------------------------------
    annotated_image = image_np.copy()

    detections = results[0].boxes

    detected_objects = []

    if detections is not None:
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            detected_objects.append((label, conf))

            # Draw rectangle
            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label text
            text = f"{label} {conf:.2f}"
            cv2.putText(
                annotated_image,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # -------------------------------------------------
    # DISPLAY RESULTS
    # -------------------------------------------------
    st.subheader("📦 Detection Results")
    st.image(annotated_image, use_column_width=True)

    if detected_objects:
        st.subheader("📝 Detected Objects")
        for label, conf in detected_objects:
            st.write(f"• **{label}** — Confidence: `{conf:.2f}`")

        st.success(f"Total objects detected: {len(detected_objects)}")
    else:
        st.warning("No objects detected with the selected confidence threshold.")

else:
    st.info("👆 Upload an image to start object detection.")
