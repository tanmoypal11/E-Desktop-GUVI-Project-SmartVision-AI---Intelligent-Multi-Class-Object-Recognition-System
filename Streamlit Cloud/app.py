import streamlit as st

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="SmartVision AI - Home",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🤖 SmartVision AI</h1>
    <h3 style="text-align:center;">
    Intelligent Multi-Class Object Recognition System
    </h3>
    <p style="text-align:center; font-size:18px;">
    Deep Learning based Image Classification & Object Detection Platform
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# PROJECT OVERVIEW
# -------------------------------------------------
st.header("📌 Project Overview")

st.markdown(
    """
**SmartVision AI** is an end-to-end **Computer Vision application** that performs  
**multi-class image classification (26 classes)** and **multi-object detection**
using state-of-the-art **Deep Learning models**.

The system is trained on a **curated 26-class subset of the COCO dataset** and
deployed as an **interactive Streamlit web application** suitable for
real-world and production-level use cases.
"""
)

# -------------------------------------------------
# PROBLEM & SOLUTION
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("❗ Problem Statement")
    st.markdown(
        """
Object detection and classification are critical in modern AI systems,
yet real-world images often contain:

- Multiple objects in a single frame
- Variations in lighting, scale, and background
- Performance constraints for real-time applications
- Scalability challenges in deployment
"""
    )

with col2:
    st.subheader("✅ Proposed Solution")
    st.markdown(
        """
SmartVision AI addresses these challenges by combining:

- **Transfer Learning-based CNN models** for image classification
- **YOLO-based object detection** for real-time inference
- **Optimized GPU/CPU inference pipelines**
- **Cloud-ready deployment** using Streamlit
"""
    )

# -------------------------------------------------
# KEY FEATURES
# -------------------------------------------------
st.header("🚀 Key Features")

st.markdown(
    """
✔ Image classification across **26 object categories**  
✔ Multi-object detection with **bounding boxes & confidence scores**  
✔ Performance comparison of **multiple CNN architectures**  
✔ High-speed inference suitable for real-time usage  
✔ Clean, intuitive multi-page web interface  
✔ Scalable and production-ready design
"""
)

# -------------------------------------------------
# MODELS USED
# -------------------------------------------------
st.header("🧠 Models Used")

col3, col4 = st.columns(2)

with col3:
    st.subheader("📷 Image Classification Models")
    st.markdown(
        """
- **VGG16**
- **ResNet18**
- **MobileNet**
- **EfficientNet-B3 (Fully Unlocked – Best Model)**
"""
    )

with col4:
    st.subheader("🎯 Object Detection Model")
    st.markdown(
        """
- **YOLO (Ultralytics)**
- Fine-tuned on **26 COCO classes**
- Supports real-time multi-object detection
"""
    )

# -------------------------------------------------
# DATASET
# -------------------------------------------------
st.header("📊 Dataset Information")

st.markdown(
    """
- **Dataset:** COCO (26-Class Subset)
- **Image Type:** Real-world RGB images
- **Annotations:** COCO JSON & YOLO format
- **Class Distribution:** Balanced across all selected classes
"""
)

# -------------------------------------------------
# USE CASES
# -------------------------------------------------
st.header("🏭 Business Use Cases")

st.markdown(
    """
- Smart Cities & Traffic Monitoring  
- Retail & E-Commerce Analytics  
- Security & Surveillance Systems  
- Wildlife Conservation  
- Healthcare Monitoring  
- Smart Homes & IoT  
- Agriculture & Livestock Monitoring  
- Logistics & Warehousing
"""
)

# -------------------------------------------------
# HOW TO USE
# -------------------------------------------------
st.header("🧭 How to Use the Application")

st.markdown(
    """
1️⃣ Navigate to the **Image Classification** page to classify uploaded images  
2️⃣ Use the **Object Detection** page to detect multiple objects in one image  
3️⃣ Analyze model predictions and confidence scores  
4️⃣ Explore technical and project details in the **About** section
"""
)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()

st.markdown(
    """
<p style="text-align:center; font-size:14px;">
Built with Python • PyTorch • YOLO • Streamlit  
<br>
SmartVision AI – Capstone Project
</p>
""",
    unsafe_allow_html=True
)
