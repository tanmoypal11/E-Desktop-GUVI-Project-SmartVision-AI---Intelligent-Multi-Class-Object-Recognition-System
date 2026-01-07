import streamlit as st

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="SmartVision AI | About",
    page_icon="ℹ️",
    layout="wide"
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">ℹ️ About SmartVision AI</h1>
    <p style="text-align:center; font-size:18px;">
    Intelligent Multi-Class Object Recognition System
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# PROJECT DESCRIPTION
# -------------------------------------------------
st.header("📌 Project Description")

st.markdown(
    """
**SmartVision AI** is an end-to-end **Computer Vision & Artificial Intelligence**
application designed to demonstrate the practical use of deep learning models
for real-world visual understanding tasks, including:

- **Multi-class image classification (26 classes)**
- **Real-time multi-object detection**
- **Interactive AI deployment using Streamlit**

The system leverages **transfer learning-based CNN architectures**
and a **YOLO-based object detection pipeline**, trained on a curated
subset of the **COCO dataset**.
"""
)

# -------------------------------------------------
# DATASET DETAILS
# -------------------------------------------------
st.header("📊 Dataset Information")

st.markdown(
    """
- **Dataset Source:** COCO (Common Objects in Context)  
- **Task Types:** Image Classification & Object Detection  
- **Number of Classes:** **26**  
- **Image Format:** RGB images  
- **Annotations:** COCO JSON & YOLO format  
- **Class Distribution:** Balanced across all selected classes  

The dataset consists of real-world images containing multiple objects,
varying illumination, occlusions, and complex backgrounds, making it
well-suited for benchmarking deep learning models.
"""
)

# -------------------------------------------------
# SELECTED CLASSES
# -------------------------------------------------
st.header("🏷️ Selected Object Classes (26)")

st.markdown(
    """
**🚗 Vehicles (7):**  
airplane, car, truck, bus, motorcycle, bicycle, train  

**👤 Human (1):**  
person  

**🚦 Outdoor Objects (3):**  
traffic light, stop sign, bench  

**🐾 Animals (6):**  
dog, cat, horse, bird, cow, elephant  

**🍽️ Kitchen & Food (6):**  
bottle, cup, bowl, pizza, cake  

**🪑 Furniture & Indoor (3):**  
chair, couch, bed, potted plant  

**Total Classes:** **26**
"""
)

# -------------------------------------------------
# MODELS & ARCHITECTURES
# -------------------------------------------------
st.header("🧠 Models & Architectures")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Image Classification Models")
    st.markdown(
        """
- **VGG16** – Deep CNN baseline for comparison  
- **ResNet18** – Residual learning for stable performance  
- **MobileNet** – Lightweight model for fast inference  
- **EfficientNet-B3 (Fully Unlocked)** –  
  Best-performing model with full fine-tuning across all layers  

All models are trained using **transfer learning** with ImageNet
initialization and fine-tuned on the project dataset.
"""
    )

with col2:
    st.subheader("🎯 Object Detection Model")
    st.markdown(
        """
- **YOLO (Ultralytics)**  
- Single-stage object detection architecture  
- Supports **real-time multi-object detection**  
- Outputs bounding boxes, class labels, and confidence scores  
"""
    )

# -------------------------------------------------
# TECHNOLOGY STACK
# -------------------------------------------------
st.header("🛠️ Technology Stack")

st.markdown(
    """
- **Programming Language:** Python  
- **Deep Learning Framework:** PyTorch  
- **Computer Vision:** OpenCV  
- **Object Detection:** YOLO (Ultralytics)  
- **Web Interface:** Streamlit  
- **Model Training:** Local GPU & Google Colab  
- **Deployment Ready:** Streamlit-based application  
"""
)

# -------------------------------------------------
# PROJECT OBJECTIVES
# -------------------------------------------------
st.header("🎯 Project Objectives")

st.markdown(
    """
- Build a robust **26-class object recognition system**
- Apply **transfer learning and full model fine-tuning**
- Compare CNN architectures based on performance and efficiency
- Develop a **user-friendly AI web application**
- Demonstrate real-world AI deployment capabilities
"""
)

# -------------------------------------------------
# EVALUATION & PERFORMANCE
# -------------------------------------------------
st.header("📈 Model Performance Highlights")

st.markdown(
    """
**Best Model – EfficientNet-B3 (Fully Unlocked):**
- **Accuracy:** 81.65%
- **Precision:** 81.88%
- **Recall:** 81.90%
- **F1 Score:** 81.86%
- **Inference Time:** ~0.49 ms
- **Model Size:** 41.49 MB  

Other models (MobileNet, ResNet18, VGG16) were evaluated
to analyze trade-offs between speed, size, and accuracy.
"""
)

# -------------------------------------------------
# DEPLOYMENT
# -------------------------------------------------
st.header("☁️ Deployment")

st.markdown(
    """
The application is deployed using **Streamlit**, providing:

- Interactive browser-based inference
- Easy image upload and visualization
- Modular multi-page AI dashboard
- Ready for cloud deployment (Hugging Face / local server)
"""
)

# -------------------------------------------------
# DEVELOPER INFO
# -------------------------------------------------
st.header("👨‍💻 Developer")

st.markdown(
    """
**Project Name:** SmartVision AI  
**Domain:** Computer Vision & Artificial Intelligence  
**Project Type:** Capstone / Final Project  

Developed as part of an advanced AI and deep learning curriculum,
following industry-aligned practices in model development,
evaluation, and deployment.
"""
)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()

st.markdown(
    """
<p style="text-align:center; font-size:14px;">
© 2026 SmartVision AI | Built with Python, PyTorch & Streamlit
</p>
""",
    unsafe_allow_html=True
)
