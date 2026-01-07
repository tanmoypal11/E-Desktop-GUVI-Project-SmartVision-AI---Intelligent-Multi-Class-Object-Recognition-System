import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Image Classification | SmartVision AI",
    layout="wide"
)

st.title("🖼️ Image Classification")
st.write("Upload an image to classify using **EfficientNet-B3 (Fully Unlocked)**")

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"**Running on:** {device}")

# -------------------------------------------------
# Class Names (26 Classes - FIXED ORDER)
# -------------------------------------------------
CLASS_NAMES = [
    "airplane", "bed", "bench", "bicycle", "bird", "bottle", "bowl",
    "bus", "cake", "car", "cat", "chair", "couch", "cow", "cup",
    "dog", "elephant", "horse", "motorcycle", "person", "pizza",
    "potted plant", "stop sign", "traffic light", "train", "truck"
]

NUM_CLASSES = 26

# -------------------------------------------------
# Load EfficientNet-B3 Model
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b3(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    # ✅ CORRECT PATH (same folder as Streamlit app)
    model.load_state_dict(
        torch.load(
            "best_efficientnet_b3.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()
    return model

model = load_model()

# -------------------------------------------------
# Image Transforms (EfficientNet-B3 expects 300×300)
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------
# Image Upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("📷 Uploaded Image")
    st.image(image, use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    # -------------------------------------------------
    # Inference
    # -------------------------------------------------
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probabilities).item()

    # -------------------------------------------------
    # Prediction Output
    # -------------------------------------------------
    st.subheader("✅ Prediction Result")
    st.success(f"**Predicted Class:** {CLASS_NAMES[pred_idx]}")
    st.write(f"**Confidence:** {probabilities[pred_idx] * 100:.2f}%")

    # -------------------------------------------------
    # Top-5 Predictions
    # -------------------------------------------------
    st.subheader("🔍 Top-5 Predictions")

    top5 = torch.topk(probabilities, 5)

    for rank in range(5):
        idx = top5.indices[rank].item()
        conf = top5.values[rank].item() * 100
        st.write(f"**{rank + 1}. {CLASS_NAMES[idx]}** — {conf:.2f}%")
