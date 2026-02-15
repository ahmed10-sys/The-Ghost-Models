import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import segmentation_models_pytorch as smp

# 1. SETUP & PAGE CONFIG
st.set_page_config(page_title="Offroad AI: Obstacle Detection", layout="wide")
st.title("🚗 Offroad Terrain Segmentation")
st.sidebar.info("Upload an image to detect rocks, bushes, and terrain.")

# 2. DEFINE CLASSES & COLORS
class_names = [
    "Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Logs", "Rocks", "Landscape", "Sky"
]

# Unique RGB colors for visualization
colors = [
    [0, 0, 0],       # Background: Black
    [0, 255, 0],     # Trees: Green
    [255, 0, 255],   # Lush Bushes: Purple (High Visibility)
    [255, 255, 0],   # Dry Grass: Yellow
    [150, 75, 0],    # Dry Bushes: Brown
    [128, 128, 128], # Clutter: Gray
    [255, 165, 0],   # Logs: Orange
    [255, 0, 0],     # Rocks: Red (Alert!)
    [200, 200, 200], # Landscape: Light Gray
    [135, 206, 235]  # Sky: Blue
]

# 3. LOAD MODEL
@st.cache_resource
def load_model():
    # Use the same config as your PC/Colab training
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50", 
        encoder_weights=None, 
        in_channels=3, 
        classes=10
    )
    # Update this to point to your 'deeplab_best.pth' file
    model.load_state_dict(torch.load("deeplab_best.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# 4. IMAGE PROCESSING
def predict(image, model):
    H, W = image.shape[:2]
    # Resize to match training
    input_img = cv2.resize(image, (960, 544)) 
    input_img = input_img.transpose(2, 0, 1) / 255.0
    input_tensor = torch.tensor(input_img).float().unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output, dim=1).squeeze().numpy()
    
    # Resize back to original size
    mask = cv2.resize(mask.astype('uint8'), (W, H), interpolation=cv2.INTER_NEAREST)
    return mask

# 5. UI COMPONENTS
uploaded_file = st.sidebar.file_uploader("Choose an Offroad Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run Prediction
    mask = predict(image_rgb, model)
    
    # Create Overlay
    colored_mask = np.zeros_like(image_rgb)
    for i, color in enumerate(colors):
        colored_mask[mask == i] = color
    
    # Blend with original
    overlay = cv2.addWeighted(image_rgb, 0.6, colored_mask, 0.4, 0)
    
    # Display Results Side-by-Side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image_rgb, use_container_width=True)
    with col2:
        st.subheader("Segmentation Overlay")
        st.image(overlay, use_container_width=True)

    # Sidebar Legend
    st.sidebar.markdown("### 📊 Detected Objects")
    found_classes = np.unique(mask)
    for cls_id in found_classes:
        st.sidebar.markdown(f"🟢 **{class_names[cls_id]}**")
else:
    st.write("Please upload an image from the sidebar to begin analysis.")