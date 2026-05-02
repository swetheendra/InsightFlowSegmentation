import streamlit as st
import os
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from UNET import UNET

# --- 1. MODEL LOADING (CACHED) ---
@st.cache_resource
def load_trained_model(weights_path):
    # Initialize model with the architecture used during training
    model = UNET(in_channels=3, out_channels=1, layers=[16,32,64, 128, 256, 512])
    # Load weights to CPU to ensure it works on any hosting platform
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

# --- 2. INFERENCE HELPER ---
def run_inference(image, model, threshold=0.5):
    # Prepare image: Resize -> Tensor (Normalization optional based on your training)
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    
    input_tensor = transform(image).unsqueeze(0) # Add batch dimension

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    binary_mask = (probs > threshold).astype(np.uint8)
    return probs, binary_mask

# --- 3. UI SETUP ---
st.set_page_config(page_title="InsightFlow: Player Seg", layout="wide")
st.title("🏃‍♂️ InsightFlow: Sports Semantic Intelligence")

# Load model once
try:
    model = load_trained_model("weights/football_unet.pth")
except Exception as e:
    st.error(f"Could not load model weights. Check the path! Error: {e}")
    st.stop()

# --- 4. SIDEBAR SELECTION ---
DEMO_DIR = "demoimages"
st.sidebar.title("🎮 Try a Demo")

# Create directory if missing
if not os.path.exists(DEMO_DIR):
    os.makedirs(DEMO_DIR)

demo_files = [f for f in os.listdir(DEMO_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

if demo_files:
    st.sidebar.write("Click a sample to test:")
    cols = st.sidebar.columns(2)
    for idx, file in enumerate(demo_files):
        img_path = os.path.join(DEMO_DIR, file)
        if cols[idx % 2].button(f"Sample {idx+1}", key=file):
            st.session_state.selected_img = img_path
            st.session_state.input_mode = "demo"

st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader("Or upload your own image:", type=["jpg", "png"])

if uploaded_file:
    st.session_state.selected_img = uploaded_file
    st.session_state.input_mode = "upload"

# Control threshold
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# --- 5. EXECUTION ---
final_image = None
if "selected_img" in st.session_state:
    final_image = Image.open(st.session_state.selected_img).convert("RGB")

if final_image:
    # RUN UNET CALL
    with st.spinner("AI is analyzing the pitch..."):
        probs, mask = run_inference(final_image, model, threshold)

    # PROCESS OUTPUTS
    # Resize mask back to original image size for better quality
    mask_resized = Image.fromarray(mask * 255).resize(final_image.size, resample=Image.NEAREST)
    mask_np = np.array(mask_resized) / 255.0
    
    # Apply mask to original image
    img_np = np.array(final_image)
    cutout = (img_np * mask_np[:, :, np.newaxis]).astype(np.uint8)

    # DISPLAY RESULTS
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(final_image, use_container_width=True)
    with col2:
        st.subheader("Isolated Players")
        st.image(cutout, use_container_width=True)
        
    st.download_button("Download Mask", data=np.array(mask_resized).tobytes(), file_name="mask.png")
else:
    st.info("Select a demo image or upload a file to start.")