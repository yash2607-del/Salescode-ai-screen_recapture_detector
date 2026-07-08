import streamlit as st
import tensorflow as tf
from PIL import Image
import os
import tempfile
import time

# Set TF logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from predict import preprocess_image, predict_single
from src import config

st.set_page_config(page_title="Screen Recapture Detector", layout="centered")
st.title(" Screen Recapture Detector")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(str(config.CHECKPOINT_PATH))

model = load_model()

st.sidebar.header("Input Source")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
camera_file = st.sidebar.camera_input("Or take a live photo")

target_file = uploaded_file if uploaded_file is not None else camera_file

if target_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    tfile.write(target_file.read())
    tfile.close()
    
    st.image(target_file, caption="Input Image", use_container_width=True)
    
    start = time.time()
    try:
        from pathlib import Path
        prob = predict_single(model, Path(tfile.name))
        latency = (time.time() - start) * 1000
        
        prediction = "SCREEN RECAPTURE" if prob > 0.50 else "REAL PHOTO"
        color = "red" if prob > 0.50 else "green"
        
        st.markdown(f"## Prediction: <span style='color:{color}'>{prediction}</span>", unsafe_allow_html=True)
        st.markdown(f"### Probability of Screen: **{prob:.4f}**")
        st.markdown(f"### Inference Latency: **{latency:.1f} ms**")
    except Exception as e:
        st.error(f"Error processing image: {e}")
        
    os.unlink(tfile.name)
else:
    st.info("Upload an image to detect if it is a physical photo or a picture of a screen.")
