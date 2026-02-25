import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Satellite Land Cover AI", layout="wide")

# ------------------------------
# Class Labels (EuroSAT)
# ------------------------------
CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation',
    'Highway', 'Industrial', 'Pasture',
    'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

IMG_SIZE = 224

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mobilenet_landcover.keras")
    return model

model = load_model()

# ------------------------------
# Image Preprocessing
# ------------------------------
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

# ------------------------------
# Grad-CAM Function
# ------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu"):
    
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# ------------------------------
# UI Styling
# ------------------------------
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1 {color: white; text-align: center;}
    </style>
""", unsafe_allow_html=True)

st.title("🛰️ Satellite Land Cover Classification")
st.markdown("Upload a satellite image and visualize AI attention using Grad-CAM 🔥")

# ------------------------------
# File Upload
# ------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Satellite Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_array = preprocess_image(image)

    col1, col2 = st.columns(2)

    # ------------------------------
    # Prediction
    # ------------------------------
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_container_width=True)

        st.markdown(f"### 🧠 Prediction: `{CLASS_NAMES[pred_index]}`")
        st.markdown(f"### 🎯 Confidence: `{confidence:.4f}`")

        # Probability Chart
        df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": predictions[0]
        })

        st.bar_chart(df.set_index("Class"))

    # ------------------------------
    # Grad-CAM Visualization
    # ------------------------------
    heatmap = make_gradcam_heatmap(img_array, model)

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_np = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
    superimposed_img = heatmap * 0.4 + img_np
    superimposed_img = np.uint8(superimposed_img)

    with col2:
        st.subheader("🔥 Grad-CAM Visualization")
        st.image(superimposed_img, use_container_width=True)

        st.markdown("""
        🔎 **Red regions show where the model focused most.**
        """)
