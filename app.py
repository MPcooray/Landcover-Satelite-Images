import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
from PIL import Image

IMG_SIZE = 224
NUM_CLASSES = 10

class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation',
    'Highway', 'Industrial', 'Pasture',
    'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# -----------------------------
# Load Model (Lazy Import TF)
# -----------------------------
@st.cache_resource
def load_model():

    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers, models

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.load_weights("mobilenet_landcover.weights.h5")

    return model

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Land Cover Classification")
st.title("Satellite Land Cover Classification")

model = load_model()

st.success("Model loaded successfully!")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = np.max(preds[0])

    st.subheader("Prediction")
    st.write(f"Class: **{class_names[class_idx]}**")
    st.write(f"Confidence: **{confidence:.4f}**")