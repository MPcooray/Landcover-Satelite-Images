import streamlit as st
st.set_page_config(page_title="Satellite Land Cover Classifier")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import numpy as np
from PIL import Image
import json

IMG_SIZE = 224

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Build architecture EXACTLY like training
def build_model():
    base_model = MobileNetV2(
        weights=None,   # IMPORTANT
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation="softmax")
    ])

    return model

@st.cache_resource
def load_model():
    model = build_model()
    model.load_weights("mobilenet.weights.h5")
    return model

model = load_model()

st.title("🛰 Satellite Land Cover Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    st.subheader(f"Prediction: {class_names[pred_index]}")
    st.write(f"Confidence: {confidence:.4f}")