import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import json

st.set_page_config(page_title="Satellite Land Cover Classification", layout="centered")

st.title("🌍 Satellite Land Cover Classification")
st.write("Upload a satellite image to classify land cover type.")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Load model (cached for performance)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("landcover_model.h5")

model = load_model()

IMG_SIZE = 224

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    st.subheader("Prediction:")
    st.success(f"{class_names[predicted_class]}")

    st.subheader("Confidence:")
    st.write(f"{confidence * 100:.2f}%")

    # Show probabilities
    st.subheader("Class Probabilities:")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{class_names[i]}: {prob * 100:.2f}%")