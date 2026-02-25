import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import json

# MUST be first Streamlit command
st.set_page_config(page_title="Satellite Land Cover Classification")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("landcover_model.keras")

model = load_model()

# Load class names
with open("class_names.json") as f:
    class_names = json.load(f)

st.title("🌍 Satellite Land Cover Classification")
st.write("Upload a satellite image to classify land cover type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}")