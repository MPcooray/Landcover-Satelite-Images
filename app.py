import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Satellite Land Cover AI", layout="wide")

CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation',
    'Highway', 'Industrial', 'Pasture',
    'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

IMG_SIZE = 224

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenet_landcover.keras")

model = load_model()

# Extract backbone and classifier
backbone = model.layers[0]
gap = model.layers[1]
dropout = model.layers[2]
dense = model.layers[3]

# -----------------------------
# Preprocess
# -----------------------------
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

# -----------------------------
# Grad-CAM (Stable Version)
# -----------------------------
def make_gradcam_heatmap(img_array):

    # Get last conv layer from backbone
    last_conv_layer = backbone.get_layer("out_relu")

    # Create a model from backbone input to last conv output
    conv_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=last_conv_layer.output
    )

    with tf.GradientTape() as tape:
        conv_output = conv_model(img_array)
        tape.watch(conv_output)

        # Forward pass through classifier manually
        x = gap(conv_output)
        x = dropout(x)
        predictions = dense(x)

        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()

# -----------------------------
# UI
# -----------------------------
st.title("🛰️ Satellite Land Cover Classification")
st.markdown("Upload an image and visualize Grad-CAM attention 🔥")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = preprocess_image(image)

    col1, col2 = st.columns(2)

    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    with col1:
        st.image(image, use_container_width=True)
        st.markdown(f"### Prediction: `{CLASS_NAMES[pred_index]}`")
        st.markdown(f"### Confidence: `{confidence:.4f}`")

        df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": predictions[0]
        })
        st.bar_chart(df.set_index("Class"))

    heatmap = make_gradcam_heatmap(img_array)

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_np = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
    superimposed_img = heatmap * 0.4 + img_np
    superimposed_img = np.uint8(superimposed_img)

    with col2:
        st.image(superimposed_img, use_container_width=True)
        st.markdown("🔥 Red areas show model focus.")