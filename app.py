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

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenet_landcover.keras")

model = load_model()

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

# 🔥 Stable Grad-CAM
def make_gradcam_heatmap(img_array, model):

    # Get backbone (MobileNetV2)
    base_model = model.layers[0]
    last_conv_layer = base_model.get_layer("out_relu")

    # Create a model that maps input to:
    # 1. last conv layer output
    # 2. final prediction
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)

    return heatmap.numpy()

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

    heatmap = make_gradcam_heatmap(img_array, model)

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_np = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
    superimposed_img = heatmap * 0.4 + img_np
    superimposed_img = np.uint8(superimposed_img)

    with col2:
        st.image(superimposed_img, use_container_width=True)
        st.markdown("🔥 Red areas show model focus.")