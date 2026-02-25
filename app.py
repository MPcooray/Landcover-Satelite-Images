import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.models import Model

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Land Cover AI", layout="wide")

IMG_SIZE = 224

class_names = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

# ---------------------------------------------------
# LOAD MODEL (.keras full model)
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mobilenet_landcover.keras")
    return model

model = load_model()

# ---------------------------------------------------
# GRAD-CAM FUNCTION (MobileNetV2)
# ---------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu"):

    # Extract base model (MobileNetV2)
    base_model = model.layers[0]

    grad_model = Model(
        inputs=model.input,
        outputs=[base_model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ---------------------------------------------------
# UI HEADER
# ---------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🛰 Satellite Land Cover Classification</h1>",
    unsafe_allow_html=True
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Satellite Image")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = np.max(preds[0])

    # Show uploaded image
    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    # Show prediction
    with col2:
        st.subheader("🤖 Prediction Result")
        st.success(f"Predicted Class: **{class_names[class_idx]}**")
        st.metric("Confidence", f"{confidence * 100:.2f}%")

        # Probability chart
        df = pd.DataFrame({
            "Class": class_names,
            "Probability": preds[0]
        })

        st.subheader("📊 Class Probabilities")
        st.bar_chart(df.set_index("Class"))

    # ---------------------------------------------------
    # GRAD-CAM VISUALIZATION
    # ---------------------------------------------------
    heatmap = make_gradcam_heatmap(img_array, model, "out_relu")

    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + np.array(img)
    superimposed_img = np.uint8(superimposed_img)

    st.markdown("---")
    st.subheader("🔥 Grad-CAM Visualization (Model Attention)")
    st.image(superimposed_img, use_column_width=True)

else:
    st.info("Upload a satellite image to begin.")