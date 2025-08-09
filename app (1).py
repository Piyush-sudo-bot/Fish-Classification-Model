
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

MODEL_PATH = "CNN_model.h5"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (128, 128)

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names

model, CLASS_NAMES = load_resources()

st.set_page_config(page_title="Fish Classifier", page_icon="🐟", layout="centered")
st.title("🐟 Fish Species Classifier")
st.write("Upload an image of a fish and get the predicted species.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds) * 100

    st.subheader("Prediction Result")
    st.write(f"**Species:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

st.markdown("---")
st.markdown("### How to Run Locally")
st.code("streamlit run app.py", language="bash")
