import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

st.set_page_config(page_title="Pneumonia Detection", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pneumonia_model.keras")

model = load_model()
class_names = ["NORMAL", "PNEUMONIA"]

st.title("🫁 Pneumonia Detection System (AI Powered)")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Detect Pneumonia"):
        img = image.resize((128, 128))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = model.predict(img_array)
        confidence = float(np.max(prediction)) * 100
        result = class_names[int(np.argmax(prediction))]

        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {confidence:.2f}%")

