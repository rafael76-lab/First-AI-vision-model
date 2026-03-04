import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Age Predictor", layout="centered")

st.title("📷 Age Prediction App")

# Cargar modelo solo una vez
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("age_prediction_model.keras")

model = load_model()

def preprocess_image(image):
    image = image.resize((300, 300))
    image = np.array(image)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Opción cámara o upload
option = st.radio("Choose input method:", ("Upload Image", "Use Camera"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Take a picture")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_container_width=True)

    if st.button("Predict Age"):
        with st.spinner("Analyzing..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)
            age = float(prediction[0][0])

        st.success(f"🎯 Predicted Age: {round(age, 2)} years")