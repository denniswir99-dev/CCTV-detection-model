#halaman prediksi gambar baru

import streamlit as st
import numpy as np
import json
import os
import gdown
import tensorflow as tf
from PIL import Image

MODEL_PATH = "best_model_accident_detection.keras"
CONFIG_PATH = "model_config.json"

#link gdrive untuk load model
GDRIVE_MODEL_URL = "https://drive.google.com/drive/folders/1JcUViymSVQq6sDWysuMx9vtFiPa_Rzbq?usp=sharing"


@st.cache_resource(show_spinner="Mengunduh & memuat model...")
def load_model_and_config():

    if not os.path.exists(MODEL_PATH):
        gdown.download(url=GDRIVE_MODEL_URL, output=MODEL_PATH)

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

#JSON menyimpan key dict sebagai string konversi kembali ke integer
    idx_to_class = {int(k): v for k, v in config["idx_to_class"].items()}
    img_size = tuple(config["img_size"])

    return model, idx_to_class, img_size


def preprocess_image(image: Image.Image, target_size):
#preprocessing gambar upload
    image = image.convert("RGB")
    image_resized = image.resize(target_size)
    image_array = np.array(image_resized) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)
    return image_batch


def run():
#fungsi utama halaman Prediksi, dipanggil dari app.py
    st.title("Prediksi Gambar CCTV Baru")
    st.write(
        "Unggah gambar frame CCTV untuk diklasifikasikan sebagai **Accident** "
        "atau **Non Accident**."
    )

#load model & config
#hanya dieksekusi sekali menggunakan cache
    model, idx_to_class, img_size = load_model_and_config()

    uploaded_file = st.file_uploader("Pilih gambar (.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar yang diunggah", use_container_width=True)

        image_batch = preprocess_image(image, img_size)

        with st.spinner("Melakukan prediksi..."):
            pred_prob = model.predict(image_batch, verbose=0)[0][0]

        pred_idx = int(pred_prob > 0.5)
#konversi angka ke label string
        pred_label = idx_to_class[pred_idx]        
        confidence = pred_prob if pred_idx == 1 else 1 - pred_prob

        with col2:
            if pred_label == "Accident":
                st.error(f"Prediksi dilakukan, hasil prediksi: **{pred_label}**")
            else:
                st.success(f"Prediksi dilakukan, hasil prediksi: **{pred_label}**")

            st.metric("Confidence", f"{confidence * 100:.2f}%")
    else:
        st.info("Silakan unggah gambar terlebih dahulu untuk melihat hasil prediksi.")
