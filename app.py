import streamlit as st
import eda          #modul halaman EDA
import prediction   #modul halaman Prediksi

#konfigurasi umum
st.set_page_config(
    page_title="Accident Detection from CCTV",
    layout="wide"
)

#sidebar navigasi
st.sidebar.title("Accident Detection")
page = st.sidebar.radio(
    "Navigasi Halaman",
    ["Home", "Exploratory Data Analysis", "Prediksi Gambar"]
)

#halaman home
if page == "Home":
    st.title("Accident Detection from CCTV Footage")
    st.write(
        """
        Computer Vision Model yang membangun model klasifikasi citra untuk mendeteksi kecelakaan lalu lintas
        dari frame CCTV secara otomatis.
        """
    )

    st.subheader("Latar Belakang")
    st.write(
        """
        Kecelakaan lalu lintas yang tidak segera terdeteksi dapat memperlambat respons, sehingga memperbesar risiko fatalitas.
        Aplikasi ini membantu mengotomatisasi proses deteksi tersebut dari visual frame CCTV.
        """
    )

    st.subheader("Cara Menggunakan")
    st.markdown(
        """
        1. Buka halaman **Exploratory Data Analysis** untuk melihat gambaran dataset
           yang dipakai untuk melatih model.
        2. Buka halaman **Prediksi Gambar** untuk mengunggah gambar CCTV baru dan
           melihat hasil klasifikasi model (Accident / Non Accident).
        """
    )

#halaman EDA
elif page == "Exploratory Data Analysis":
    eda.run()

#halaman Prediksi
elif page == "Prediksi Gambar":
    prediction.run()
