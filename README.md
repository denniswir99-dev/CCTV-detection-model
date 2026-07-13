---
title: Accident Detection from CCTV Footage
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# Accident Detection from CCTV Footage

Aplikasi ini mendeteksi apakah sebuah frame CCTV menunjukkan kondisi **Accident**
atau **Non Accident**, menggunakan model CNN (Transfer Learning MobileNetV2) yang
dilatih pada Graded Challenge 7.

- **Dataset**: https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage
- **Notebook training**: lihat repository GitHub tugas ini (link ada di url.txt)

## Cara Menggunakan
1. Buka halaman **Exploratory Data Analysis** untuk melihat gambaran dataset yang dipakai melatih model.
2. Buka halaman **Prediksi Gambar** untuk mengunggah gambar CCTV baru dan melihat hasil klasifikasi model.

> Catatan: `sdk_version` sengaja tidak diisi agar Space otomatis memakai versi Streamlit
> terbaru yang didukung Hugging Face. Jika muncul error kompatibilitas versi, samakan
> `sdk_version` di sini dengan versi Streamlit yang dipin di `requirements.txt`
> (lihat https://huggingface.co/docs/hub/spaces-config-reference).
