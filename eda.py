#halaman Exploratory Data Analysis

import streamlit as st
import pandas as pd
import json
import os
from PIL import Image


def load_class_distribution(json_path="assets/class_distribution.json"):
#load data per kelas & split dari file JSON
    with open(json_path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def run():
#fungsi utama halaman EDA
    st.title("Exploratory Data Analysis")
    st.write(
        """
        Ringkasan eksplorasi data dari dataset **Accident Detection from CCTV Footage**
        yang digunakan untuk melatih model.
        """
    )

#distribusi Jumlah Data per Kelas & Split

    st.subheader("1. Distribusi Jumlah Data")

    df_dist = load_class_distribution()
#transpose agar split (train/val/test) jadi sumbu x
    st.bar_chart(df_dist.T)

    st.dataframe(df_dist)

    st.info(
        """
        **Insight:** Total dataset berjumlah 989 gambar, terbagi menjadi train (791),
        val (98), dan test (100). Distribusi kelas cukup seimbang pada tiap split
        (contoh pada train: 369 Accident vs 422 Non Accident, rasio sekitar 47:53),
        sehingga model tidak memerlukan penanganan class imbalance khusus seperti
        class_weight.
        """
    )

#gambar tiap kelas
    st.subheader("2. Contoh Gambar Tiap Kelas")

    class_names = ["Accident", "Non Accident"]
    cols = st.columns(len(class_names))

    for col, cls in zip(cols, class_names):
        col.markdown(f"**{cls}**")
        sample_dir = os.path.join("assets", "sample_images", cls)

        if os.path.exists(sample_dir):
            sample_files = [
                f for f in os.listdir(sample_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
#maksimal 3 gambar per kelas agar halaman tidak terlalu panjang
            ][:3]

            if sample_files:
                for fname in sample_files:
                    img = Image.open(os.path.join(sample_dir, fname))
                    col.image(img, use_container_width=True)
            else:
                col.warning(f"Belum ada sample gambar untuk kelas '{cls}'.")
        else:
            col.warning(f"Folder sample untuk kelas '{cls}' belum ditemukan.")
