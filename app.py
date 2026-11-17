import streamlit as st
import pandas as pd
import pickle

# Konfigurasi halaman web
st.set_page_config(
    page_title="Klasifikasi Anemia - Puskesmas Wates",
    page_icon="🩺",
    layout="centered"
)

# Fungsi memuat model (di-cache agar cepat)
@st.cache_resource
def load_model():
    with open('model_anemia_nb.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# --- TAMPILAN SIDEBAR ---
with st.sidebar:
    st.header("Tentang Sistem")
    st.write(
        "klasifikasi penyakit anemia menggunakan  "
        "algoritma **Naive Bayes** berdasarkan dataset kaggle."
    )
    st.markdown("---")
    st.subheader("Panduan Nilai Rujukan Umum:")
    st.markdown(
        "- **Hemoglobin (Hb):**\n"
        "  - Perempuan normal: ≥ 12.0 g/dL\n"
        "  - Laki-laki normal: ≥ 13.5 g/dL\n"
        "- **MCV:** 80 – 100 fL\n"
        "- **MCH:** 27 – 31 pg\n"
        "- **MCHC:** 32 – 36 g/dL"
    )
    st.markdown("---")
    st.caption("Skripsi Teknik Informatika - Universitas Aisyah Pringsewu")

# --- TAMPILAN UTAMA ---
st.title("🩺 Sistem Klasifikasi Penyakit Anemia")
st.subheader("Puskesmas Wates — Algoritma Naive Bayes")
st.write("Masukkan data hasil pemeriksaan laboratorium pasien pada formulir di bawah ini:")

# Formulir input data pasien
with st.form("form_prediksi"):
    nama_pasien = st.text_input("Nama / No. Rekam Medis Pasien (Opsional)", placeholder="Contoh: Pasien-001")
    
    col1, col2 = st.columns(2)
    with col1:
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["Perempuan", "Laki-laki"])
        hemoglobin = st.number_input("Kadar Hemoglobin / Hb (g/dL)", min_value=1.0, max_value=25.0, value=12.5, step=0.1)
        mch = st.number_input("MCH (pg)", min_value=5.0, max_value=50.0, value=24.0, step=0.1)
    with col2:
        mchc = st.number_input("MCHC (g/dL)", min_value=10.0, max_value=50.0, value=30.0, step=0.1)
        mcv = st.number_input("MCV (fL)", min_value=30.0, max_value=150.0, value=85.0, step=0.1)

    tombol_prediksi = st.form_submit_button("🔍 Prediksi Status Anemia", type="primary")

# Proses prediksi saat tombol diklik
if tombol_prediksi:
    # Encoding jenis kelamin sesuai data training (Perempuan=0, Laki-laki=1)
    gender_num = 0 if jenis_kelamin == "Perempuan" else 1

    # Susun input menjadi DataFrame dengan urutan kolom persis seperti saat training
    data_input = pd.DataFrame(
        [[gender_num, hemoglobin, mch, mchc, mcv]],
        columns=['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']
    )

    # Prediksi kelas dan probabilitas
    prediksi = model.predict(data_input)[0]
    probabilitas = model.predict_proba(data_input)[0]
    prob_normal = probabilitas[0] * 100
    prob_anemia = probabilitas[1] * 100

    st.markdown("---")
    st.subheader("📋 Hasil Klasifikasi")
    if nama_pasien.strip():
        st.write(f"**Pasien:** {nama_pasien}")

    if prediksi == 1:
        st.error("⚠️ **HASIL PREDIKSI: POSITIF ANEMIA**")
        st.write(f"Tingkat keyakinan model (Probabilitas Anemia): **{prob_anemia:.2f}%**")
    else:
        st.success("✅ **HASIL PREDIKSI: TIDAK ANEMIA (NORMAL)**")
        st.write(f"Tingkat keyakinan model (Probabilitas Normal): **{prob_normal:.2f}%**")

    # Tampilkan rincian probabilitas
    col_p1, col_p2 = st.columns(2)
    col_p1.metric("Probabilitas Tidak Anemia", f"{prob_normal:.2f}%")
    col_p2.metric("Probabilitas Anemia", f"{prob_anemia:.2f}%")

    st.info(
        "Catatan: Hasil ini merupakan prediksi sistem pendukung keputusan berbasis data historis. "
        "Keputusan diagnosis akhir tetap berada di tangan tenaga medis."
    )
