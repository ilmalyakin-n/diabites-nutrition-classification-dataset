# 🩺 DiaBites: Nutrition Classification Model (Data Science Pipeline)

Repositori ini berisi keseluruhan *pipeline* Data Science untuk proyek **DiaBites**, sebuah Sistem Pendukung Keputusan (*Decision Support System*) berbasis *Machine Learning* yang dirancang untuk mengklasifikasikan tingkat keamanan produk makanan/minuman kemasan, khususnya bagi penderita diabetes.

Proyek ini merupakan luaran dari **Checkpoint 1 (Data Science & AI Engine)**, mencakup tahapan *Data Wrangling*, *Exploratory Data Analysis (EDA)*, hingga *Model Training* dan *Export* menggunakan TensorFlow Functional API.

🔗 **Tautan Penting:**
* **Sumber Data Asli (Kaggle):** [Informasi Nilai Gizi Indonesia](https://www.kaggle.com/datasets/fritze/informasi-nilai-gizi-indonesia)
* **Source Code Lengkap (Google Drive):** [Jupyter Notebooks (.ipynb)](https://drive.google.com/drive/folders/1RTcK1ZuyaJRQ2rLuorRDDAtR_hLNPiCV?usp=sharing)

---

## 📂 Struktur Repositori

Repositori ini dibagi menjadi dua komponen utama: **Source Code** dan **Production Artifacts**.

### 1. Source Code (Tersedia di Tautan Google Drive di atas)
File-file ini dijalankan menggunakan Google Colab untuk proses analitik dan pelatihan model:
* 📓 `Gathering_Assessing_Cleaning.ipynb`: Proses akuisisi data, pembersihan, dan *Synthetic Augmentation* (menghasilkan 11.557 baris data yang *Context-Aware*).
* 📊 `EDA.ipynb`: Analisis untuk menjawab pertanyaan bisnis dan memvalidasi korelasi gizi.
* 🧠 `pengujian.ipynb`: Pembuatan dan pengujian model Deep Learning dengan TensorFlow Functional API, SMOTE, dan Custom Callback.

### 2. Production Artifacts (Tersedia di Repositori Ini)
File-file ini adalah hasil *training* yang siap diserahkan kepada tim *AI Engineer* / *Backend* untuk proses *inference*:
* ⚙️ `model.keras`: Model *Deep Learning* utama.
* 📏 `scaler.pkl`: Objek *StandardScaler* untuk normalisasi skala fitur input.
* 🔠 `label_encoder.pkl`: Objek *LabelEncoder* untuk menerjemahkan output prediksi kembali menjadi teks (`Recommended`, `Caution`, `Not Recommended`).
* **dapatkan Model (Releases):** [diabites_model_artifacts_v1.0.zip](https://github.com/ilmalyakin-n/diabites-nutrition-classification-dataset/releases/tag/v1.0.0-checkpoint)
---

## 📊 Exploratory & Explanatory Data Analysis (EDA)

Berikut adalah beberapa temuan kunci dari proses analisis data yang mendasari logika medis pada aplikasi DiaBites:

### 1. Lanskap Nutrisi Produk Kemasan di Indonesia
Berdasarkan data produk asli (sebelum augmentasi), mayoritas produk memiliki gula di bawah 15g per sajian. Namun, terdapat *outlier* ekstrem dengan kandungan gula melebihi 40g per sajian, yang sangat berbahaya bagi penderita diabetes jika tidak ada sistem peringatan dini.

![Distribusi Gula dan Sodium](https://github.com/ilmalyakinn/assets/blob/1f1b2b93ac24e9c8080c8d02f65f275d7c0fee91/dataset_clasification/Visualisasi%20Distribusi%20Gula%20dan%20Sodium%20pada%20produk%20ASLI%20(tanpa%20augmentasi).png)
*(Gambar 1: Distribusi kandungan gula dan sodium pada makanan/minuman kemasan)*

### 2. Korelasi Antar Kandungan Gizi
Korelasi kuat terlihat antara Kalori dan Lemak (0.80), membuktikan bahwa manajemen kalori bagi penderita Diabetes Tipe 2 secara simultan akan membantu mengurangi asupan lemak berlebih.

![Heatmap Korelasi Gizi](https://github.com/ilmalyakinn/assets/blob/1f1b2b93ac24e9c8080c8d02f65f275d7c0fee91/dataset_clasification/Korelasi%20antar%20kandungan%20gizi.png)

*(Gambar 2: Heatmap korelasi antar komponen makronutrisi)*

### 3. Keberhasilan Personalisasi Sistem AI (Core Insight)
Grafik di bawah membuktikan bahwa sistem DiaBites tidak memukul rata semua pengguna. Terjadi pergeseran drastis pada label "Recommended" yang berkurang signifikan saat profil beralih dari Normal ke Diabetes Tipe 1 & 2, sementara produk berlabel "Not Recommended" melonjak tajam. Ini adalah bukti empiris bahwa fitur personalisasi proteksi medis bekerja dengan sangat baik.

![Perbandingan Label Rekomendasi](https://github.com/ilmalyakinn/assets/blob/1f1b2b93ac24e9c8080c8d02f65f275d7c0fee91/dataset_clasification/Membandingkan%20Label%20Rekomendasi%20berdasarkan%20Tipe%20Diabetes.png)
*(Gambar 3: Perubahan keputusan AI berdasarkan Tipe Diabetes)*

### 4. Faktor Penentu Risiko Utama (Diabetes Tipe 2)
Sebaran data menunjukkan bahwa pada label "Not Recommended" untuk pasien Tipe 2, variabel gula menjadi salah satu faktor penentu utama dengan nilai yang jauh melampaui batas aman.

![Sebaran Gula Pasien Tipe 2](https://github.com/ilmalyakinn/assets/blob/1f1b2b93ac24e9c8080c8d02f65f275d7c0fee91/dataset_clasification/pasien%20Diabetes%20Tipe%202.png)
*(Gambar 4: Sebaran kandungan gula berdasarkan output label)*

---

## 🚀 Performa Model (Medical-Grade Standard)

Model dievaluasi menggunakan data uji (*Test Set*) murni tanpa SMOTE untuk merepresentasikan kondisi dunia nyata.
* **Accuracy:** 93%
* **F1-Score:** Terdistribusi merata (>0.89) untuk ketiga kelas.
* **Keamanan Medis (Zero Fatal Errors):** Model memiliki angka **0 False Negatives** pada prediksi produk berbahaya ("Not Recommended") yang ditebak aman ("Recommended"). Ini membuktikan sistem sangat aman digunakan.

---

## 📋 Shared Contract (Untuk AI Engineer / Backend)

Tim *Backend* **WAJIB** mengikuti urutan 8 parameter input (Array 1D) berikut sebelum melakukan *scaling* dan *predict*:

1. `sugar_g` (Float) - Gula (Gram)
2. `carbs_g` (Float) - Karbohidrat (Gram)
3. `calories` (Float) - Kalori (Kkal)
4. `sodium_mg` (Float) - Sodium (Miligram)
5. `fat_g` (Float) - Lemak (Gram)
6. `age_group` (Integer) - `0: Child, 1: Adult, 2: Elderly`
7. `bmi_category` (Integer) - `0: Underweight, 1: Normal, 2: Overweight, 3: Obese`
8. `diabetes_type` (Integer) - `0: Normal, 1: Type 1, 2: Type 2`

---

## 💻 Panduan Inference (Penggunaan Model)

Contoh implementasi *inference* pada sisi server (Python):

```python
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 1. Load Model & Transformers
model = load_model('model.keras')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# 2. Contoh Input User (Sesuai Shared Contract)
user_input = np.array([[12.5, 30.0, 150.0, 250.0, 4.5, 1, 2, 2]]) 

# 3. Preprocessing & Predict
input_scaled = scaler.transform(user_input)
prediction_probs = model.predict(input_scaled)
predicted_index = np.argmax(prediction_probs)

# 4. Output
final_recommendation = label_encoder.inverse_transform([predicted_index])[0]
print(f"Rekomendasi Sistem: {final_recommendation}")
