# 🚧 Pothole Detection — Interactive Dashboard

🔗 **Live Demo**: [https://potholedetectiondashboard.streamlit.app/](https://potholedetectiondashboard.streamlit.app/)

Dashboard interaktif berbasis Streamlit untuk menampilkan insight dan kesimpulan dari proyek deteksi kerusakan jalan (pothole) menggunakan YOLOv8n segmentation.

---

## 📋 Deskripsi Proyek

Proyek ini merupakan bagian dari **Capstone Project Dicoding** yang membahas pipeline data science lengkap:

1. **Data Gathering** — Dataset pothole segmentation dari Roboflow Universe
2. **Data Assessment** — Analisis distribusi, kualitas, dan karakteristik data
3. **Data Cleaning** — Penanganan missing label dan verifikasi konsistensi data
4. **Augmentasi** — Penambahan data dengan Albumentations
5. **Pemodelan** — Training YOLOv8n segmentation (50 epoch)
6. **Evaluasi** — Analisis metrik model dan kurva training
7. **Insight & Kesimpulan** — Jawaban atas pertanyaan bisnis

---

## 📁 Struktur Folder

```
dashboard_pothole/
├── dashboard_pothole.py          # Aplikasi Streamlit utama
├── generate_data_dictionary.py   # Script generate Data Dictionary Excel
├── data_dictionary.xlsx          # Output Data Dictionary (df_final_full)
├── requirements.txt              # Daftar library yang dibutuhkan
├── results.csv                   # Log training YOLO (50 epoch)
├── README.md                     # Dokumentasi ini
├── dataset/
│   ├── train/
│   │   ├── images/               # 720 gambar training (640×640 JPG)
│   │   └── labels/               # 720 file anotasi YOLO segmentation
│   ├── valid/
│   │   ├── images/               # 60 gambar validasi
│   │   └── labels/               # 60 file anotasi
│   └── test/
│       ├── images/               # 60 gambar test
│       └── labels/               # 60 file anotasi
└── augmented_pothole_images/     # 30 gambar hasil augmentasi
```

---

## 🚀 Cara Menjalankan Dashboard

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Streamlit

```bash
streamlit run dashboard_pothole.py
```

### 3. (Opsional) Generate ulang Data Dictionary

```bash
python generate_data_dictionary.py
```

---

## 📊 Halaman Dashboard

| Halaman | Konten |
|---|---|
| 🏠 Overview | KPI dataset, distribusi pothole, proporsi split |
| 📊 Data Assessment | Analisis kualitas, area bbox, distribusi label |
| 🧹 Data Cleaning | Hasil cleaning, resolusi gambar, Data Dictionary |
| 🔄 Augmentasi | Teknik augmentasi, radar chart, hasil augmentasi |
| 🤖 Evaluasi Model | Metrik YOLOv8n, confusion matrix, kurva training |
| 📹 Video Inferensi | Simulasi deteksi per frame, distribusi bahaya |
| 💡 Insight & Kesimpulan | Jawaban PB1 & PB2, rekomendasi, profil sistem |

---

## 🤖 Model

| Atribut | Nilai |
|---|---|
| Arsitektur | YOLOv8n Segmentation |
| Epoch | 50 |
| Best Epoch | 37 |
| mAP@50 | 0.7168 |
| mAP@50-95 | 0.4230 |
| Precision | 0.6537 |
| Recall | 0.7137 |

---

## 📦 Dataset

- **Sumber**: [Roboflow Universe — Pothole Segmentation YOLOv8](https://universe.roboflow.com)
- **Task**: Instance Segmentation
- **Kelas**: 1 (pothole)
- **Total gambar**: 840 (720 train + 60 valid + 60 test + 30 augmented)
- **Resolusi**: 640 × 640 px (seragam)
- **Format label**: YOLO segmentation polygon (normalized coordinates)

---

## 🛠️ Library

| Library | Kegunaan |
|---|---|
| `streamlit` | Framework dashboard interaktif |
| `pandas` | Manipulasi data tabular |
| `numpy` | Komputasi numerik |
| `matplotlib` | Visualisasi chart (histogram, line, scatter, polar) |
| `plotly` | Visualisasi interaktif (bar, pie) |
| `openpyxl` | Generate file Excel Data Dictionary |
