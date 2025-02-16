# SmartCash: Deteksi Nominal Mata Uang Rupiah (WORK IN PROGRESS)

## 🎯 Tujuan Proyek

SmartCash adalah solusi canggih untuk deteksi dan identifikasi nominal mata uang Rupiah menggunakan teknologi deteksi objek berbasis deep learning.

## 📚 Panduan Penting Sebelum Memulai

### 🤝 Kontribusi dan Alur Kerja
Sebelum mulai berkontribusi, harap baca dengan seksama:
- [Panduan Alur Kerja Git](docs/WORKFLOW.md)
  - Cara clone repository
  - Membuat cabang develop dan fitur
  - Praktik commit terbaik
  - Etika kontribusi

- [Pedoman Pesan Commit](docs/COMMIT_GUIDELINES.md)
  - Format pesan commit yang benar
  - Jenis commit
  - Praktik penulisan pesan commit

## 🛠 Struktur Proyek

```
smartcash/             # Kode sumber utama
│                
├── main.py            # Titik masuk aplikasi
├── interfaces/        # Antarmuka pengguna
│   └──  handlers/          # Penanganan antarmuka dataset
├── models/            # Definisi model deteksi
├── data/              # Manajemen dataset
│   └── rupiah/            # Dataset mata uang Rupiah
│       ├── train/
│       ├── val/
│       └── test/
├── evaluation/        # Modul evaluasi model
├── training/          # Modul pelatihan model
├── utils/             # Utilitas pendukung
├── runs/              # Hasil eksperimen
├── weights/           # Model yang dilatih
├── docs/              # Dokumentasi proyek
├── config/            # Konfigurasi aplikasi
│
├── requirements.txt       # Dependensi utama
├── dev-requirements.txt   # Dependensi pengembangan
└── README.md              # Dokumentasi proyek
```

## 🚀 Persiapan Awal

### Prasyarat
- Python 3.8+
- Conda terinstal
- GPU CUDA (disarankan, tetapi tidak wajib)
- Minimal 16GB RAM
- Git terinstal

### Instalasi

1. Clone repository:
```bash
git clone https://github.com/masdevid/smartcash.git
cd smartcash
```

2. Buat virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac  
# venv\Scripts\activate   # Windows 
```

3. Instal dependensi:
```bash
pip install -r dev_requirements.txt
pip install -r requirements.txt
```

## 🖥️ Menjalankan Aplikasi

Cukup jalankan:
```bash
python main.py
```

### Menu Utama

1. **Manajemen Data**
   - Persiapan dataset
   - Augmentasi data
   - Verifikasi dataset
   - Statistik dataset

2. **Operasi Model**
   - Training model
   - Tuning hyperparameter
   - Melihat status model

3. **Pengujian & Evaluasi**
   - Jalankan skenario pengujian
   - Evaluasi performa model
   - Analisis error

4. **Ekspor Model**
   - Konversi model ke format berbeda

## 🧪 Skenario Pengujian

- Deteksi dalam pencahayaan normal
- Deteksi dalam pencahayaan rendah
- Deteksi objek kecil
- Deteksi dengan oklusi parsial

## 📊 Metrik Evaluasi

- Mean Average Precision (mAP)
- Precision
- Recall 
- Waktu Inferensi

## 🤝 Kontribusi

1. Baca panduan [panduan Alur Kerja Git](docs/WORKFLOW.md)
2. Buat branch fitur (`git checkout -b fitur/AturDeteksi`)
3. Commit perubahan sesuai [Pedoman Pesan Commit](docs/COMMIT_GUIDELINES.md)
4. Push ke branch
5. Buka Pull Request

## 📜 Lisensi

Proyek ini dilisensikan di bawah MIT License.

## 🏆 Sitasi

```
Sabar, A. (2025). Optimasi Deteksi Nominal Mata Uang dengan YOLOv5 dan EfficientNet-B4. (Unpublished)
```
