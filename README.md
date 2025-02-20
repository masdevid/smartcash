# 📑 SmartCash - Overview & Petunjuk Penggunaan

## 🔍 Daftar Isi
- [Overview Project](#overview-project)
- [Petunjuk Penggunaan](#petunjuk-penggunaan)
- [Dokumentasi](#dokumentasi)
- [Struktur Project](#struktur-project)
- [Lisensi](#lisensi)
- [Sitasi](#sitasi)

## 🔍 Overview Project
SmartCash adalah sistem deteksi nilai mata uang Rupiah menggunakan algoritma YOLOv5 yang dioptimasi dengan arsitektur EfficientNet-B4 sebagai backbone. Tujuan project ini adalah meningkatkan akurasi deteksi nilai mata uang Rupiah dengan mempertimbangkan berbagai kondisi pengambilan gambar.

## 🚀 Petunjuk Penggunaan

### Persiapan Environment
1. Install ekstensi VSCode untuk Jupyter Notebook
   - Buka VSCode, klik menu Extensions (Ctrl+Shift+X)
   - Cari ekstensi "Jupyter" dan install
   - Restart VSCode agar ekstensi aktif

2. Install ekstensi "Markdown Preview Mermaid Support"
   - Ekstensi ini dibutuhkan untuk preview diagram Mermaid di README
   - Buka VSCode, klik menu Extensions (Ctrl+Shift+X)
   - Cari ekstensi "Markdown Preview Mermaid Support" dan install
   - Restart VSCode agar ekstensi aktif

3. Aktivasi conda environment
   ```bash
   conda create -n smartcash python=3.9
   conda activate smartcash
   pip install -r requirements.txt
   ```

4. Persiapkan dataset
   - Download dataset dari Roboflow atau gunakan dataset lokal
   - Struktur folder data:
     ```
     data/
     ├── train/
     │   ├── images/
     │   └── labels/
     ├── valid/
     │   ├── images/
     │   └── labels/
     └── test/
         ├── images/
         └── labels/
     ```

Untuk petunjuk lengkap penggunaan aplikasi, silakan lihat:
- [📱 Panduan Pengguna](docs/PANDUAN_PENGGUNA.md)

## 📚 Dokumentasi
Dokumentasi lengkap tentang project SmartCash tersedia di folder [`docs/`](./docs/):

### 📖 Panduan
- [📱 Panduan Pengguna](./docs/PANDUAN_PENGGUNA.md) - Petunjuk lengkap penggunaan aplikasi
- [📊 Panduan Anotasi Dataset](./docs/PANDUAN_ANOTASI.md) - Cara membuat & mengelola dataset
- [🔧 Panduan Preprocessing](./docs/PREPROCESSING.md) - Langkah-langkah preprocessing data

### 🏗️ Dokumentasi Teknis
- [🔍 Overview Project](./docs/SUMMARY.md) - Ringkasan & tujuan project
- [⚙️ Arsitektur Sistem](./docs/DOKUMENTASI_TEKNIS.md) - Detil implementasi sistem
- [🔄 Alur Kerja Git](./docs/GIT_WORKFLOW.md) - Panduan kolaborasi & version control

### 🧪 Eksperimen & Evaluasi
- [📈 Roboflow Integration](./docs/ROBOFLOW.md) - Konfigurasi & penggunaan Roboflow
- [🔬 YOLOv5 + EfficientNet](./docs/YOLOv5_EfficientNetB4_Backbone.md) - Detil arsitektur model

## 🏗️ Struktur Project
```
SmartCash/
├── configs/           # Konfigurasi eksperimen
├── data/             # Dataset (train/valid/test)
├── docs/             # Dokumentasi lengkap
├── handlers/         # Modul penanganan data
├── models/           # Implementasi model
├── notebooks/        # Jupyter notebooks
├── tests/           # Unit & integration tests
├── utils/           # Utilitas & helpers
├── .env.example     # Template environment vars
├── README.md        # Dokumentasi utama
└── requirements.txt # Dependencies
```

## 📜 Lisensi
Project ini dilisensikan di bawah MIT License - lihat file [LICENSE](LICENSE) untuk detil.

## 📚 Sitasi
Jika Anda menggunakan SmartCash dalam penelitian Anda, silakan sitasi:

```bibtex
@software{smartcash2025,
  author = {Your Name},
  title = {SmartCash: Indonesian Banknote Detection with YOLOv5 & EfficientNet-B4},
  year = {2025},
  url = {https://github.com/yourusername/smartcash}
}