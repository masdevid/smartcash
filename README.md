# 🔍 SmartCash: Deteksi Nilai Mata Uang Rupiah

SmartCash adalah sistem deteksi nilai mata uang Rupiah menggunakan model YOLOv5 dengan integrasi backbone EfficientNet-B4. Sistem ini dioptimasi untuk akurasi tinggi dalam berbagai kondisi pengambilan gambar dan mendukung deteksi multilayer untuk mata uang Rupiah.

> **UPDATE**: Aplikasi kini menggunakan framework Streamlit untuk menyediakan antarmuka pengguna yang lebih ringan dan menghindari konflik dependensi yang terjadi dengan implementasi Gradio sebelumnya.

## 📋 Quick Start

### 1. Setup Environment
```bash
# Create and activate environment
conda create -n smartcash python=3.9
conda activate smartcash

# Install dependencies
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi
```bash
# Metode 1: Menggunakan script helper
python run_app.py

# Metode 2: Langsung dengan Streamlit
streamlit run app.py
```

### 3. Persiapkan Dataset
- [Panduan Dataset](docs/dataset/README.md) - Persiapan & preprocessing dataset
- [Roboflow Integration](docs/dataset/ROBOFLOW.md) - Penggunaan dataset Roboflow

### 4. Gunakan Notebook di Google Colab
Untuk pengguna yang ingin menjalankan pelatihan atau eksperimen di Google Colab:

- Folder `notebooks/` berisi template cells yang siap di-copy paste ke Colab
- Template ini telah dikonfigurasi untuk memudahkan setup environment, loading dataset, dan proses pelatihan di Colab
- Gunakan template sesuai kebutuhan eksperimen Anda

## ✨ Fitur Utama

- ✅ **Deteksi Multi-layer**: Deteksi mata uang penuh, area nominal, dan fitur keamanan
- ✅ **Backbone Fleksibel**: Dukungan untuk EfficientNet-B4 dan CSPDarknet
- ✅ **Dataset Manager**: Pengelolaan dataset multilayer
- ✅ **Augmentasi**: Augmentasi gambar dengan berbagai teknik
- ✅ **Training Pipeline**: Pipeline training dengan visualisasi progres
- ✅ **Evaluasi**: Evaluasi model dan perbandingan performa
- ✅ **Deteksi Realtime**: Deteksi melalui kamera atau gambar
- ✅ **UI Streamlit**: Antarmuka pengguna intuitif dan mudah digunakan

## 📚 Dokumentasi

### 🎯 Panduan Pengguna
- [Instalasi & Setup](docs/user_guide/INSTALASI.md) - Persiapan environment
- [CLI Interface](docs/user_guide/CLI.md) - Penggunaan antarmuka CLI
- [Training Guide](docs/user_guide/TRAINING.md) - Pelatihan model
- [Evaluation Guide](docs/user_guide/EVALUATION.md) - Evaluasi model
- [Troubleshooting](docs/user_guide/TROUBLESHOOTING.md) - Solusi masalah umum

### 💻 Dokumentasi Teknis
- [Arsitektur](docs/technical/ARSITEKTUR.md) - Desain sistem & komponen
- [Model](docs/technical/MODEL.md) - Implementasi YOLOv5 dengan berbagai backbone
- [Dataset](docs/technical/DATASET.md) - Format & preprocessing data
- [Evaluasi](docs/technical/EVALUASI.md) - Metrik & metodologi evaluasi
- [API](docs/technical/API.md) - Dokumentasi API & integrasi

### 🛠️ Development
- [Contribution Guide](docs/dev/CONTRIBUTING.md) - Panduan kontribusi
- [Git Workflow](docs/dev/GIT_WORKFLOW.md) - Manajemen kode & versioning
- [Testing](docs/dev/TESTING.md) - Unit & integration testing
- [Code Style](docs/dev/CODE_STYLE.md) - Konvensi & best practices

## 🏗️ Struktur Project

```
smartcash/
├── app.py                     # Entry point aplikasi Streamlit
├── run_app.py                 # Script untuk menjalankan aplikasi
├── cli.py                     # Command Line Interface
├── configs/                   # Konfigurasi
│   ├── base_config.yaml       # Konfigurasi dasar
│   └── experiment/            # Konfigurasi eksperimen
├── data/                      # Dataset
│   ├── raw/                   # Data mentah
│   └── processed/             # Data terproses
├── docs/                      # Dokumentasi
│   ├── dataset/               # Dokumentasi dataset
│   ├── dev/                   # Dokumentasi developer
│   ├── technical/             # Dokumentasi teknis
│   └── user_guide/            # Panduan pengguna
├── exceptions/                # Custom exceptions
│   ├── __init__.py
│   ├── base.py                # Exception dasar
│   ├── factory.py             # Factory untuk pembuatan error
│   └── handler.py             # Handler untuk pengelolaan error
├── factories/                 # Factory pattern
│   ├── __init__.py
│   ├── dataset_component_factory.py  # Factory untuk komponen dataset
│   └── model_component_factory.py    # Factory untuk komponen model
├── handlers/                  # Handler untuk operasi kompleks
│   ├── __init__.py
│   ├── dataset/               # Pengelolaan dataset multilayer
│   ├── checkpoint/            # Pengelolaan checkpoint model
│   ├── detection/             # Proses deteksi mata uang
│   ├── evaluation/            # Evaluasi model
│   ├── model/                 # Pengelolaan model
│   ├── preprocessing/         # Preprocessing dataset
│   └── ui_handlers/           # UI-specific handlers
├── models/                    # Definisi model dan arsitektur
│   ├── __init__.py
│   ├── yolov5_model.py        # Model YOLOv5 dengan backbone fleksibel
│   ├── baseline.py            # Model baseline untuk SmartCash
│   ├── detection_head.py      # Detection head dengan opsi multilayer
│   ├── losses.py              # Custom loss functions
│   ├── backbones/             # Backbone architectures
│   └── necks/                 # Feature processing
├── notebooks/                 # Jupyter notebooks untuk Google Colab
├── pretrained/                # Pretrained model weights
├── runs/                      # Training & evaluation runs
├── tests/                     # Unit & integration tests
├── ui_components/             # UI components
│   ├── __init__.py
│   ├── data_components.py
│   ├── dataset_components.py
│   ├── model_components.py
│   ├── training_components.py
│   └── evaluation_components.py
├── utils/                     # Utilitas untuk berbagai fungsi
│   ├── __init__.py
│   ├── logger.py              # Sistem logging untuk SmartCash
│   ├── coordinate_utils.py    # Utilitas koordinat dan bounding box
│   ├── metrics.py             # Perhitungan metrik evaluasi model
│   ├── augmentation/          # Augmentasi dataset
│   ├── cache/                 # Sistem caching
│   ├── dataset/               # Validasi dan analisis dataset
│   ├── training/              # Pipeline training
│   └── visualization/         # Visualisasi hasil
├── .env.example               # Environment template
├── LICENSE                    # License file
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

## 📱 Penggunaan Aplikasi Streamlit

### 1. Setup Dataset

1. Buka tab **Dataset & Preprocessing**
2. Pilih sumber dataset (Roboflow atau lokal)
3. Unduh atau unggah dataset mata uang Rupiah
4. Validasi dataset untuk memastikan integritas

### 2. Augmentasi Data

1. Buka tab **Augmentasi Data**
2. Pilih jenis augmentasi yang diinginkan
3. Atur jumlah variasi per gambar
4. Jalankan augmentasi untuk memperkaya dataset

### 3. Training Model

1. Buka tab **Training**
2. Atur parameter training (epochs, batch size, learning rate, dll)
3. Pilih backbone (EfficientNet-B4 atau CSPDarknet)
4. Jalankan training dan pantau progres secara real-time

### 4. Evaluasi Model

1. Buka tab **Evaluasi**
2. Pilih model yang akan dievaluasi
3. Atur threshold confidence dan IoU
4. Jalankan evaluasi untuk mendapatkan metrik performa (mAP, precision, recall, F1)

### 5. Deteksi Mata Uang

1. Buka tab **Deteksi**
2. Pilih sumber gambar (upload, contoh, atau kamera)
3. Atur parameter deteksi
4. Jalankan deteksi dan lihat hasilnya

## 🔧 Prasyarat

- Python 3.7+
- CUDA toolkit (opsional, untuk akselerasi GPU)
- Dependensi dari requirements.txt

## 🌐 Integrasi dengan Google Colab

Untuk pengguna yang ingin menjalankan pelatihan atau eksperimen di Google Colab:

- Folder `notebooks/` berisi template cells yang siap di-copy paste ke Colab
- Template ini telah dikonfigurasi untuk memudahkan setup environment, loading dataset, dan proses pelatihan di Colab
- Gunakan template sesuai kebutuhan eksperimen Anda

Langkah-langkah penggunaan:
1. Buka notebook template di `notebooks/` yang sesuai dengan kebutuhan Anda
2. Copy cell-cell yang ada ke Google Colab 
3. Jalankan cell setup environment terlebih dahulu
4. Ikuti instruksi dalam notebook untuk menjalankan eksperimen

## 🤝 Kontribusi

Kami menyambut kontribusi! Untuk berkontribusi:
1. Fork repository
2. Buat branch fitur baru (`git checkout -b feature-baru`)
3. Commit perubahan (`git commit -m 'Menambahkan fitur baru'`)
4. Push ke branch (`git push origin feature-baru`)
5. Buat Pull Request

Lihat [CONTRIBUTING.md](docs/dev/CONTRIBUTING.md) untuk panduan lengkap.

## 📜 Lisensi / License

This project is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

**English:**
You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

for any purpose, even commercially.

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

**Bahasa Indonesia:**
Anda bebas untuk:
- **Berbagi** — menyalin dan menyebarluaskan materi dalam bentuk apa pun
- **Adaptasi** — menggubah, mengubah, dan membuat turunan dari materi

untuk tujuan apa pun, termasuk komersial.

Dengan ketentuan berikut:
- **Atribusi** — Anda harus memberikan kredit yang sesuai, menyediakan tautan ke lisensi, dan menunjukkan jika perubahan telah dilakukan. Anda dapat melakukannya dengan cara yang wajar, tetapi tidak dengan cara yang menyarankan pemberi lisensi mendukung Anda atau penggunaan Anda.

## 📚 Sitasi
```bibtex
@software{smartcash2025,
  title={SmartCash: Indonesian Banknote Detection with YOLOv5 & Various Backbones},
  author={Alfrida Sabar},
  year={2025},
  url={https://github.com/masdevid/smartcash}
}
```