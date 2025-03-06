# 🔍 SmartCash
## Deteksi Nilai Mata Uang Rupiah

SmartCash adalah sistem deteksi nilai mata uang Rupiah yang menggunakan YOLOv5 dengan berbagai backbone, termasuk EfficientNet. Sistem ini dioptimasi untuk akurasi tinggi dalam berbagai kondisi pengambilan gambar.

## 📋 Quick Start

### 1. Setup Environment
```bash
# Create and activate environment
conda create -n smartcash python=3.9
conda activate smartcash

# Install dependencies
pip install -r requirements.txt
```

### 2. Persiapkan Dataset
- [Panduan Dataset](docs/dataset/README.md) - Persiapan & preprocessing dataset
- [Roboflow Integration](docs/dataset/ROBOFLOW.md) - Penggunaan dataset Roboflow

### 3. Jalankan Aplikasi di Google Colab
Untuk pengguna yang ingin menjalankan pelatihan atau eksperimen di Google Colab:

- Folder `notebooks/` berisi template cells yang siap di-copy paste ke Colab
- Template ini telah dikonfigurasi untuk memudahkan setup environment, loading dataset, dan proses pelatihan di Colab
- Gunakan template sesuai kebutuhan eksperimen Anda

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
├── configs/                 # Konfigurasi
│   ├── base_config.yaml    # Konfigurasi dasar
│   └── experiment/         # Konfigurasi eksperimen
├── data/                   # Dataset
│   ├── raw/               # Data mentah
│   └── processed/         # Data terproses
├── datasets/              # Dataset handlers & utilities
├── docs/                   # Dokumentasi
│   ├── dataset/           # Dokumentasi dataset
│   ├── dev/               # Dokumentasi developer
│   ├── technical/         # Dokumentasi teknis
│   └── user_guide/        # Panduan pengguna
├── exceptions/            # Custom exceptions
├── handlers/              # Data & model handlers
│   ├── ui_handlers/       # UI-specific handlers
│   ├── model_handler.py   # Model management
│   ├── data_manager.py    # Data management
│   └── evaluation_handler.py # Model evaluation
├── models/                # Model implementations
│   ├── backbones/         # Model backbones (EfficientNet, etc.)
│   ├── necks/             # Model neck components
│   └── yolov5_model.py    # YOLOv5 implementation
├── notebooks/             # Jupyter notebooks dengan cells template untuk Google Colab
├── pretrained/            # Pretrained model weights
├── runs/                  # Training & evaluation runs
├── tests/                 # Unit & integration tests
├── ui_components/         # UI components for the application
│   ├── training_components.py   # Training UI
│   ├── evaluation_components.py # Evaluation UI
│   └── model_components.py      # Model management UI
├── utils/                 # Utility functions
├── __main__.py           # Main application entry point
├── .env.example          # Environment template
├── LICENSE               # License file
├── README.md            # This file
└── requirements.txt     # Python dependencies
```

## 🤝 Kontribusi
Kami menyambut kontribusi! Lihat [CONTRIBUTING.md](docs/dev/CONTRIBUTING.md) untuk panduan.

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