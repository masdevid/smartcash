# 🔍 SmartCash: Deteksi Nilai Mata Uang Rupiah

SmartCash adalah sistem deteksi nilai mata uang Rupiah yang menggunakan YOLOv5 dengan EfficientNet-B4 sebagai backbone. Sistem ini dioptimasi untuk akurasi tinggi dalam berbagai kondisi pengambilan gambar.

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

### 3. Jalankan Aplikasi
```bash
python run.py
```

## 📚 Dokumentasi

### 🎯 Panduan Pengguna
- [Instalasi & Setup](docs/user_guide/INSTALASI.md) - Persiapan environment
- [CLI Interface](docs/user_guide/CLI.md) - Penggunaan antarmuka CLI
- [Training Guide](docs/user_guide/TRAINING.md) - Pelatihan model
- [Evaluation Guide](docs/user_guide/EVALUATION.md) - Evaluasi model
- [Troubleshooting](docs/user_guide/TROUBLESHOOTING.md) - Solusi masalah umum

### 💻 Dokumentasi Teknis
- [Arsitektur](docs/technical/ARSITEKTUR.md) - Desain sistem & komponen
- [Model](docs/technical/MODEL.md) - Implementasi YOLOv5 + EfficientNet
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
├── docs/                   # Dokumentasi
│   ├── dataset/           # Dokumentasi dataset
│   ├── dev/               # Dokumentasi developer
│   ├── technical/         # Dokumentasi teknis
│   └── user_guide/        # Panduan pengguna
├── notebooks/              # Jupyter notebooks
│   ├── 01_prepare_dataset.ipynb
│   ├── 02_train_model.ipynb
│   └── 03_evaluate_model.ipynb
├── smartcash/             # Source code
│   ├── handlers/          # Data & model handlers
│   ├── models/           # Model implementations
│   └── utils/            # Utility functions
├── tests/                 # Unit & integration tests
├── .env.example          # Environment template
├── LICENSE               # MIT License
├── README.md            # This file
└── requirements.txt     # Python dependencies
```

## 🤝 Kontribusi
Kami menyambut kontribusi! Lihat [CONTRIBUTING.md](docs/dev/CONTRIBUTING.md) untuk panduan.

## 📜 Lisensi
Project ini dilisensikan di bawah MIT License - lihat [LICENSE](LICENSE).

## 📚 Sitasi
```bibtex
@software{smartcash2025,
  title={SmartCash: Indonesian Banknote Detection with YOLOv5 & EfficientNet-B4},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/smartcash}
}