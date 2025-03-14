# RT01 - SmartCash Root Project Architecture Guide

Dokumen ini menjelaskan arsitektur tingkat root project SmartCash, yang menggabungkan komponen dataset, detection, dan colab dalam satu project terintegrasi. Panduan ini mencakup struktur direktori dan hubungan antar komponen utama.

## Struktur Direktori Lengkap

Berikut adalah struktur direktori lengkap untuk project SmartCash yang mengintegrasikan semua komponen:

*Catatan: Struktur ini menggunakan pendekatan berbasis domain untuk organisasi code, dengan setiap domain fungsional (dataset, detection, colab) memiliki struktur modular yang konsisten.*

```
smartcash/
├── __init__.py             # Deklarasi package dan versi
├── main.py                 # Entry point untuk aplikasi CLI
├── cli.py                  # Command Line Interface
│
├── dataset/                # Komponen dataset 
│   ├── __init__.py         
│   ├── manager.py
│   └── ...                 # (lihat Dataset Architecture Refactor Guide)
│
├── detection/              # Komponen deteksi
│   ├── __init__.py
│   ├── detector.py
│   └── ...                 # (lihat Detection Architecture Refactor Guide)
│
├── colab/                  # Komponen notebook Colab
│   ├── __init__.py
│   ├── notebook_builder.py
│   └── ...                 # (lihat Colab Cell Template Architecture Guide)
│
├── config/                 # Konfigurasi aplikasi
│   ├── __init__.py
│   ├── default_config.yaml # Konfigurasi default
│   ├── training.yaml       # Konfigurasi training
│   ├── inference.yaml      # Konfigurasi inference
│   ├── augmentation.yaml   # Konfigurasi augmentasi
│   └── layers.yaml         # Konfigurasi layer deteksi
│
├── common/                 # Komponen umum terpusat
│   ├── __init__.py
│   ├── config.py           # Utilitas konfigurasi
│   ├── constants.py        # Konstanta global
│   ├── logger.py           # Sistem logging
│   ├── types.py            # Type definitions
│   └── exceptions.py       # Custom exceptions
│
├── handlers/               # Handler tingkat aplikasi
│   ├── __init__.py
│   ├── app_handler.py      # Handler aplikasi utama
│   ├── cli_handler.py      # Handler untuk CLI
│   └── api_handler.py      # Handler untuk API
│
├── api/                    # REST API (opsional)
│   ├── __init__.py
│   ├── server.py           # Server API
│   ├── routes.py           # Definisi routes API
│   └── models.py           # Model data API
│
├── scripts/                # Scripts utilitas
│   ├── benchmark.py        # Benchmark model
│   ├── export_model.py     # Export model ke berbagai format
│   ├── convert_dataset.py  # Konversi format dataset
│   └── setup_env.py        # Setup lingkungan pengembangan
│
├── tests/                  # Unit dan integration tests
│   ├── __init__.py
│   ├── test_dataset.py     # Test untuk dataset
│   ├── test_detection.py   # Test untuk detection
│   └── test_api.py         # Test untuk API
│
├── docs/                   # Dokumentasi
│   ├── architecture/       # Dokumentasi arsitektur (guides)
│   ├── api/                # Dokumentasi API
│   ├── user_guides/        # Panduan pengguna
│   └── research/           # Dokumentasi penelitian
│
├── notebooks/              # Jupyter Notebooks
│   ├── training.ipynb      # Notebook untuk training
│   ├── inference.ipynb     # Notebook untuk inferensi
│   ├── evaluation.ipynb    # Notebook untuk evaluasi
│   └── research.ipynb      # Notebook untuk penelitian
│
├── setup.py                # Script instalasi package
├── requirements.txt        # Dependency requirements
├── LICENSE                 # Informasi lisensi
└── README.md               # Dokumentasi utama project