# MG01 - SmartCash Model Architecture Refactor Guide

## Struktur Direktori


```
smartcash/model/
│
├── __init__.py             # Paket inisialisasi model
├── manager.py              # Koordinator alur kerja model tingkat tinggi
│
├── services/               # Layanan spesifik model
│   ├── __init__.py
│   │
│   ├── checkpoint/         # Manajemen checkpoint model
│   │   ├── __init__.py
│   │   ├── local_storage.py
│   │   ├── drive_storage.py
│   │   ├── sync_storage.py
│   │   └── cleanup.py
│   │
│   ├── training/           # Layanan pelatihan model
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── optimizer.py
│   │   ├── scheduler.py
│   │   ├── early_stopping.py
│   │   └── callbacks.py
│   │
│   ├── evaluation/         # Layanan evaluasi model
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   │
│   ├── prediction/         # Layanan prediksi model
│   │   ├── __init__.py
│   │   ├── core.py
│   │   └── postprocessing.py
│   │
│   ├── experiment/         # Manajemen eksperimen
│   │   ├── __init__.py
│   │   ├── tracking.py
│   │   ├── comparison.py
│   │   └── visualization.py
│   │
│   └── research/           # Skenario penelitian
│       ├── __init__.py
│       ├── scenarios.py
│       ├── benchmarking.py
│       └── ablation_study.py
│
├── config/                 # Konfigurasi model
│   ├── __init__.py
│   ├── base.py             # Konfigurasi dasar
│   ├── backbone.py         # Konfigurasi backbone
│   └── experiment.py       # Konfigurasi eksperimen
│
├── utils/                  # Utilitas model
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── metrics.py
│   └── validation.py
│
├── components/             # Komponen model yang dapat digunakan kembali
│   ├── __init__.py
│   └── losses.py
│
├── architectures/          # Arsitektur model
│   ├── __init__.py
│   ├── backbones/
│   │   ├── __init__.py
│   │   ├── efficientnet.py
│   │   ├── cspdarknet.py
│   │   └── base.py
│   ├── necks/
│   │   ├── __init__.py
│   │   └── fpn_pan.py
│   └── heads/
│       ├── __init__.py
│       └── detection_head.py
│
└── exceptions.py           # Eksepsi khusus model
```

## Migrasi dan Sejarah Refaktoring

### Dokumentasi Migrasi

Setiap metode dan kelas kritis didokumentasikan dengan catatan singkat yang menjelaskan:
- Path File absolut saat ini, contoh `File: /smartcash/model/services/training/core.py`
- Lokasi implementasi sebelumnya, contoh `Old: smartcash.model.handlers.model.core.model_trainer.train()`
- Ringkasan perubahan utama

Contoh dokumentasi:
```python
def train(self, dataset):
    """
    * old: handlers.model.core.model_trainer.train()
    * migrated: Simplified service-based training
    """
```
Variasi Key baris kedua:
```
* new: More modular evaluation strategy
* removed: Direct metrics computation
* added: Flexible service-based evaluation
* merged: Consolidate a() and b()
* migrated: Simplified service-based training
```
Konsep Arsitektur
1. Model Manager
File `manager.py` bertindak sebagai koordinator utama untuk alur kerja model. Ia menyediakan antarmuka tingkat tinggi untuk:

- Inisialisasi model
- Koordinasi proses pelatihan
- Manajemen evaluasi
- Kontrol prediksi

2. Layanan Model
Layanan dalam direktori services/ memecah fungsionalitas model menjadi komponen yang lebih kecil dan terfokus:

- Checkpoint Service

    - Manajemen penyimpanan lokal dan cloud
    - Sinkronisasi checkpoint
    - Pembersihan checkpoint lama

- Training Service

    - Konfigurasi optimizer
    - Manajemen scheduler
    - Implementasi early stopping
    - Penanganan callback

- Evaluation Service

    - Komputasi metrik
    - Visualisasi hasil evaluasi

- Prediction Service

    - Proses inferensi
    - Pascapemrosesan prediksi

3. Konfigurasi Model
Direktori config/ menyediakan manajemen konfigurasi yang fleksibel:

- Konfigurasi dasar model
- Konfigurasi spesifik backbone
- Konfigurasi eksperimen

4. Arsitektur Model
Direktori architectures/ mendefinisikan komponen model:

- Berbagai implementasi backbone
- Neck dan head untuk deteksi objek
- Basis untuk eksperimen arsitektur

5. Penanganan Kesalahan
File exceptions.py mendefinisikan hierarki eksepsi khusus:

- `ModelError`: Eksepsi dasar
- `ModelConfigurationError`
- `ModelTrainingError`
- `ModelInferenceError`

Prinsip Desain

- **Modularitas**: Setiap komponen memiliki tanggung jawab yang jelas
- **Fleksibilitas**: Mudah dikonfigurasi dan diperluas
- **Pemisahan Kepentingan**: Setiap layanan fokus pada fungsinya
- **Dapat Diuji**: Struktur yang mendukung pengujian unit
- **DRY**: Don't Repeat Yourself!


## Konvensi Penamaan dan Organisasi File

### Aturan Penamaan Kelas
- Setiap file berisi tepat satu kelas utama
- Nama kelas kontekstual dan menggambarkan fungsinya
- File menggunakan snake_case, kelas menggunakan PascalCase

### Contoh Konvensi Penamaan
```
| File                   | Kelas                   | Deskripsi                            |
|------------------------|-------------------------|--------------------------------------|
| `manager.py`           | `ModelManager`          | Koordinator utama alur kerja model   |
| `local_storage.py`     | `LocalCheckpointStorage`| Manajemen checkpoint di penyimpanan lokal |
| `drive_storage.py`     | `DriveCheckpointStorage`| Manajemen checkpoint di Google Drive  |
| `sync_storage.py`      | `CheckpointStorageSynchronizer` | Sinkronisasi checkpoint antar storage |
```

### Batasan Ukuran File
- Maksimal 500 baris kode per file
- Jika melebihi 500 baris, pertimbangkan untuk memecah menjadi kelas/file terpisah
- Prioritaskan Single Responsibility Principle (SRP)

### Panduan Modularisasi
- Setiap kelas harus memiliki tanggung jawab tunggal yang jelas
- Gunakan komposisi daripada warisan yang kompleks
- Manfaatkan dependency injection untuk fleksibilitas
- Sertakan dokumentasi singkat untuk setiap kelas dan metode
