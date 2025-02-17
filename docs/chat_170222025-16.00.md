# Ringkasan Sesi Pengembangan SmartCash
**Tanggal:** 17 Februari 2025  
**Waktu:** Sesi 1  
**Author:** Alfrida Sabar

## 📋 Overview
Pengembangan sistem deteksi nilai mata uang menggunakan YOLOv5 yang dioptimasi dengan EfficientNet-B4 sebagai backbone. Project ini bertujuan untuk meningkatkan akurasi deteksi nilai mata uang Rupiah dengan mempertimbangkan berbagai kondisi pengambilan gambar.

## 🎯 Objectives
1. Implementasi YOLOv5 dengan CSPDarknet (baseline)
2. Optimasi menggunakan EfficientNet-B4 backbone
3. Evaluasi performa pada 4 skenario pengujian
4. Analisis komparatif antara baseline dan model yang dioptimasi

## 📁 Struktur Project
```
smartcash/
├── configs/                  # Konfigurasi eksperimen
├── data/                    # Dataset storage
├── handlers/                # Data & model handlers
├── models/                  # Model implementations
├── utils/                   # Utilities
├── notebooks/              # Jupyter notebooks
└── README.md
```

## 💻 Komponen yang Telah Dibuat

### 1. Konfigurasi (configs/base_config.yaml)
- Parameter dataset dan model
- Definisi 4 skenario eksperimen
- Resource usage limit (60%)
- Konfigurasi visualisasi

### 2. Utils (utils/logger.py)
- Custom logger dengan emoji kontekstual
- Colored output untuk metrics
- Progress tracking untuk proses panjang

### 3. Data Handlers
a. **Local Handler (handlers/data_handler.py)**
   - Pengelolaan dataset lokal
   - Verifikasi struktur dan integritas
   - Statistik dataset

b. **Roboflow Handler (handlers/roboflow_handler.py)**
   - Integrasi dengan Roboflow API
   - Download dan setup otomatis
   - Metadata dataset

### 4. Model Components
a. **Baseline (models/baseline.py)**
   - Implementasi YOLOv5 + CSPDarknet
   - Training pipeline
   - Evaluasi metrics

b. **Model Handler (handlers/model_handler.py)**
   - Manajemen eksperimen
   - Training & evaluasi pipeline
   - Tracking hasil

## 📊 Dataset
- Total data: 2.740 samples
- 7 kelas denominasi Rupiah
- Split: train (70%), validation (15%), test (15%)
- Format anotasi: YOLOv5

## 🛠️ Next Steps
1. Implementasi EfficientNet-B4 backbone
2. Pembuatan notebook eksperimen baseline
3. Implementasi metric handler & visualisasi

## 📝 Catatan Penting
1. Dataset menggunakan struktur folder lokal, Roboflow handler disediakan untuk pengembangan masa depan
2. Resource usage dibatasi 60% dari available hardware
3. Visualisasi hasil perlu disimpan untuk analisis kualitatif
4. Evaluasi menggunakan metrik standar (accuracy, precision, recall, F1-score, mAP, inference time)

## 💡 Decisions Made
1. Pemisahan data handler untuk local dan Roboflow
2. Implementasi baseline sebelum model yang dioptimasi
3. Penggunaan config file untuk parameter eksperimen
4. Struktur modular untuk reusability

---
*Note: Dokumen ini akan diupdate seiring progress pengembangan*