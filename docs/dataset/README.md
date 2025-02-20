# 📊 Persiapan Dataset SmartCash

## 📋 Overview

Dokumen ini menjelaskan langkah-langkah untuk mempersiapkan dataset yang akan digunakan dalam training model SmartCash.

## 🎯 Spesifikasi Dataset

### Format Gambar
- Resolusi: 640x640 pixels
- Format: JPG/JPEG
- Channel: RGB
- Kualitas: Minimum 80%

### Format Label
- Format: YOLO txt
- Struktur: `<class_id> <x_center> <y_center> <width> <height>`
- Class IDs:
  - 0: 1000
  - 1: 2000
  - 2: 5000
  - 3: 10000
  - 4: 20000
  - 5: 50000
  - 6: 100000

## 🔄 Preprocessing Steps

1. **Resize Images**
```bash
python scripts/resize_images.py --input data/raw --output data/processed --size 640
```

2. **Generate Labels**
```bash
python scripts/generate_labels.py --input data/raw --output data/processed
```

3. **Split Dataset**
```bash
python scripts/split_dataset.py --input data/processed --split 0.7 0.15 0.15
```

## 📁 Struktur Dataset

```
data/
├── raw/                    # Data mentah
│   ├── images/            # Gambar original
│   └── annotations/       # Anotasi original
├── processed/             # Data terproses
│   ├── train/            # Training split (70%)
│   │   ├── images/      
│   │   └── labels/      
│   ├── valid/            # Validation split (15%)
│   │   ├── images/      
│   │   └── labels/      
│   └── test/             # Testing split (15%)
│       ├── images/      
│       └── labels/      
└── augmented/            # Data augmentasi
    └── train/            # Hanya untuk training
        ├── images/      
        └── labels/      
```

## 🔍 Validasi Dataset

Gunakan script validasi untuk memastikan kualitas dataset:

```bash
python scripts/validate_dataset.py --input data/processed
```

Script ini akan memeriksa:
- Resolusi gambar
- Format label
- Distribusi kelas
- Kualitas gambar

## 📊 Statistik Dataset

- Total gambar: 17,500
- Distribusi per kelas: 2,500 gambar
- Rasio split: 70/15/15 (train/valid/test)
- Augmentasi: 3x lipat (hanya training)

## 🔄 Data Augmentation

Teknik augmentasi yang digunakan:
- Random rotation (±30°)
- Random brightness (±20%)
- Random contrast (±20%)
- Random noise (gaussian)
- Horizontal flip
- Vertical flip

## 🚀 Next Steps

1. Lihat [ROBOFLOW.md](ROBOFLOW.md) untuk integrasi dengan Roboflow
2. Mulai [training](../user_guide/TRAINING.md) model
3. [Evaluasi](../user_guide/EVALUATION.md) performa model
