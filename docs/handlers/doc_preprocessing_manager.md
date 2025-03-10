Saya akan memperbarui dokumentasi preprocessing manager dengan fokus pada implementasi saat ini, secara singkat dan tanpa contoh kode:

# Dokumentasi PreprocessingManager SmartCash

## Deskripsi

`PreprocessingManager` adalah komponen pusat untuk pengelolaan preprocessing dataset multilayer mata uang Rupiah di SmartCash. Komponen ini menggunakan pola desain Facade untuk menyediakan antarmuka terpadu bagi berbagai operasi preprocessing.

## Struktur dan Komponen

```
smartcash/handlers/preprocessing/
├── __init__.py                  # Export komponen utama
├── manager.py                   # PreprocessingManager (facade utama)
├── validators.py                # Komponen untuk validasi dataset
├── augmentors.py                # Komponen untuk augmentasi dataset
└── analyzers.py                 # Komponen untuk analisis dataset
```

## Komponen Utama

### 1. PreprocessingManager

Manager utama yang berfungsi sebagai facade, mengintegrasikan semua komponen preprocessing dan mengekspos fungsionalitas untuk digunakan oleh klien.

Tanggung jawab:
- Inisialisasi komponen preprocessing (validator, augmentor, analyzer)
- Pengelolaan environment (Colab/lokal)
- Setup observer pattern untuk monitoring
- Menjalankan pipeline preprocessing lengkap
- Menyediakan akses ke fungsionalitas individual komponen

### 2. DatasetValidator

Komponen untuk validasi dataset yang menggunakan `EnhancedDatasetValidator` dari utils/dataset.

Tanggung jawab:
- Validasi integritas dataset (gambar dan label)
- Pendeteksian masalah label
- Perbaikan otomatis masalah yang ditemukan
- Notifikasi perubahan melalui observer pattern

### 3. DatasetAugmentor

Komponen untuk augmentasi dataset yang menggunakan `AugmentationManager` dari utils/augmentation.

Tanggung jawab:
- Augmentasi dataset dengan berbagai teknik
- Dukungan untuk kombinasi parameter augmentasi kustom
- Monitoring kemajuan proses augmentasi
- Validasi hasil augmentasi

### 4. DatasetAnalyzer

Komponen untuk analisis dataset yang menggunakan fungsionalitas dari utils/dataset.

Tanggung jawab:
- Analisis distribusi kelas
- Analisis ukuran gambar
- Analisis bounding box
- Pelaporan insight tentang karakteristik dataset

## Fitur dan Fungsionalitas

### 1. Validasi Dataset
- Validasi integritas file
- Validasi format label
- Validasi koordinat bounding box
- Pemindahan file tidak valid
- Perbaikan otomatis masalah umum

### 2. Augmentasi Dataset
- Teknik augmentasi: kombinasi, pencahayaan, posisi
- Variasi parameter augmentasi
- Validasi hasil augmentasi
- Resume proses yang terganggu

### 3. Analisis Dataset
- Analisis distribusi kelas dan imbalance
- Analisis ukuran gambar dominan
- Analisis statistik bounding box
- Pelaporan insight

### 4. Integrasi dengan Environment
- Deteksi otomatis Colab
- Mount Google Drive jika di Colab
- Pembuatan symlink untuk integrasi
- Visualisasi struktur direktori

### 5. Pemantauan dan Logging
- Integrasi dengan observer pattern
- Notifikasi event preprocessing
- Progress tracking dengan tqdm
- Logging informatif dengan emoji

## Cara Penggunaan

1. Inisialisasi manager dengan konfigurasi
2. Jalankan pipeline lengkap atau operasi individual
3. Dapatkan hasil sebagai dictionary
4. Analisis hasil dan insight

## Integrasi dengan Komponen Lain

1. **EnvironmentManager**: Untuk mengelola lingkungan runtime
2. **ObserverPattern**: Untuk monitoring dan notifikasi
3. **EnhancedDatasetValidator**: Untuk validasi dataset
4. **AugmentationManager**: Untuk augmentasi dataset

## Lazy Initialization

Komponen menggunakan teknik lazy initialization untuk meminimalkan overhead dan menginisialisasi komponen berat hanya saat diperlukan.

## Observer Pattern

Menggunakan pola observer terkonsolidasi dari utils/observer untuk:
- Monitoring kemajuan operasi
- Notifikasi event preprocessing
- Logging terstruktur
- Integrasi dengan UI dan visualisasi

## Penggunaan Efisien Sumber Daya

- Manajemen memori yang optimal
- Penggunaan multiprocessing untuk operasi yang intensif
- Caching untuk operasi yang berulang