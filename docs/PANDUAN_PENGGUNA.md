# 📱 Panduan Pengguna SmartCash

## 📋 Daftar Isi
- [Instalasi](#instalasi)
- [Antarmuka CLI](#antarmuka-cli)
- [Pelatihan Model](#pelatihan-model)
- [Evaluasi Model](#evaluasi-model)
- [Troubleshooting](#troubleshooting)

## 🔧 Instalasi

### Persyaratan Sistem
- Python 3.9+
- CUDA-capable GPU (opsional, untuk training lebih cepat)
- Minimal 8GB RAM
- 20GB ruang disk

### Langkah Instalasi
1. Clone repository:
   ```bash
   git clone https://github.com/yourusername/smartcash.git
   cd smartcash
   ```

2. Buat virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # atau
   .\venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Antarmuka CLI

### Memulai Aplikasi
```bash
python run.py
```

### Menu Utama
```
=== SmartCash - Sistem Deteksi Uang Kertas ===

=== Model ===
> Pelatihan Model
  Evaluasi Model

=== Sistem ===
  Keluar

[↑↓] Navigasi  [Enter] Pilih  [q] Keluar
```

### Status Konfigurasi
```
Status Konfigurasi:
  Sumber Data: Dataset Lokal
  Mode Deteksi: Deteksi Lapis Banyak
  Arsitektur: EfficientNet-B4

Parameter Pelatihan:
  Ukuran Batch: 32
  Learning Rate: 0.001
  Jumlah Epoch: 100
  Early Stopping: 10
```

## 🎯 Pelatihan Model

### Menu Pelatihan
```
=== Menu Pelatihan Model ===

=== Konfigurasi ===
  Pilih Sumber Data
  Pilih Mode Deteksi
  Pilih Arsitektur Model
  Konfigurasi Parameter

=== Aksi ===
  Mulai Pelatihan

=== Navigasi ===
  Kembali
```

### Langkah-langkah Pelatihan
1. **Pilih Sumber Data**
   - Dataset Lokal: Gunakan dataset yang tersimpan di folder `data/`
   - Dataset Roboflow: Unduh dataset dari Roboflow (memerlukan API key)

2. **Pilih Mode Deteksi**
   - Deteksi Lapis Tunggal: Hanya deteksi uang kertas
   - Deteksi Lapis Banyak: Deteksi uang kertas, nominal, dan fitur keamanan

3. **Pilih Arsitektur Model**
   - CSPDarknet: Backbone standar YOLOv5
   - EfficientNet-B4: Backbone yang dioptimasi

4. **Konfigurasi Parameter**
   - Ukuran Batch: 8-128 (default: 32)
   - Learning Rate: 0.0001-0.01 (default: 0.001)
   - Jumlah Epoch: 10-1000 (default: 100)
   - Early Stopping: 5-50 (default: 10)

5. **Mulai Pelatihan**
   - Progress pelatihan akan ditampilkan
   - Konfigurasi akan disimpan otomatis
   - Model terbaik akan disimpan di `outputs/`

## 📊 Evaluasi Model

### Menu Evaluasi
```
=== Menu Evaluasi ===

=== Evaluasi ===
  Evaluasi Model Reguler
  Evaluasi Skenario Penelitian

=== Navigasi ===
  Kembali
```

### Jenis Evaluasi
1. **Evaluasi Model Reguler**
   - Evaluasi pada dataset testing standar
   - Menampilkan metrik: Akurasi, Precision, Recall, F1-Score, mAP
   - Waktu inferensi per gambar

2. **Evaluasi Skenario Penelitian**
   - Evaluasi pada berbagai skenario:
     * Kondisi Cahaya Normal
     * Pencahayaan Rendah
     * Pencahayaan Tinggi
     * Rotasi
     * Oklusi Parsial
   - Hasil lengkap disimpan dalam CSV

### Contoh Hasil Evaluasi
```
📊 Model: EfficientNet-B4
  Akurasi: 0.9856
  Precision: 0.9734
  Recall: 0.9812
  F1-Score: 0.9773
  mAP: 0.9845
  Waktu Inferensi: 45.3ms
```

## ❗ Troubleshooting

### Masalah Umum
1. **Aplikasi tidak bisa dijalankan**
   - Pastikan virtual environment aktif
   - Periksa semua dependencies terinstall
   - Periksa versi Python (3.9+)

2. **Dataset tidak ditemukan**
   - Periksa struktur folder `data/`
   - Pastikan file `.env` terkonfigurasi untuk Roboflow

3. **GPU tidak terdeteksi**
   - Periksa instalasi CUDA
   - Pastikan driver GPU up-to-date
   - Periksa torch terinstall dengan CUDA support

4. **Memori tidak cukup**
   - Kurangi ukuran batch
   - Gunakan resolusi input lebih kecil
   - Tutup aplikasi lain yang berat

### Pesan Error
- `"Konfigurasi tidak ditemukan"`: Latih model terlebih dahulu
- `"CUDA out of memory"`: Kurangi ukuran batch
- `"Dataset tidak valid"`: Periksa struktur dataset
- `"API key tidak valid"`: Periksa konfigurasi Roboflow

### Tips
- Gunakan `q` untuk kembali ke menu sebelumnya
- Konfirmasi sebelum keluar aplikasi
- Backup konfigurasi penting di `configs/`
- Monitor penggunaan GPU dengan `nvidia-smi`
