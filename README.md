# 📑 SmartCash - Overview & Petunjuk Penggunaan

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

2. Aktivasi conda environment
   - Buka terminal (Ctrl+`)
   - Jalankan perintah berikut untuk membuat & mengaktifkan conda env:
     ```
     conda create -n smartcash python=3.9
     conda activate smartcash
     ```
   - Install semua dependencies:
     ```
     pip install -r requirements.txt
     ```

3. Persiapkan dataset
   - Jika menggunakan dataset lokal:
     - Download dataset dari Roboflow
     - Ekstrak dataset ke folder `data/` dengan struktur berikut:
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
   - Jika menggunakan Roboflow API:
     - Dapatkan API key dari akun Roboflow Anda
     - Buat file `.env` berdasarkan `.env.example`
     - Set nilai `ROBOFLOW_API_KEY` dengan API key Anda

### Menjalankan Eksperimen
1. Buka notebook `notebooks/smartcash_experiment.ipynb`
2. Jalankan setiap cell secara berurutan
3. Notebook akan secara otomatis:
   - Membaca konfigurasi dari `configs/base_config.yaml`
   - Menginisialisasi model sesuai skenario eksperimen
   - Memuat dataset (lokal atau via API)
   - Melakukan training & evaluasi model
   - Memvisualisasikan hasil eksperimen

## 📊 Hasil Eksperimen
Setelah menjalankan notebook, Anda dapat melihat hasil eksperimen berupa:
- Grafik metrics pelatihan
- Confusion matrix
- Precision-Recall curve
- Perbandingan inference time
- Analisis performa per kelas

Semua hasil akan disimpan di folder `results/` untuk analisis lebih lanjut.

## 🧪 Pengujian Proyek SmartCash

### 📋 Struktur Pengujian

Direktori pengujian ini berisi serangkaian tes unit dan integrasi untuk proyek SmartCash:

- `test_data_handlers.py`: Pengujian untuk handler data
- `test_models.py`: Pengujian komponen model
- `test_evaluation.py`: Pengujian proses evaluasi
- `conftest.py`: Konfigurasi global pytest

### 🚀 Menjalankan Pengujian

#### Perintah Dasar
```bash
# Jalankan semua pengujian
pytest

# Jalankan pengujian spesifik
pytest tests/test_data_handlers.py

# Jalankan tes lambat
pytest --runslow

# Tampilkan informasi tambahan
pytest -v
```

##### 🔍 Marker Pengujian

- `@pytest.mark.slow`: Tes yang membutuhkan waktu lama
- `@pytest.mark.integration`: Tes integrasi

#### 💡 Tips
- Pastikan Anda berada di root project saat menjalankan pengujian
- Gunakan virtual environment untuk menghindari konflik dependencies
- Periksa keluaran pytest untuk detail kesalahan


## 📜 Lisensi
Proyek ini dilisensikan di bawah MIT License.

## 🏆 Sitasi
Jika Anda menggunakan SmartCash dalam penelitian Anda, harap sitasi repository ini:

```
Sabar, A. (2025). Optimasi Deteksi Nominal Mata Uang dengan YOLOv5 dan EfficientNet-B4. (Unpublished)
```

Terima kasih telah menggunakan SmartCash! 🙏 Semoga bermanfaat untuk penelitian Anda. Jika Anda menemukan bug atau ingin request fitur baru, silakan buat Issue di repository ini. Happy detecting! 🪙💰