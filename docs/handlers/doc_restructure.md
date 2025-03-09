# Dokumentasi Restrukturisasi Handler SmartCash

## Latar Belakang
Project SmartCash saat ini memiliki beberapa handler yang cenderung monolitik dengan tanggung jawab yang terlalu luas. Restrukturisasi diperlukan untuk membuat komponen yang lebih atomik, dapat diuji dengan mudah, dan lebih sesuai dengan prinsip SOLID. 

## Tujuan Restrukturisasi
1. Meningkatkan modularitas kode
2. Memperjelas tanggung jawab masing-masing komponen
3. Memudahkan pengujian (unit testing)
4. Memudahkan pemeliharaan dan pengembangan di masa depan
5. Menerapkan pola desain yang sesuai

## Strategi Restrukturisasi

### 1. Penerapan Pola Desain
- **Factory Pattern**: Untuk pembuatan objek dengan konfigurasi yang tepat
- **Strategy Pattern**: Untuk mengisolasi algoritma yang berbeda untuk tugas yang sama
- **Visitor Pattern**: Untuk operasi pada struktur data yang berbeda
- **Adapter Pattern**: Untuk komponen yang perlu berinteraksi dengan sistem eksternal
- **Observer Pattern**: Untuk fitur log dan progress tracking

### 2. Pemisahan Tanggung Jawab
- Memecah handler besar menjadi kelas-kelas kecil dengan tanggung jawab tunggal
- Memisahkan logika bisnis dari infrastruktur
- Memisahkan konfigurasi dari implementasi

### 3. Standarisasi Interface
- Mendefinisikan interface yang jelas untuk setiap komponen
- Memastikan konsistensi error handling
- Standarisasi logging dan progress tracking

## Struktur Direktori yang Diusulkan

```
smartcash/
├── handlers/
│   ├── dataset/               # Pengelolaan data & dataset
│   ├── preprocessing/      # Validasi & augmentasi dataset
│   ├── training/           # Training & manajemen model
│   ├── detection/          # Deteksi & inferensi 
│   ├── evaluation/         # Evaluasi & skenario penelitian
│   └── errors/             # Penanganan error
```

## Urutan Refactor yang Direkomendasikan

### Fase 1: Refaktorisasi Dasar dan Infrastruktur
1. **Refaktor Error Handling**
   - Buat sistem error yang lebih granular
   - Implementasi error factory
   - Standarisasi error handling di seluruh aplikasi

2. **Refaktor Logging System**
   - Pisahkan logger menjadi komponen yang dapat dikomposisi
   - Standarisasi format log dan level logging
   - Tambahkan dukungan untuk berbagai output target

3. **Refaktor Konfigurasi**
   - Pisahkan konfigurasi dari implementasi
   - Buat sistem load konfigurasi yang lebih robust
   - Implementasi validasi konfigurasi

### Fase 2: Refaktorisasi Domain Inti
4. **Refaktor Data Management**
   - Pisahkan `DataManager` menjadi komponen-komponen atomik
   - Buat class terpisah untuk transformasi data
   - Buat abstraksi untuk berbagai sumber data

5. **Refaktor Preprocessing**
   - Refaktor `UnifiedPreprocessingHandler` menjadi komponen yang lebih kecil
   - Implementasi factory untuk berbagai strategi augmentasi
   - Pisahkan validasi dan perbaikan dataset

6. **Refaktor Training**
   - Atomisasi `ModelHandler` menjadi komponen kecil
   - Buat factory untuk optimizer dan scheduler
   - Pisahkan logika training, validasi, dan checkpoint


## Panduan Implementasi
### Dependency Injection
- Semua handler baru harus menerima dependensi mereka melalui konstruktor
- Hindari pembuatan objek secara langsung di dalam method
- Gunakan factory pattern untuk pembuatan objek kompleks

### Error Handling
- Selalu gunakan custom exception yang spesifik
- Jangan menelan exception tanpa logging
- Pastikan pesan error informatif dan membantu debug

### Logging dan Progres
- Gunakan tqdm untuk proses yang panjang
- Pastikan logging tidak terlalu verbos
- Gunakan level logging yang tepat (DEBUG, INFO, WARNING, ERROR)

Dengan mengikuti rencana refaktorisasi ini, code base SmartCash akan menjadi lebih modular, mudah dipelihara, dan sesuai dengan prinsip-prinsip desain software yang baik. Pendekatan bertahap memastikan stabilitas selama proses perubahan.