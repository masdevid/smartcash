# Modul Dataset Downloader

**Versi Dokumen**: 1.0.0  
**Terakhir Diperbarui**: 4 Juli 2024  
**Kompatibilitas**: SmartCash v1.0.0+

## Daftar Isi
- [Gambaran Umum](#gambaran-umum)
- [Struktur Direktori](#struktur-direktori)
- [Komponen Utama](#komponen-utama)
- [Alur Kerja](#alur-kerja)
- [Diagram](#diagram)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Gambaran Umum
Modul Dataset Downloader menyediakan antarmuka untuk mengunduh dataset dari Roboflow dengan fitur manajemen dataset yang lengkap. Modul ini mendukung konfigurasi koneksi, pengecekan dataset yang sudah ada, dan manajemen ruang penyimpanan.

## Struktur Direktori
```
smartcash/ui/dataset/downloader/
├── __init__.py
├── components/               # Komponen UI
│   ├── __init__.py
│   ├── input_options.py     # Form input dan konfigurasi
│   └── ui_components.py     # Komponen UI utama
├── handlers/                # Penangan logika bisnis
│   ├── __init__.py
│   ├── config_extractor.py  # Ekstraksi konfigurasi
│   ├── config_handler.py    # Handler konfigurasi
│   ├── config_updater.py    # Pembaruan konfigurasi
│   ├── defaults.py          # Nilai default
│   └── download_handler.py  # Handler download
├── utils/                   # Utilitas pendukung
│   ├── backend_utils.py     # Fungsi backend
│   ├── button_manager.py    # Manajemen tombol
│   ├── colab_secrets.py     # Pengelolaan kredensial
│   ├── dialog_utils.py      # Utilitas dialog
│   ├── progress_utils.py    # Utilitas progress
│   ├── ui_utils.py          # Fungsi bantu UI
│   └── validation_utils.py  # Validasi input
└── downloader_initializer.py # Inisialisasi modul
```

## Komponen Utama

### 1. DownloaderInitializer
- **Lokasi**: `downloader_initializer.py`
- **Fungsi**: Inisialisasi modul downloader
- **Fitur**:
  - Membuat komponen UI
  - Mengatur handler
  - Mengelola konfigurasi

### 2. UI Components
- **Lokasi**: `components/`
- **Fitur**:
  - Form konfigurasi koneksi Roboflow
  - Panel status dan progress
  - Tombol aksi (Download, Check, Cleanup)
  - Area konfirmasi operasi

### 3. DownloadHandler
- **Lokasi**: `handlers/download_handler.py`
- **Fungsi**: Menangani logika download
- **Fitur**:
  - Validasi konfigurasi
  - Konfirmasi operasi
  - Manajemen progress
  - Penanganan error

### 4. Utilitas
- **Lokasi**: `utils/`
- **Fitur**:
  - Manajemen kredensial
  - Dialog interaktif
  - Tracking progress
  - Validasi input

## Alur Kerja

1. **Inisialisasi**
   - Memuat konfigurasi yang tersimpan
   - Membuat komponen UI
   - Menyiapkan handler interaksi

2. **Konfigurasi**
   - Masukkan kredensial Roboflow
   - Atur path penyimpanan
   - Konfigurasi opsi download

3. **Validasi**
   - Verifikasi koneksi
   - Cek dataset yang ada
   - Konfirmasi operasi

4. **Eksekusi**
   - Unduh dataset
   - Ekstrak dan validasi
   - Update status

## Diagram

### Class Diagram
```mermaid
classDiagram
    class DownloaderInitializer {
        +__init__()
        +_create_ui_components()
        +_setup_module_handlers()
        +_get_default_config()
    }
    
    class DownloadHandler {
        +setup_download_handlers()
        +setup_download_handler()
        +setup_check_handler()
        +setup_cleanup_handler()
    }
    
    class UIComponents {
        +progress_callback
        +config_handler
        +logger
        +status_panel
        +progress_tracker
    }
    
    DownloaderInitializer --> DownloadHandler
    DownloaderInitializer --> UIComponents
    DownloadHandler --> UIComponents
```

### Sequence Diagram - Proses Download
```mermaid
sequenceDiagram
    participant User
    participant UI as DownloaderUI
    participant Handler as DownloadHandler
    participant Backend as RoboflowAPI
    
    User->>UI: Masukkan kredensial
    UI->>Handler: Validasi input
    Handler->>Backend: Cek koneksi
    Backend-->>Handler: Respon koneksi
    
    User->>UI: Klik "Download"
    UI->>Handler: Validasi konfigurasi
    Handler->>UI: Tampilkan konfirmasi
    
    User->>UI: Konfirmasi download
    UI->>Handler: Eksekusi download
    Handler->>Backend: Request download
    Backend-->>Handler: Stream data
    Handler-->>UI: Update progress
    UI-->>User: Tampilkan hasil
```

### Flow Diagram
```mermaid
flowchart TD
    A[Start] --> B[Load Konfigurasi]
    B --> C[Tampilkan Form]
    
    C --> D{Input Pengguna}
    D -->|Ubah Konfigurasi| E[Validasi Input]
    D -->|Check Dataset| F[Cek Dataset]
    D -->|Download| G[Validasi Koneksi]
    
    E --> C
    F --> C
    
    G --> H{Dataset Ada?}
    H -->|Ya| I[Tampilkan Konfirmasi]
    H -->|Tidak| J[Proses Download]
    
    I -->|Konfirmasi| J
    I -->|Batal| C
    
    J --> K[Download & Ekstrak]
    K --> L[Validasi Dataset]
    L --> M[Update Status]
    M --> C
```

## Best Practices

1. **Manajemen Koneksi**
   - Simpan kredensial dengan aman
   - Validasi koneksi sebelum download
   - Handle timeout dengan baik

2. **Manajemen Penyimpanan**
   - Cek ruang disk sebelum download
   - Bersihkan file sementara
   - Backup konfigurasi

3. **Feedback Pengguna**
   - Tampilkan progress real-time
   - Berikan pesan error yang jelas
   - Konfirmasi operasi kritis

4. **Performansi**
   - Gunakan streaming untuk file besar
   - Batasi penggunaan memori
   - Optimalkan kecepatan download

## Troubleshooting

### Koneksi Gagal
1. Periksa koneksi internet
2. Verifikasi kredensial Roboflow
3. Cek firewall/proxy

### Download Gagal
1. Periksa ruang disk
2. Verifikasi izin akses
3. Cek log error

### Dataset Tidak Valid
1. Verifikasi format dataset
2. Periksa integritas file
3. Cek versi dataset

### Performa Lambat
1. Periksa kecepatan internet
2. Kurangi ukuran batch
3. Nonaktifkan fitur tidak perlu

---

Dokumentasi terakhir diperbarui: 21 Juni 2025
