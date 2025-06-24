# Modul Pretrained Model

## Daftar Isi
- [Gambaran Umum](#gambaran-umum)
- [Struktur Direktori](#struktur-direktori)
- [Komponen Utama](#komponen-utama)
- [Alur Kerja](#alur-kerja)
- [Diagram](#diagram)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Gambaran Umum
Modul Pretrained Model menyediakan antarmuka untuk mengunduh, memeriksa, dan menyinkronkan model YOLOv5 dan EfficientNet-B4 yang diperlukan untuk SmartCash. Modul ini menangani seluruh siklus hidup model, mulai dari pengunduhan hingga verifikasi integritas file.

## Struktur Direktori
```
smartcash/ui/pretrained_model/
├── __init__.py
├── pretrained_init.py       # Inisialisasi modul
├── components/             # Komponen UI
│   ├── __init__.py
│   └── ui_components.py    # Komponen antarmuka pengguna
├── handlers/               # Penangan logika bisnis
│   ├── __init__.py
│   ├── check_handler.py    # Pemeriksaan model
│   ├── config_handler.py   # Manajemen konfigurasi
│   ├── download_handler.py # Pengunduhan model
│   ├── pretrained_handlers.py # Handler terpadu
│   ├── reset_handler.py    # Reset konfigurasi
│   └── status_handler.py   # Manajemen status
├── services/               # Layanan backend
│   ├── __init__.py
│   ├── model_checker.py    # Pemeriksa model
│   ├── model_downloader.py # Pengunduh model
│   └── model_syncer.py     # Sinkronisasi model
└── utils/                  # Utilitas
    └── model_utils.py      # Fungsi bantu model
```

## Komponen Utama

### 1. PretrainedInit
- **Lokasi**: `pretrained_init.py`
- **Fungsi**: Inisialisasi modul pretrained model
- **Fitur**:
  - Membuat komponen UI
  - Mengatur handler
  - Menangani siklus hidup modul

### 2. UI Components
- **Lokasi**: `components/ui_components.py`
- **Fitur**:
  - Form konfigurasi model
  - Tombol aksi (download, sync, reset)
  - Panel status dan log
  - Progress tracker

### 3. Handlers
- **Lokasi**: `handlers/`
- **Fitur**:
  - `download_handler.py`: Menangani proses pengunduhan
  - `check_handler.py`: Memeriksa ketersediaan model
  - `status_handler.py`: Mengelola status operasi
  - `reset_handler.py`: Mengatur ulang konfigurasi

### 4. Services
- **Lokasi**: `services/`
- **Fitur**:
  - `model_downloader.py`: Logika pengunduhan model
  - `model_checker.py`: Verifikasi model
  - `model_syncer.py`: Sinkronisasi model

## Alur Kerja

1. **Inisialisasi**
   - Muat konfigurasi
   - Siapkan komponen UI
   - Atur handler interaksi

2. **Pemeriksaan Model**
   - Periksa model yang tersedia
   - Tandai model yang hilang
   - Tampilkan status model

3. **Pengunduhan**
   - Unduh model yang dibutuhkan
   - Tampilkan progres
   - Verifikasi integritas file

4. **Sinkronisasi**
   - Sinkronkan model dengan sistem
   - Update konfigurasi
   - Beri umpan balik

## Diagram

### Class Diagram
```mermaid
classDiagram
    class PretrainedInit {
        +__init__()
        +_create_ui_components()
        +_setup_module_handlers()
    }
    
    class ModelDownloader {
        +download_model()
        +verify_download()
    }
    
    class ModelChecker {
        +check_all_models()
        +verify_model()
    }
    
    class ModelSyncer {
        +sync_models()
        +update_config()
    }
    
    PretrainedInit --> ModelDownloader
    PretrainedInit --> ModelChecker
    PretrainedInit --> ModelSyncer
```

### Sequence Diagram - Download Model
```mermaid
sequenceDiagram
    participant User
    participant UI as PretrainedUI
    participant Handler as DownloadHandler
    participant Service as DownloadService
    
    User->>UI: Klik Download
    UI->>Handler: Execute Download
    Handler->>Service: Check Models
    Service-->>Handler: Missing Models
    Handler->>Service: Download Models
    loop Untuk setiap model
        Service->>Service: Download Chunk
        Service-->>Handler: Update Progress
    end
    Handler-->>UI: Show Result
    UI-->>User: Show Success
```

### Flow Diagram
```mermaid
flowchart TD
    A[Start] --> B[Periksa Model]
    B --> C{Ada Model yang Hilang?}
    C -->|Ya| D[Unduh Model]
    C -->|Tidak| E[Model Sudah Ada]
    D --> F{Verifikasi Model}
    F -->|Gagal| G[Coba Ulang]
    F -->|Berhasil| H[Sinkronkan]
    H --> I[Update Konfigurasi]
    I --> J[Selesai]
    E --> J
    G -->|3x Gagal| K[Error]
```

## Best Practices

1. **Manajemen Unduhan**
   - Gunakan progress tracking
   - Validasi checksum file
   - Dukung resume download

2. **Manajemen Status**
   - Tampilkan status jelas
   - Log setiap operasi
   - Beri umpan balik visual

3. **Error Handling**
   - Tangani error jaringan
   - Beri opsi retry
   - Log error detail

4. **Optimasi**
   - Gunakan threading untuk operasi I/O
   - Batasi ukuran buffer
   - Cache hasil pemeriksaan

## Troubleshooting

### Gagal Mengunduh
1. Periksa koneksi internet
2. Cek ruang disk
3. Verifikasi URL unduhan

### Checksum Tidak Cocok
1. Ulangi unduhan
2. Periksa versi model
3. Hapus file korup

### Izin Ditolak
1. Periksa izin direktori
2. Jalankan sebagai admin
3. Verifikasi kepemilikan file

### Sinkronisasi Gagal
1. Periksa konfigurasi
2. Verifikasi path model
3. Cek log error

---

Dokumentasi terakhir diperbarui: 21 Juni 2025
