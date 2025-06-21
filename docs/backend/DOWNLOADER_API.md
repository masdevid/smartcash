# 📥 SmartCash Dataset Downloader API Documentation

## 🎯 Overview

Dataset downloader yang dirancang untuk mengunduh dataset dari Roboflow dengan fitur-fitur canggih. Dirancang untuk:
- 🔍 Auto-deteksi dataset yang tersedia
- ⚡ Download paralel dengan progress tracking
- 🔄 Dukungan resume download
- 🧹 Pembersihan otomatis file yang tidak perlu
- 📊 Validasi integritas dataset
- 🔒 Manajemen API key yang aman

## 🚀 Main Downloader API

### `get_downloader_instance(config, logger=None)`

Factory utama untuk membuat instance downloader yang kompatibel dengan UI.

```python
from smartcash.dataset.downloader import get_downloader_instance

# Konfigurasi dasar
downloader = get_downloader_instance({
    'api_key': 'your_roboflow_key',
    'workspace': 'your_workspace',
    'project': 'your_project',
    'version': 'latest',
    'target_dir': './datasets',
    'max_workers': 4,
    'timeout': 30,
    'retry_count': 3,
    'chunk_size': 32768  # 32KB chunks
})
```

### `create_download_session(api_key, workspace=None, project=None, version=None, **kwargs)`

Membuat sesi download yang siap digunakan dengan konfigurasi yang diberikan.

```python
from smartcash.dataset.downloader import create_download_session

# Membuat sesi download
session = create_download_session(
    api_key='your_roboflow_key',
    workspace='smartcash',
    project='idr-detection',
    version=3,
    target_dir='./datasets/roboflow'
)

# Menjalankan download
result = session.download()
```

## 🔧 Core Components

### `DownloadService`

Kelas inti yang menangani proses download dengan fitur-fitur:
- Download paralel dengan thread pool
- Progress tracking
- Error handling dan retry mekanisme
- Validasi dataset

```python
from smartcash.dataset.downloader.download_service import DownloadService

# Inisialisasi dengan konfigurasi
service = DownloadService({
    'api_key': 'your_key',
    'workspace': 'smartcash',
    'project': 'idr-detection',
    'version': 3,
    'target_dir': './datasets/roboflow',
    'max_workers': 4
})

# Set progress callback
def progress_callback(stage, current, total, message):
    print(f"{stage}: {current}/{total} - {message}")

service.set_progress_callback(progress_callback)

# Jalankan download
result = service.download()
```

### `RoboflowClient`

Klien untuk berinteraksi dengan Roboflow API.

```python
from smartcash.dataset.downloader.roboflow_client import create_roboflow_client

client = create_roboflow_client(api_key='your_key')
datasets = client.list_datasets(workspace='smartcash')
```

## 🔄 Cleanup Service

Layanan untuk membersihkan file-file yang tidak diperlukan setelah download.

```python
from smartcash.dataset.downloader.cleanup_service import CleanupService

cleaner = CleanupService()
cleaner.cleanup(directory='./datasets/roboflow')
```

## 📊 Progress Tracking

Komponen untuk melacak kemajuan download.

```python
from smartcash.dataset.downloader.progress_tracker import DownloadProgressTracker, DownloadStage

tracker = DownloadProgressTracker()
tracker.start_stage(DownloadStage.DOWNLOADING, total_files=100)

# Update progress
tracker.update(DownloadStage.DOWNLOADING, progress=50, message="Downloading files...")

# Selesaikan stage
tracker.complete_stage(DownloadStage.DOWNLOADING)
```

## 🛠️ Utility Functions

### `get_default_config(api_key='')`

Mendapatkan konfigurasi default untuk downloader.

```python
from smartcash.dataset.downloader import get_default_config

config = get_default_config(api_key='your_key')
```

### `validate_service_compatibility(service)`

Memvalidasi kompatibilitas service dengan versi UI saat ini.

```python
from smartcash.dataset.downloader import validate_service_compatibility, create_download_session

session = create_download_session(api_key='your_key')
is_compatible = validate_service_compatibility(session)
```

## 📝 Contoh Penggunaan Lengkap

```python
from smartcash.dataset.downloader import create_download_session

def download_progress(stage, current, total, message):
    print(f"{stage}: {current}/{total} - {message}")

try:
    # Buat sesi download
    session = create_download_session(
        api_key='your_roboflow_key',
        workspace='smartcash',
        project='idr-detection',
        version=3,
        target_dir='./datasets/roboflow',
        max_workers=4
    )
    
    # Set progress handler
    session.set_progress_callback(download_progress)
    
    # Jalankan download
    result = session.download()
    
    if result['status'] == 'success':
        print(f"✅ Download berhasil! Dataset tersedia di: {result['download_dir']}")
    else:
        print(f"❌ Gagal mendownload: {result.get('message', 'Unknown error')}")
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
```

## 🐛 Troubleshooting

### Error API Key Tidak Valid
1. Pastikan API key yang digunakan benar
2. Periksa apakah API key memiliki akses ke workspace dan project yang dimaksud
3. Coba generate API key baru dari dashboard Roboflow

### Download Terputus
1. Periksa koneksi internet
2. Tingkatkan nilai `timeout` dan `retry_count`
3. Gunakan `chunk_size` yang lebih kecil

### Error Izin
1. Pastikan direktori target memiliki izin tulis
2. Coba jalankan dengan hak akses administrator jika diperlukan

---

Dokumentasi terakhir diperbarui: 21 Juni 2025
