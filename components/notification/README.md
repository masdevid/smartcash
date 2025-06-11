# Modul Notifikasi SmartCash

Modul ini menyediakan sistem notifikasi terpadu untuk aplikasi SmartCash dengan integrasi observer pattern dan progress tracking.

## Fitur Utama

- **Notifikasi Proses**: Mengirimkan notifikasi awal, progress, dan penyelesaian proses
- **Penanganan Error**: Menangani dan melaporkan error dengan cara yang konsisten
- **Integrasi Observer**: Terintegrasi dengan sistem observer untuk komunikasi antar komponen
- **Fallback Mechanism**: Menyediakan fallback ke metode notifikasi lama jika diperlukan
- **Thread-Safe**: Menggunakan lock untuk mencegah race condition pada notifikasi
- **Defensive Programming**: Pengecekan defensif untuk mencegah error saat komponen UI tidak tersedia

## Penggunaan

### Inisialisasi NotificationManager

```python
from smartcash.components.notification import get_notification_manager

# Inisialisasi dengan UI components
notification_manager = get_notification_manager(ui_components)
```

### Notifikasi Proses

```python
# Notifikasi awal proses
notification_manager.notify_process_start(
    "download", 
    "Memulai proses download dataset",
    workspace="my-workspace",
    project="my-project"
)

# Update progress
notification_manager.update_progress(
    "download",
    progress=50,
    total=100,
    message="Mendownload file..."
)

# Notifikasi penyelesaian proses
notification_manager.notify_process_complete(
    "download",
    "Download selesai",
    stats={"total_files": 100}
)

# Notifikasi error
notification_manager.notify_process_error(
    "download",
    "Koneksi terputus",
    exception=e
)
```

### Update Status

```python
# Update status panel
notification_manager.update_status(
    "Menunggu input user...",
    status_type="info"
)
```

## Integrasi dengan Service

Untuk mengintegrasikan NotificationManager dengan service yang sudah ada:

1. Tambahkan inisialisasi NotificationManager di `__init__`
2. Gunakan metode NotificationManager untuk notifikasi proses
3. Tambahkan fallback ke metode notifikasi lama jika NotificationManager tidak tersedia

Contoh:

```python
def __init__(self, ui_components):
    self.ui_components = ui_components
    
    # Inisialisasi notification manager
    self._notification_manager = None
    try:
        self._notification_manager = get_notification_manager(ui_components)
    except Exception:
        pass  # Fallback ke metode lama
        
def process_data(self, params):
    try:
        # Notifikasi awal
        if self._notification_manager:
            self._notification_manager.notify_process_start("process", "Memulai proses")
            
        # Proses data
        result = self._do_process(params)
        
        # Notifikasi selesai
        if self._notification_manager:
            self._notification_manager.notify_process_complete("process", "Proses selesai")
            
        return result
        
    except Exception as e:
        # Notifikasi error
        if self._notification_manager:
            self._notification_manager.notify_process_error("process", str(e), exception=e)
        raise
```

## Tipe Status

NotificationManager mendukung beberapa tipe status:

- `info`: Informasi umum (biru)
- `warning`: Peringatan (kuning)
- `error`: Error (merah)
- `success`: Sukses (hijau)

## Integrasi dengan Observer Pattern

NotificationManager terintegrasi dengan sistem observer untuk memastikan konsistensi notifikasi di seluruh aplikasi. Notifikasi akan dikirim melalui:

1. Observer Manager (jika tersedia)
2. EventDispatcher (fallback)
3. Fungsi notify global (fallback terakhir)

## Defensive Programming

NotificationManager menggunakan pengecekan defensif untuk mencegah error saat komponen UI tidak tersedia:

```python
if 'progress_bar' in self.ui_components and hasattr(self.ui_components['progress_bar'], 'value'):
    self.ui_components['progress_bar'].value = progress
```
