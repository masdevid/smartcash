Refactor `smartcash/ui/dataset/download` init dan handlersnya untuk menerapkan pattern logger, load/save config yang baru dari `smartcash/ui/initializers/common_initializer`. Refaktor implementasi lama konfigurasi dengan memecah menjadi 3 file `handlers/config_extractor`, `handlers/config_updater` dan `handlers/defaults` menyesuaikan implementasi baru pada `CommonInitializer`. 

Ketentuan refaktor:
- Buat implementasi baru modul UI ke dari `smartcash/ui/dataset/download` ke `smartcash/ui/dataset/downloader`. 
- Buat implementasi baru juga untuk modul backendnya dari `smartcash/dataset/services/downloader/**` ke `smartcash/dataset/downloader/**`. Jangan ubah alur bisnisnya.
- Pastikan modul downloder ini menggunakan `dataset_config.yaml` yang inherit `base_config.yaml`. 
- Hapus implementasi observer, dan gantikan dengan progress callback. 
- Progress tracking menunjukkan step-by-step dan overall progress secara informatif tapi tidak membanjiri log_output.
- Perbaiki layout menggunakan flex/grid layout dan hindari memunculkan horizontal scrollbar.
- Pertahankan logika bisnis yang ada, hanya refaktor pattern dan struktur kode. 
- Pecah kode yang lebih dari 500 baris menjadi file-file SRP yang lebih atomic dan reusable. 
- Kurangi redundancy dengan menggunakan implementasi yang ada supaya code tetap DRY. 
- Gunakan one-liner style code. 

Catatan: Tidak semua shared components dan utils diunggah, tapi jika dibutuhkan minta saja. Tunjukkan file mana yang perlu diubah/dihapus. 
Frontend:
```
smartcash/ui/dataset/downloader/
├── __init__.py
├── downloader_initializer.py      # Extends CommonInitializer (done)
├── components/
│   ├── main_ui.py                 # (done)
│   ├── form_fields.py             # (done)   
│   ├── action_buttons.py          # (done)
│   └── progress_display.py        # (missing)
├── handlers/
│   ├── config_extractor.py        # Extract config dari UI (done)
│   ├── config_updater.py          # Update UI dari config (done)
│   ├── defaults.py                # Default values (done)
│   ├── download_handler.py        # Download logic (done)
│   ├── validation_handler.py      # Parameter validation (missing)
│   └── progress_handler.py        # Progress tracking (done)
└── configs/
    └── dataset_config.yaml        # Inherit base_config.yaml (no need to implement)
```
Backend:
```
smartcash/dataset/downloader/
├── __init__.py
├── download_service.py            # Main service (done)
├── roboflow_client.py             # API client (missing)
├── file_processor.py              # File operations (missing)
├── progress_tracker.py            # Progress callbacks (missing)
└── validators.py                  # Data validation (missing)
```