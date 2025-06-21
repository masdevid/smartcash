# Modul Manajemen Dependency

## Daftar Isi
- [Gambaran Umum](#gambaran-umum)
- [Struktur Direktori](#struktur-direktori)
- [Komponen Utama](#komponen-utama)
- [Alur Kerja](#alur-kerja)
- [Diagram Alir](#diagram-alir)
- [Penggunaan](#penggunaan)
- [Best Practices](#best-practices)

## Gambaran Umum
Modul manajemen dependency menyediakan antarmuka pengguna untuk mengelola package Python yang dibutuhkan oleh aplikasi SmartCash. Modul ini memungkinkan pengguna untuk menganalisis, menginstal, dan memantau status package dengan fitur pelacakan progres dan pencatatan yang komprehensif.

## Struktur Direktori
```
ui/setup/dependency/
├── __init__.py
├── components/           # Komponen UI
│   ├── __init__.py
│   ├── package_selector.py
│   └── ui_components.py
├── handlers/            # Penangan logika bisnis
│   ├── __init__.py
│   ├── analysis_handler.py
│   ├── config_handler.py
│   └── installation_handler.py
└── utils/               # Utilitas pendukung
    ├── __init__.py
    ├── package_utils.py
    └── system_info_utils.py
```

## Komponen Utama

### 1. Inisialisasi (`dependency_init.py`)
- Titik masuk utama sistem manajemen dependency
- Menginisialisasi antarmuka pengguna
- Mengelola konfigurasi
- Menyediakan API publik untuk modul lain

### 2. Komponen UI (`components/`)
- **package_selector.py**: Tampilan grid untuk memilih package
- **ui_components.py**: Tampilan utama termasuk:
  - Grid pemilihan package
  - Input package kustom
  - Tombol aksi (Install, Analyze, Check)
  - Pelacak progres
  - Tampilan log

### 3. Handler (`handlers/`)
- **analysis_handler.py**: Menganalisis dependency package
- **config_handler.py**: Mengelola penyimpanan konfigurasi
- **installation_handler.py**: Menangani instalasi package
- **status_check_handler.py**: Memeriksa status package

### 4. Utilitas (`utils/`)
- **package_utils.py**: Fungsi manajemen package
- **system_info_utils.py**: Pengumpulan informasi sistem
- **report_generator_utils.py**: Pembuatan laporan

## Alur Kerja

1. **Inisialisasi**
   - Memuat konfigurasi yang tersimpan
   - Menyiapkan antarmuka pengguna
   - Memeriksa status sistem

2. **Analisis**
   - Pengguna memilih package
   - Sistem menganalisis dependency
   - Menghasilkan laporan kompatibilitas

3. **Instalasi**
   - Persiapan lingkungan
   - Proses instalasi package
   - Verifikasi instalasi
   - Pencatatan hasil

4. **Pemeriksaan**
   - Memeriksa versi package
   - Memvalidasi dependency
   - Menghasilkan status sistem

## Diagram Alir

### Class Diagram
```mermaid
classDiagram
    class DependencyInitializer {
        +__init__()
        +_create_ui_components()
        +_setup_handlers()
        +_setup_default_config()
        +_setup_critical_components()
        +get_config()
        +update_config()
        +cleanup()
    }
    
    class DependencyConfigHandler {
        +__init__(module_name, parent_module)
        +load_config()
        +save_config()
        +update_ui()
    }
    
    class PackageSelector {
        +get_selected_packages()
        +reset_selections()
        +update_package_status()
    }
    
    class InstallationHandler {
        +setup_handler()
        +_install_packages_parallel()
        +_handle_installation_result()
    }
    
    class AnalysisHandler {
        +setup_handler()
        +_analyze_dependencies()
        +_generate_report()
    }
    
    class StatusCheckHandler {
        +setup_handler()
        +_check_package_status()
        +_update_ui_status()
    }
    
    class ProgressTracker {
        +update_overall()
        +update_current()
        +complete()
        +error()
    }
    
    class ButtonStateManager {
        +__init__(ui_components)
        +operation_context(operation_name)
    }
    
    DependencyInitializer --> DependencyConfigHandler
    DependencyInitializer --> PackageSelector
    DependencyInitializer --> InstallationHandler
    DependencyInitializer --> AnalysisHandler
    DependencyInitializer --> StatusCheckHandler
    DependencyInitializer --> ProgressTracker
    DependencyInitializer --> ButtonStateManager
    
    InstallationHandler --> ProgressTracker
    AnalysisHandler --> ProgressTracker
    StatusCheckHandler --> ProgressTracker
    
    PackageSelector --> DependencyConfigHandler
    InstallationHandler --> DependencyConfigHandler
    AnalysisHandler --> DependencyConfigHandler
    StatusCheckHandler --> DependencyConfigHandler
```

### Diagram Alir Utama
```mermaid
graph TD
    A[Mulai] --> B[Inisialisasi UI]
    B --> C[Load Konfigurasi]
    C --> D[Tampilkan Daftar Package]
    D --> E{Input Pengguna}
    E -->|Analisis| F[Proses Analisis]
    E -->|Install| G[Proses Instalasi]
    E -->|Periksa| H[Periksa Status]
    F --> I[Tampilkan Hasil]
    G --> I
    H --> I
    I --> D
```

### Diagram Aktivitas
```mermaid
stateDiagram-v2
    [*] --> Menunggu
    Menunggu --> Menganalisis: Klik Analisis
    Menunggu --> Menginstal: Klik Install
    Menunggu --> Memeriksa: Klik Periksa
    
    state Menganalisis {
        [*] --> KumpulkanPackage
        KumpulkanPackage --> AnalisisDependency
        AnalisisDependency --> HasilkanLaporan
        HasilkanLaporan --> UpdateUI
        UpdateUI --> [*]
    }
    
    state Menginstal {
        [*] --> SiapkanInstalasi
        SiapkanInstalasi --> ProsesInstalasi
        ProsesInstalasi --> Verifikasi
        Verifikasi --> [*]
    }
    
    state Memeriksa {
        [*] --> PeriksaVersi
        PeriksaVersi --> ValidasiDependency
        ValidasiDependency --> UpdateStatus
        UpdateStatus --> [*]
    }
    
    Menganalisis --> Menunggu: Selesai
    Menginstal --> Menunggu: Selesai
    Memeriksa --> Menunggu: Selesai
```

## Best Practices

1. **Manajemen Error**
   - Selalu tangkap dan catat eksepsi
   - Berikan pesan error yang informatif
   - Kembalikan ke state yang stabil setelah error

2. **Kinerja**
   - Gunakan operasi asinkron untuk tugas berat
   - Batasi pembaruan UI yang tidak perlu
   - Cache hasil yang sering digunakan

3. **Keamanan**
   - Validasi semua input pengguna
   - Gunakan environment yang terisolasi untuk instalasi
   - Batasi izin akses file

4. **Pemeliharaan**
   - Dokumentasikan semua fungsi publik
   - Gunakan tipe hint untuk kejelasan
   - Tulis unit test untuk logika bisnis

---

Dokumentasi terakhir diperbarui: 21 Juni 2025
