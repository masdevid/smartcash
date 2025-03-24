# Panduan Implementasi Cell SmartCash

## Prinsip Utama
- **Hindari Duplikasi** - Gunakan komponen dan services yang sudah ada
- **Granular File Structure** - Pecah setiap fitur ke file independen (terutama masing-masing ui handler)
- **Selektif Singleton** - Gunakan singleton untuk core services, non-singleton untuk UI handlers
- **Shared Components** - Komponen bersama di `ui/components`,`ui/handlers`,`ui/charts`,`ui/info_boxes`, dan , `ui/reports`
- **Consistent Component Naming** - Penggunaan name komponent yang konsisten seperti log output yang sebelumnya punya banyak variasi name

## Penggunaan Singleton

### Wajib Singleton (Core Services)
- `config_manager` - Manajemen konfigurasi
- `environment_manager` - Deteksi dan setup environment
- `cache_manager` - Manajemen cache
- `progress_tracker` - Tracking progres
- Semua manager di `common/`, `dataset/`, `model/`, dan `detection/`

### Non-Singleton (UI Components)
- Handler spesifik modul
- Komponen UI spesifik modul
- Event handlers di UI

## Struktur Folder Granular

```
ui/
├── cells/                         # Cell sederhana entry point
│   └── cell_x_y_name.py           # Cell sederhana (5-10 baris)
├── module/                        # Modul specific implementasi
│   ├── name_initializer.py        # Main initializer
│   ├── name_component.py          # Definisi UI components
│   ├── name_handlers.py           # Definisi UI handlers awal dengan logic handler yang terpisah
│   ├── handlers/                  # Handler implementations
│   │   ├── button_handlers.py     # Handler untuk tombol UI
│   │   ├── form_handlers.py       # Handler untuk form UI
│   ├── charts/                    # Modul specific chart/visualizations
│   │   └── specific_plot.py       # plot spesifik modul
│   └── components/                # UI Components spesifik modul
│       └── module_components.py   # Komponen spesifik modul
├── components/                    # Shared UI Components
│   └── action_buttons.py          # Shared button components
├── info_boxes/                    # Kumpulan panduan info
│   └── name_info.py               # Konten html info box accordion
├── charts/                        # Shared UI Visualizations
│   ├── plot_base.py               # Base Plot
│   └── plot_x.py                  # Shared Plot
├── reports/                       # Shared UI Visualizations
│   ├── name_stats.py              # Shared Data/Model Stats 
│   └── name_summary.py            # Shared Summary Reports
└── handlers/                      # Shared Event Handlers
    ├── multi_progress.py          # Shared progress handler
    └── observer_handler.py        # Shared observer handler
```

## Implementasi Cell Sederhana

```python
"""
File: smartcash/ui/cells/cell_x_y_name.py
Deskripsi: Entry point untuk [nama] cell
"""

def setup_name():
    """Setup dan tampilkan UI untuk [nama proses]."""
    from smartcash.ui.module.name_initializer import initialize_name_ui
    return initialize_name_ui()

# Eksekusi saat modul diimpor
ui_components = setup_name()
```

## Initializer Contoh

```python
"""
File: smartcash/ui/module/name_initializer.py
Deskripsi: Initializer untuk modul [nama]
"""

def initialize_name_ui():
    """Inisialisasi UI modul [nama]."""
    ui_components = {'module_name': 'name'}
    
    try:
        # Setup environment dan config (gunakan singleton)
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.config import get_config_manager
        env = get_environment_manager()
        config = get_config_manager().config
        
        # Buat komponen UI
        from smartcash.ui.module.name_component import create_name_ui
        ui_components = create_name_ui(env, config)
        
        # Setup logging
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        logger = setup_ipython_logging(ui_components)
        ui_components['logger'] = logger
        
        # Setup multi-progress tracking
        from smartcash.ui.handlers.multi_progress import setup_multi_progress_tracking
        setup_multi_progress_tracking(ui_components, "name", "name_step")
        
        # Setup handlers (non-singleton)
        from smartcash.ui.module.handlers.button_handlers import setup_button_handlers
        setup_button_handlers(ui_components)
        
        # Kaitkan ui_components dengan tombol
        for button_name in ['process_button', 'stop_button', 'reset_button']:
            if button_name in ui_components:
                setattr(ui_components[button_name], 'ui_components', ui_components)
        
    except Exception as e:
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        ui_components = create_fallback_ui(ui_components, str(e), "error")
    
    return ui_components
```

## Penggunaan Services Existing

```python
# BENAR - Menggunakan service existing
from smartcash.dataset.manager import DatasetManager
dataset_manager = DatasetManager(config=config, logger=logger)

from smartcash.dataset.services.preprocessor.dataset_preprocessor import DatasetPreprocessor
preprocessor = DatasetPreprocessor(config=config, logger=logger)

# SALAH - Jangan membuat ulang implementasi
def my_own_preprocessing_logic():  # ❌ Jangan buat ulang
    # Implementasi duplikat...
```

## Best Practices

1. **Pemecahan File** - Satu file untuk satu tanggung jawab
2. **Widget Reference** - Simpan reference ke ui_components di button
3. **Reuse Services** - Gunakan services dari `dataset/`, `model/`, dan `detection/`
4. **Shared Components** - Komponen bersama di `ui/components` dan `ui/handlers`
5. **Selektif Singleton** - Singleton untuk core services, non-singleton untuk UI handlers

## Multi Progress Tracking

```python
# Setup dual progress tracking
from smartcash.ui.handlers.multi_progress import setup_multi_progress_tracking
setup_multi_progress_tracking(
    ui_components,
    overall_tracker_name="preprocessing", 
    step_tracker_name="preprocess_step",
    overall_progress_key="overall_progress",
    step_progress_key="step_progress"
)
```

## Migrasi Dari Cell Lama

1. Buat file cell sederhana dengan 5-10 baris
2. Buat initializer yang menggunakan services existing
3. Pecah handler per jenis (button, form, dll)
4. Buat komponen UI dalam file terpisah
5. Pindahkan komponen bersama ke `ui/components`,`ui/handlers`,`ui/charts`,`ui/info_boxes`, dan , `ui/reports`