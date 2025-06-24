# Panduan Common Initializer 

### 🎯 Entry Point
- `cell_x_x_[module].py`: Entry point minimalis yang tidak perlu diubah
```
"""
File: smartcash/ui/cells/cell_x_x_[module].py
Deskripsi: Entry point untuk cell [module]
"""
from smartcash.ui.[group].[module] import initialize_[module]_ui
initialize_[module]_ui()
```

### 🏗️ Core Initialization
- `[module]_init.py`: Logic inisialisasi utama yang menggunakan `ui_logger.py` sebagai default logger

### 🎨 UI Components
- `ui_form.py`: 
  - Form UI barrels
- `ui_components.py`: 
 - Main UI layout creation
 - Pastikan semua menggunakan flexbox
 - Tidak ada horizontal scrollbar

### 🎛️ Main Orchestrator
`[module]_handler` adalah main orchestrator yang mengatur button handler utama dan shared handler

### ⚙️ Configuration Management
- `config_handler.py`: 
  - Extends `smartcash/ui/components/config_handler.py`
  - Berperan sebagai jembatan antara:
    - `config_extractor.py`
    - `defaults.py`
    - `config_updater.py`

### 🔧 Default Configuration
- `defaults.py`:
  - Berisi konfigurasi default saat reset button diklik
  - Memiliki struktur konfigurasi yang sama persis atau lebih sederhana (sesuai kebutuhan UI) dari `[modul]_config.yaml`

### 🔄 Config Helpers
- `config_extractor.py`: Helper function untuk mengekstrak nilai dari UI yang akan diselaraskan dengan struktur config YAML
- `config_updater.py`: Helper function untuk mengupdate nilai UI dari config YAML

### ♻️ Shared Components and Utils
Gunakan shared component dan utils yang sudah ada di:
- `smartcash/ui/components`
- `smartcash/ui/utils`

**Shared component wajib:**
- `create_header`: Untuk header section dengan judul dan deskripsi
- `create_status_panel`: Untuk menampilkan status operasi
- `create_action_buttons`: Tombol aksi utama (primary, secondary, warning)
- `create_save_reset_buttons`: Tombol simpan dan reset dengan opsi sinkronisasi
- `create_dual_progress_tracker`: Pelacak progres ganda (current/total)
- `create_confirmation_area`: Area konfirmasi untuk operasi kritis
- `create_log_accordion`: Area log yang bisa di-expand

### 📁 Struktur Folder

#### Struktur Umum
```
smartcash/ui/
└── [module_name]/                    # Nama modul (contoh: 'dataset')
    └── [module_name]_initializer.py  # Inisialisasi modul

# Atau dengan group module:
smartcash/
└── [group_name]/                     # Nama group (contoh: 'setup')
    └── [module_name]/                # Nama modul (contoh: 'dependency')
        └── [module_name]_initializer.py
```

#### Contoh: Dependency Installer (dengan group 'setup')
```
smartcash/
├── configs/
│   └── config.yaml              # Template konfigurasi
├── cells/
│   └── cell_1_3_dependency_installer.py  # Entry point
└── ui/
    └── setup/                           # Group module
        └── dependency/                  # Nama module
            ├── __init__.py
            ├── dependency_init.py                # Inisialisasi utama
            ├── components/
            │   ├── __init__.py
            │   ├── ui_forms.py                     # Form UI barrels (opsional. Jika ada multiple form)
            │   └── ui_components.py                # Komponen UI
            ├── handlers/
            │   ├── __init__.py
            │   ├── dependency_handler.py             # Main orchestrator
            │   ├── config_handler.py                 # Handler konfigurasi
            │   ├── defaults.py                       # Nilai default
            │   ├── config_extractor.py               # Ekstraksi nilai dari UI
            │   └── config_updater.py                 # Update UI dari config
            └── utils/
                ├── __init__.py
                ├── state_utils.py                    # Helper state UI
                └── status_utils.py                   # Helper status
```

#### Contoh: Tanpa Group Module
```
smartcash/ui/
└── hyperparameters/                          # Nama module
    ├── __init__.py
    ├── hyperparameters_initializer.py        # Inisialisasi
    ├── components/
    │   ├── __init__.py
    │   └── ui_components.py                  # Komponen UI
    ├── handlers/
    │   ├── __init__.py
    │   ├── hyperparameters_handler.py        # Main orchestrator
    │   ├── config_handler.py                 # Handler konfigurasi
    │   ├── defaults.py                       # Nilai default
    │   ├── config_extractor.py               # Ekstraksi nilai dari UI
    │   └── config_updater.py                 # Update UI dari config
    └── utils/
        ├── __init__.py
        ├── state_utils.py                    # Helper state UI
        └── status_utils.py                   # Helper status
```

## 🛠️ Fungsi Bantuan

```python
def _clear_outputs(ui_components: Dict[str, Any]) -> None:
    """Bersihkan output dan log"""
    if log_output := ui_components.get('log_output'):
        if hasattr(log_output, 'clear_output'):
            log_output.clear_output()

def _log_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info') -> None:
    """Tampilkan pesan ke UI log
    
    Args:
        message: Pesan yang akan ditampilkan
        level: Tingkat keparahan ('info', 'warning', 'error', 'success')
    """
    if log_output := ui_components.get('log_output'):
        if hasattr(log_output, 'append_stdout'):
            log_output.append_stdout(f"[{level.upper()}] {message}\n")

def _disable_buttons(ui_components: Dict[str, Any]) -> None:
    """Nonaktifkan tombol selama operasi berlangsung"""
    for key in ['start_button', 'stop_button', 'save_button', 'reset_button']:
        if button := ui_components.get(key):
            button.disabled = True

def _enable_buttons(ui_components: Dict[str, Any]) -> None:
    """
    Aktifkan kembali tombol setelah operasi selesai
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    button_keys = [k for k in ui_components.keys() if k.endswith('_button')]
    for key in button_keys:
        if button := ui_components.get(key):
            button.disabled = False

def _show_fallback_progress(message: str):
    print(f"📊 {message}")
```

## 🔔 UI Logging & Error Handling

```python
def _log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info"):
    """Log ke UI dengan emoji context"""
    
    if status_panel := ui_components.get('status_panel'):
        status_panel.value = message
    
    if log_output := ui_components.get('log_output'):
        with log_output:
            emoji_map = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
            print(f"{emoji_map.get(level, 'ℹ️')} {message}")
    else:
        print(f"📝 {message}")

def _clear_outputs(ui_components: Dict[str, Any]):
    """Clear semua output widgets"""
    if log_output := ui_components.get('log_output'):
        log_output.clear_output()
    if status_panel := ui_components.get('status_panel'):
        status_panel.value = ""

def _handle_error(ui_components: Dict[str, Any], error_message: str):
    """Comprehensive error handling"""
    _log_to_ui(ui_components, error_message, "error")
    if status_panel := ui_components.get('status_panel'):
        status_panel.value = error_message
    _enable_buttons(ui_components)
    _hide_confirmation_area(ui_components)

def _disable_buttons(ui_components: Dict[str, Any]):
    """Disable action buttons"""
    button_keys = [k for k in ui_components.keys() if k.endswith('_button')]
    for key in button_keys:
        if button := ui_components.get(key):
            button.disabled = True

def _enable_buttons(ui_components: Dict[str, Any]):
    """Enable action buttons"""
    button_keys = [k for k in ui_components.keys() if k.endswith('_button')]
    for key in button_keys:
        if button := ui_components.get(key):
            button.disabled = False
```

## ⚠️ Manajemen Dialog Konfirmasi

```python
def _show_confirmation_area(ui_components: Dict[str, Any]):
    """Tampilkan area konfirmasi"""
    if confirmation_area := ui_components.get('confirmation_area'):
        confirmation_area.layout.display = 'block'

def _hide_confirmation_area(ui_components: Dict[str, Any]):
    """Sembunyikan area konfirmasi"""
    if confirmation_area := ui_components.get('confirmation_area'):
        confirmation_area.layout.display = 'none'

def _show_operation_confirmation(ui_components: Dict[str, Any]):
    """Tampilkan ringkasan konfigurasi untuk konfirmasi"""
    config_handler = ui_components.get('config_handler')
    if config_handler:
        current_config = config_handler.extract_config(ui_components)
        summary = f"Konfigurasi: {len(current_config)} pengaturan"
        _log_to_ui(ui_components, f"📋 Konfirmasi: {summary}", "info")

def _should_execute_operation(ui_components: Dict[str, Any]) -> bool:
    """Periksa apakah operasi harus dieksekusi langsung"""
    confirm_checkbox = ui_components.get('confirm_operation_checkbox')
    return confirm_checkbox and getattr(confirm_checkbox, 'value', False)

def _is_confirmation_pending(ui_components: Dict[str, Any]) -> bool:
    """Periksa apakah ada konfirmasi yang sedang menunggu"""
    confirmation_area = ui_components.get('confirmation_area')
    return confirmation_area and confirmation_area.layout.display != 'none'
```
