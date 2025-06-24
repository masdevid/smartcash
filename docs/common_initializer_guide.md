# Panduan Common Initializer

## 📁 Struktur File Modul

```
[module]/
├── __init__.py
├── [module]_initializer.py          # Entry point modul
├── components/
│   ├── __init__.py
│   ├── ui_components.py            # Komponen UI utama
│   └── input_options.py            # Form input
├── handlers/
│   ├── __init__.py
│   ├── config_handler.py           # Handler konfigurasi
│   ├── config_extractor.py         # Logika ekstraksi konfigurasi
│   ├── config_updater.py           # Pembaruan UI dari konfigurasi
│   ├── [module]_handlers.py        # Handler operasi
│   └── defaults.py                 # Nilai default dan konstanta
└── utils/
    ├── __init__.py
    └── ui_utils.py                 # Fungsi bantuan UI
```

## 🚀 Implementasi Initializer

### 1. Kelas Initializer Utama
```python
# File: [module]/[module]_initializer.py
from typing import Dict, Any
from smartcash.ui.initializers.common_initializer import CommonInitializer
from .handlers.[module]_handlers import setup_[module]_handlers
from .handlers.config_handler import [Module]ConfigHandler

class [Module]Initializer(CommonInitializer):
    """Inisialisasi modul dengan integrasi API dan dukungan dialog"""
    
    def __init__(self):
        super().__init__(
            module_name='[module]',
            config_handler_class=[Module]ConfigHandler,
            parent_module='parent_module'  # Contoh: 'dataset' untuk modul dataset
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat dan validasi komponen UI"""
        from .components.ui_components import create_[module]_ui
        ui_components = create_[module]_ui(config or {})
        
        # Validasi komponen kritis
        required_widgets = ['main_button', 'status_panel', 'log_output']
        missing = [w for w in required_widgets if w not in ui_components]
        if missing:
            raise ValueError(f"Komponen UI yang diperlukan tidak ditemukan: {', '.join(missing)}")
        
        # Tambahkan metadata modul
        ui_components.update({
            'module_name': '[module]',
            'api_integration_enabled': True,
            'dialog_components_loaded': True,
            'progress_tracking_enabled': True
        })
        
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], 
                             config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Siapkan handler dengan pengecekan environment"""
        # Pengaturan khusus environment
        if env == 'colab':
            ui_components['drive_enabled'] = True
            
        return setup_[module]_handlers(ui_components, config)
    
    def _post_initialization_hook(self, ui_components: Dict[str, Any], 
                               config: Dict[str, Any], env=None, **kwargs) -> None:
        """Hook yang dipanggil setelah inisialisasi selesai"""
        # Muat konfigurasi ke UI jika ada
        config_handler = ui_components.get('config_handler')
        if config_handler and config:
            config_handler.update_ui(ui_components, config)
            self._log_to_ui(ui_components, "✅ Konfigurasi berhasil dimuat", "success")
        
        # Setup callback tambahan jika diperlukan
        self._setup_auto_save_callbacks(ui_components)

# Fungsi entry point
def initialize_[module]_ui(config=None, env=None, **kwargs):
    """
    Inisialisasi UI modul dengan integrasi API.
    
    Args:
        config: Konfigurasi awal (opsional)
        env: Informasi environment (contoh: 'colab')
        **kwargs: Parameter tambahan
        
    Returns:
        Dictionary berisi komponen UI
    """
    return _[module]_initializer.initialize(config=config, env=env, **kwargs)

# Buat instance global
_[module]_initializer = [Module]Initializer()

## 🧩 Implementasi Komponen UI

### 1. Komponen UI Utama
```python
# File: [module]/components/ui_components.py
"""Komponen UI untuk [Module] dengan Integrasi API"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_[module]_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Buat komponen UI utama dengan integrasi API"""
    # Impor komponen shared
    from smartcash.ui.components import (
        create_header, create_action_buttons, create_status_panel,
        create_log_accordion, create_save_reset_buttons, create_dual_progress_tracker
    )
    from .input_options import create_[module]_input_options
    
    # Buat komponen UI
    ui_components = {
        # Header dan status
        'header': create_header(
            "🔧 Judul Modul", 
            "Deskripsi modul dengan integrasi API",
            "🚀"
        ),
        'status_panel': create_status_panel("Siap memulai", "info"),
        
        # Form input
        'input_form': create_[module]_input_options(config or {}),
        
        # Tombol aksi
        'action_buttons': create_action_buttons(
            start_label="▶️ Mulai Proses",
            stop_label="⏹️ Berhenti",
            disabled=False
        ),
        
        # Pelacak progres
        'progress_tracker': create_dual_progress_tracker(
            current_label="Saat ini:", 
            total_label="Total:",
            description="Progres:"
        ),
        
        # Logging
        'log_output': create_log_accordion("Catatan Proses")
    }
    
    # Tambahkan tombol simpan/reset
    ui_components.update(create_save_reset_buttons())
    
    return ui_components
```

### 2. Komponen Input Form
```python
# File: [module]/components/input_options.py
"""Komponen form input untuk [Module]"""

import ipywidgets as widgets
from typing import Dict, Any

def create_[module]_input_options(config: Dict[str, Any]) -> widgets.Widget:
    """Buat form input untuk konfigurasi [Module]"""
    # Buat widget input
    input_widgets = {
        'setting1': widgets.Text(
            value=config.get('setting1', ''),
            description='Pengaturan 1:',
            style={'description_width': 'initial'}
        ),
        'enabled': widgets.Checkbox(
            value=config.get('enabled', True),
            description='Aktifkan fitur',
            style={'description_width': 'initial'}
        )
    }
    
    # Susun dalam layout vertikal
    form_items = [
        widgets.HTML(value='<h3>Konfigurasi [Module]</h3>'),
        *input_widgets.values()
    ]
    
    # Tambahkan ke container
    form = widgets.VBox(form_items, layout={'width': '100%'})
    
    # Simpan referensi ke widget untuk akses nanti
    form.input_widgets = input_widgets
    
    return form
```
### 3. Implementasi Config Handler
```python
# File: [module]/handlers/config_handler.py
"""Handler konfigurasi untuk [Module]"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler

class [Module]ConfigHandler(ConfigHandler):
    """Menangani manajemen konfigurasi untuk [Module]"""
    
    def __init__(self):
        super().__init__(
            module_name='[module]',
            parent_module='modul_induk'  # Contoh: 'dataset'
        )
        
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Ekstrak konfigurasi dari komponen UI"""
        from .config_extractor import extract_[module]_config
        return extract_[module]_config(ui_components)
        
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Perbarui komponen UI dari konfigurasi"""
        from .config_updater import update_[module]_ui
        update_[module]_ui(ui_components, config)
```

### 4. Implementasi Defaults dan Ekstraktor

#### 4.1 File Defaults
```python
# File: [module]/handlers/defaults.py
"""Nilai default untuk konfigurasi [Module]"""

MODULE_DEFAULTS = {
    'setting1': 'default_value1',
    'setting2': 'default_value2',
    'enabled': True
}

def get_default_[module]_config() -> Dict[str, Any]:
    """Dapatkan konfigurasi default untuk modul"""
    return {
        'module_name': '[module]',
        'version': '1.0.0',
        'settings': MODULE_DEFAULTS.copy()
    }
```

#### 4.2 File Ekstraktor Konfigurasi
```python
# File: [module]/handlers/config_extractor.py
"""Ekstraksi konfigurasi dari komponen UI"""

from typing import Dict, Any

def extract_[module]_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ekstrak konfigurasi dari komponen UI
    
    Args:
        ui_components: Dictionary berisi komponen UI
        
    Returns:
        Dictionary konfigurasi yang diekstrak
    """
    from .defaults import get_default_[module]_config
    
    config = get_default_[module]_config()
    
    # Contoh ekstraksi nilai dari komponen UI
    if 'setting1_input' in ui_components:
        config['settings']['setting1'] = ui_components['setting1_input'].value
    
    if 'enabled_checkbox' in ui_components:
        config['settings']['enabled'] = ui_components['enabled_checkbox'].value
        
    return config
```

#### 4.3 File Pembaruan UI
```python
# File: [module]/handlers/config_updater.py
"""Pembaruan UI dari konfigurasi"""

from typing import Dict, Any

def update_[module]_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Perbarui komponen UI dari konfigurasi
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi yang akan diterapkan
    """
    settings = config.get('settings', {})
    
    # Contoh pembaruan komponen UI
    if 'setting1_input' in ui_components and 'setting1' in settings:
        ui_components['setting1_input'].value = settings['setting1']
    
    if 'enabled_checkbox' in ui_components and 'enabled' in settings:
        ui_components['enabled_checkbox'].value = settings['enabled']
```

## 🎯 Pola Handler dan Alur Operasi

### 1. Handler Konfigurasi (Simpan/Reset)
```python
def _setup_config_handlers(ui_components: Dict[str, Any]) -> None:
    """
    Siapkan handler untuk tombol simpan dan reset dengan umpan balik UI
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    def save_config(button=None):
        """Handler untuk tombol simpan"""
        _clear_outputs(ui_components)
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                _log_to_ui(ui_components, "❌ Handler konfigurasi tidak tersedia", "error")
                return
            
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            config_handler.save_config(ui_components)
            _log_to_ui(ui_components, "✅ Konfigurasi berhasil disimpan", "success")
            
        except Exception as e:
            _log_to_ui(ui_components, f"❌ Gagal menyimpan konfigurasi: {str(e)}", "error")
    
    def reset_config(button=None):
        """Handler untuk tombol reset"""
        _clear_outputs(ui_components)
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                _log_to_ui(ui_components, "❌ Handler konfigurasi tidak tersedia", "error")
                return
            
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            config_handler.reset_config(ui_components)
            _log_to_ui(ui_components, "✅ Konfigurasi direset ke nilai default", "success")
            
        except Exception as e:
            _log_to_ui(ui_components, f"❌ Gagal mereset konfigurasi: {str(e)}", "error")
    
    # Bind handlers ke tombol
    if save_button := ui_components.get('save_button'):
        save_button.on_click(save_config)
    if reset_button := ui_components.get('reset_button'):
        reset_button.on_click(reset_config)
```

### 2. Handler Operasi dengan Konfirmasi
```python
def _setup_operation_handlers(ui_components: Dict[str, Any]) -> None:
    """
    Siapkan handler untuk operasi yang membutuhkan konfirmasi
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    def operation_handler(button=None):
        """Handler untuk tombol operasi"""
        return _handle_operation_with_confirmation(ui_components)
    
    if operation_button := ui_components.get('operation_button'):
        operation_button.on_click(operation_handler)

def _handle_operation_with_confirmation(ui_components: Dict[str, Any]) -> bool:
    """
    Tangani operasi dengan alur kerja konfirmasi
    
    Returns:
        bool: True jika operasi berhasil diproses, False jika terjadi error
    """
    try:
        _clear_outputs(ui_components)
        
        # Langsung eksekusi jika tidak membutuhkan konfirmasi
        if _should_execute_operation(ui_components):
            return _execute_operation_with_progress(ui_components)
        
        # Tampilkan area konfirmasi jika belum ditampilkan
        if not _is_confirmation_pending(ui_components):
            _show_confirmation_area(ui_components)
            _log_to_ui(ui_components, "⏳ Menunggu konfirmasi...", "info")
            _show_operation_confirmation(ui_components)
        
        return True
        
    except Exception as e:
        _handle_error(ui_components, f"❌ Terjadi kesalahan saat memproses operasi: {str(e)}")
        return False
```

### 3. Integrasi Pelacak Progres
```python
def _execute_operation_with_progress(ui_components: Dict[str, Any]) -> bool:
    """
    Eksekusi operasi dengan pelacakan progres
    
    Returns:
        bool: True jika operasi berhasil diselesaikan
    """
    try:
        _disable_buttons(ui_components)
        
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'start'):
            progress_tracker.start("🚀 Memulai operasi...")
        else:
            _show_fallback_progress("🚀 Memulai operasi...")
        
        # Contoh operasi dengan beberapa langkah
        total_steps = 5
        for step in range(total_steps):
            # Update progress tracker jika tersedia
            if progress_tracker and hasattr(progress_tracker, 'update'):
                progress_tracker.update(
                    f"📋 Langkah {step+1}...", 
                    step + 1, 
                    total_steps
                )
            else:
                _show_fallback_progress(f"📋 Langkah {step+1}/{total_steps}")
            
            # Eksekusi langkah operasi
            _perform_operation_step(step, ui_components)
        
        # Tandai operasi selesai
        if progress_tracker and hasattr(progress_tracker, 'complete'):
            progress_tracker.complete("✅ Operasi selesai")
        else:
            _show_fallback_progress("✅ Operasi selesai")
            
        _enable_buttons(ui_components)
        return True
        
    except Exception as e:
        _handle_error(ui_components, f"❌ Gagal mengeksekusi operasi: {str(e)}")
        return False
```

## 🔄 Alur Operasi Standar

1. **Inisialisasi**
   - Muat konfigurasi default
   - Siapkan komponen UI
   - Daftarkan handler untuk semua interaksi pengguna

2. **Konfigurasi**
   - Pengguna menyesuaikan setelan
   - Konfigurasi dapat disimpan atau direset ke default

3. **Eksekusi**
   - Pengguna memulai operasi
   - Sistem memvalidasi input
   - Tampilkan konfirmasi jika diperlukan
   - Jalankan operasi dengan umpan balik progres
   - Tampilkan hasil atau pesan error

4. **Penyelesaian**
   - Reset UI ke keadaan semula
   - Aktifkan kembali tombol yang dinonaktifkan
   - Tampilkan ringkasan operasi

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

## 📋 Konvensi Penamaan Widget

```python
# Pola penamaan widget yang konsisten
konvensi_widget = {
    'input': 'tujuan_input',
    'dropdown': 'tujuan_dropdown', 
    'checkbox': 'tujuan_checkbox',
    'button': 'tujuan_tombol',
    'output': 'tujuan_output',
    'panel': 'tujuan_panel',
    'area': 'tujuan_area'
}

# Contoh penggunaan:
# - nama_file_input
# - format_dropdown
# - aktifkan_checkbox
# - simpan_tombol
# - log_output
# - status_panel
# - konfirmasi_area