# 🏗️ Panduan Pola UI Module SmartCash

## Struktur Wajib UI Module dengan CommonInitializer

### 📁 Struktur Direktori
```
smartcash/ui/[domain]/[module]/
├── __init__.py                    # Ekspor initializer
├── [module]_initializer.py        # Kelas initializer utama
├── handlers/
│   ├── __init__.py
│   ├── config_handler.py         # Manajemen konfigurasi
│   ├── config_extractor.py       # UI → Konfigurasi
│   ├── config_updater.py         # Konfigurasi → UI
│   ├── defaults.py               # Nilai default hardcoded
│   └── [module]_handlers.py      # Handler logika bisnis (Jika terlalu panjang, pecah jadi SRP handler)
├── components/
│   ├── __init__.py
│   ├── ui_components.py          # Penyusun antarmuka utama
│   └── input_options.py          # Komponen form
└── utils/
    ├── __init__.py
    ├── ui_utils.py               # Utilitas tampilan UI
    ├── button_manager.py         # Manajemen status tombol (HARUS disable semua tombol saat process)
    ├── dialog_utils.py           # Dialog konfirmasi (Opsional)
    ├── progress_utils.py         # Pelacakan kemajuan
    └── backend_utils.py          # Integrasi backend (Opsional)
```

## 🎯 Pola UI Terkini (Modul Augmentasi)

### 1. Pola Area Konfirmasi

#### 1.1. Penggunaan Dasar
```python
from smartcash.ui.dataset.augmentation.utils.dialog_utils import (
    show_confirmation_in_area,
    show_info_in_area,
    show_warning_in_area,
    clear_confirmation_area
)

# Menampilkan dialog konfirmasi
show_confirmation_in_area(
    ui_components,
    title="Konfirmasi",
    message="Apakah Anda yakin ingin melanjutkan?",
    on_confirm=lambda b: print("Dikonfirmasi"),
    on_cancel=lambda b: print("Dibatalkan"),
    confirm_text="Ya",
    cancel_text="Tidak",
    danger_mode=True
)

# Menampilkan pesan info
show_info_in_area(
    ui_components,
    title="Informasi",
    message="Proses telah selesai",
    on_close=lambda b: print("Tutup")
)

# Membersihkan area konfirmasi
clear_confirmation_area(ui_components)
```

#### 1.2. Praktik Terbaik
- Selalu gunakan `clear_confirmation_area` sebelum menampilkan dialog baru
- Gunakan `danger_mode=True` untuk aksi yang berisiko
- Sediakan fungsi callback untuk `on_confirm` dan `on_cancel`
- Gunakan `on_close` untuk membersihkan sumber daya setelah dialog ditutup

### 2. Pola Pencatatan Log

#### 2.1. Penggunaan Dasar
```python
from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui

# Mencatat log dengan level berbeda
log_to_ui(ui_components, "Pesan informasi", level='info')
log_to_ui(ui_components, "Pesan sukses", level='success')
log_to_ui(ui_components, "Pesan peringatan", level='warning')
log_to_ui(ui_components, "Pesan kesalahan", level='error')
log_to_ui(ui_components, "Pesan debug", level='debug')


#### 2.2. Praktik Terbaik
- Gunakan level yang sesuai untuk setiap jenis pesan
- Sertakan konteks yang cukup dalam pesan log
- Hindari pencatatan log berlebihan yang dapat membanjiri antarmuka pengguna

### 3. Pelacakan Kemajuan

#### 3.1. Manajemen Status Tombol
```python
from smartcash.ui.dataset.augmentation.utils.button_manager import (
    disable_all_buttons,
    enable_all_buttons,
    set_button_processing_state
)

# Menonaktifkan semua tombol saat operasi berjalan
disable_all_buttons(ui_components)

# Mengatur tombol ke status 'sedang diproses'
set_button_processing_state(ui_components, 'augment_button', processing=True)

# Mengaktifkan kembali tombol setelah selesai
enable_all_buttons(ui_components)
```

#### 3.2. Bilah Kemajuan
```python
def update_progress(ui_components, current, total, message=None):
    progress = ui_components.get('progress_bar')
    progress_label = ui_components.get('progress_label')
    
    if progress and hasattr(progress, 'value'):
        progress.value = current
        progress.max = total
    
    if progress_label and hasattr(progress_label, 'value'):
        persen = (current / total) * 100 if total > 0 else 0
        status = f"{message}: " if message else ""
        progress_label.value = f"{status}{current}/{total} ({persen:.1f}%)"

# Contoh penggunaan
update_progress(ui_components, 5, 10, "Memproses")
```

## 📋 Konsistensi Penamaan Berkas

### Pola Penamaan Berkas
- **Inisialisasi**: `[module]_initializer.py`
- **Pengelola Konfigurasi**: `config_handler.py` (standar)
- **Utama Handler**: `[module]_handlers.py`
- **Antarmuka Utama**: `ui_components.py` (standar)
- **Utilitas**: `[fungsi]_utils.py`

### Pola Penamaan Kelas
- **Inisialisasi**: `[Module]Initializer`
- **Pengelola Konfigurasi**: `[Module]ConfigHandler`
- **Komponen Antarmuka**: Ekspor fungsional

## 🔧 Template Standar

### 1. Template Inisialisasi

```python
"""
File: smartcash/ui/[domain]/[module]/[module]_initializer.py
Deskripsi: Inisialisasi [Module] yang mewarisi CommonInitializer
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.[domain].[module].handlers.config_handler import [Module]ConfigHandler
from smartcash.ui.[domain].[module].components.ui_components import create_[module]_main_ui
from smartcash.ui.[domain].[module].handlers.[module]_handlers import setup_[module]_handlers

class [Module]Initializer(CommonInitializer):
    """Inisialisasi [Module] dengan antarmuka dan integrasi backend yang lengkap"""
    
    def __init__(self):
        super().__init__(
            module_name='[module]',
            config_handler_class=[Module]ConfigHandler,
            parent_module='[domain]'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Membuat komponen antarmuka pengguna untuk modul"""
        ui_components = create_[module]_main_ui(config)
        ui_components.update({
            '[module]_initialized': True,
            'module_name': '[module]',
            'data_dir': config.get('data', {}).get('dir', 'data')
        })
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], 
                             env=None, **kwargs) -> Dict[str, Any]:
        """Menyiapkan handler dengan pemuatan konfigurasi dan pembaruan UI otomatis"""
        # Siapkan handler terlebih dahulu
        result = setup_[module]_handlers(ui_components, config, env)
        
        # PENTING: Muat konfigurasi dari berkas dan perbarui UI
        self._load_and_update_ui(ui_components)
        
        return result
    
    def _load_and_update_ui(self, ui_components: Dict[str, Any]):
        """PENTING: Memuat konfigurasi dari berkas dan memperbarui UI saat inisialisasi"""
        try:
            config_handler = ui_components.get('config_handler')
            if config_handler:
                # Atur komponen UI untuk pencatatan log
                if hasattr(config_handler, 'set_ui_components'):
                    config_handler.set_ui_components(ui_components)
                
                # Muat konfigurasi dari berkas dengan pewarisan
                loaded_config = config_handler.load_config()
                
                # Perbarui UI dengan konfigurasi yang dimuat
                config_handler.update_ui(ui_components, loaded_config)
                
                # Perbarui referensi konfigurasi
                ui_components['config'] = loaded_config
                
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.warning(f"⚠️ Gagal memuat konfigurasi: {str(e)}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Mendapatkan konfigurasi bawaan"""
        from smartcash.ui.[domain].[module].handlers.defaults import get_default_[module]_config
        return get_default_[module]_config()
    
    def _get_critical_components(self) -> List[str]:
        return [
            'ui', '[primary]_button', 'check_button', 'cleanup_button',
            'save_button', 'reset_button', 'log_output', 'status_panel'
        ]

# Instance global
_[module]_initializer = [Module]Initializer()

def initialize_[module]_ui(env=None, config=None, **kwargs):
    """Fungsi factory untuk UI [module] dengan pemuatan konfigurasi otomatis"""
    return _[module]_initializer.initialize(env=env, config=config, **kwargs)
```

### 2. Template Penangan Konfigurasi (POLA PENTING)

```python
"""
File: smartcash/ui/[domain]/[module]/handlers/config_handler.py
Deskripsi: Penangan konfigurasi dengan pencatatan log yang baik dan penyegaran UI otomatis
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.[domain].[module].handlers.config_extractor import extract_[module]_config
from smartcash.ui.[domain].[module].handlers.config_updater import update_[module]_ui
from smartcash.common.config.manager import get_config_manager

class [Module]ConfigHandler(ConfigHandler):
    """Penangan konfigurasi dengan pencatatan log UI dan pewarisan yang tepat"""
    
    def __init__(self, module_name: str = '[module]', parent_module: str = '[domain]'):
        super().__init__(module_name, parent_module)
        self.config_manager = get_config_manager()
        self.config_filename = '[module]_config.yaml'
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Ekstrak konfigurasi dengan pendekatan DRY"""
        return extract_[module]_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Memperbarui UI berdasarkan konfigurasi"""
        update_[module]_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Mendapatkan konfigurasi bawaan dari defaults.py"""
        from smartcash.ui.[domain].[module].handlers.defaults import get_default_[module]_config
        return get_default_[module]_config()
    
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """PENTING: Memuat konfigurasi dengan penanganan pewarisan"""
        try:
            filename = config_filename or self.config_filename
            config = self.config_manager.load_config(filename)
            
            if not config:
                self._log_to_ui("⚠️ Konfigurasi kosong, menggunakan setelan bawaan", "warning")
                return self.get_default_config()
            
            # PENTING: Menangani pewarisan dari _base_
            if '_base_' in config:
                base_config = self.config_manager.load_config(config['_base_']) or {}
                merged_config = self._merge_configs(base_config, config)
                self._log_to_ui(f"📂 Konfigurasi dimuat dari {filename} dengan pewarisan", "info")
                return merged_config
            
            self._log_to_ui(f"📂 Konfigurasi dimuat dari {filename}", "info")
            return config
            
        except Exception as e:
            self._log_to_ui(f"❌ Gagal memuat konfigurasi: {str(e)}", "error")
            return self.get_default_config()
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """PENTING: Menggabungkan konfigurasi dasar dengan penggantian"""
        import copy
        merged = copy.deepcopy(base_config)
        
        for key, value in override_config.items():
            if key == '_base_':
                continue
                
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Menggabungkan kamus secara rekursif"""
        import copy
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """PENTING: Menyimpan konfigurasi dengan penyegaran otomatis"""
        try:
            filename = config_filename or self.config_filename
            ui_config = self.extract_config(ui_components)
            
            success = self.config_manager.save_config(ui_config, filename)
            
            if success:
                self._log_to_ui(f"✅ Konfigurasi tersimpan ke {filename}", "success")
                self._refresh_ui_after_save(ui_components, filename)
                return True
            else:
                self._log_to_ui(f"❌ Gagal menyimpan konfigurasi ke {filename}", "error")
                return False
                
        except Exception as e:
            self._log_to_ui(f"❌ Kesalahan saat menyimpan konfigurasi: {str(e)}", "error")
            return False
    
    def reset_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """PENTING: Mengatur ulang konfigurasi ke nilai bawaan"""
        try:
            filename = config_filename or self.config_filename
            default_config = self.get_default_config()
            
            success = self.config_manager.save_config(default_config, filename)
            
            if success:
                self._log_to_ui(f"🔄 Konfigurasi diatur ulang ke setelan bawaan", "success")
                self.update_ui(ui_components, default_config)
                return True
            else:
                self._log_to_ui(f"❌ Gagal mengatur ulang konfigurasi", "error")
                return False
                
        except Exception as e:
            self._log_to_ui(f"❌ Kesalahan saat mengatur ulang konfigurasi: {str(e)}", "error")
            return False
    
    def _refresh_ui_after_save(self, ui_components: Dict[str, Any], filename: str):
        """PENTING: Menyegarkan UI secara otomatis setelah penyimpanan"""
        try:
            saved_config = self.load_config(filename)
            if saved_config:
                self.update_ui(ui_components, saved_config)
                self._log_to_ui("🔄 Antarmuka disegarkan dengan konfigurasi terbaru", "info")
        except Exception as e:
            self._log_to_ui(f"⚠️ Gagal menyegarkan antarmuka: {str(e)}", "warning")
    
    def _log_to_ui(self, message: str, level: str = "info"):
        """PENTING: Mencatat log ke komponen UI dengan fallback"""
        try:
            ui_components = getattr(self, '_ui_components', {})
            logger = ui_components.get('logger')
            
            if logger and hasattr(logger, level):
                log_method = getattr(logger, level)
                log_method(message)
                return
            
            # Fallback ke log_to_accordion
            from smartcash.ui.[domain].[module].utils.ui_utils import log_to_accordion
            log_to_accordion(ui_components, message, level)
                
        except Exception:
            print(f"[{level.upper()}] {message}")
    
    def set_ui_components(self, ui_components: Dict[str, Any]):
        """PENTING: Mengatur komponen UI untuk pencatatan log"""
        self._ui_components = ui_components
```

### 3. Template Ekstraktor Konfigurasi (PENDEKATAN DRY)

```python
"""
File: smartcash/ui/[domain]/[module]/handlers/config_extractor.py
Deskripsi: Ekstraksi konfigurasi DRY dengan nilai bawaan sebagai dasar
"""

from typing import Dict, Any

def extract_[module]_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """PENTING: Pendekatan DRY - dasar dari nilai bawaan + nilai form"""
    from smartcash.ui.[domain].[module].handlers.defaults import get_default_[module]_config
    
    # Struktur dasar dari nilai bawaan (DRY)
    config = get_default_[module]_config()
    
    # Helper untuk mendapatkan nilai form
    get_value = lambda key, default: getattr(ui_components.get(key, type('', (), {'value': default})()), 'value', default)
    
    # Hanya memperbarui nilai dari form - contoh:
    config['[module]']['option1'] = get_value('option1_input', 'default')
    config['[module]']['option2'] = get_value('option2_checkbox', False)
    config['performance']['num_workers'] = get_value('worker_slider', 8)
    
    return config
```

### 4. Template Pembaruan Konfigurasi (DENGAN PEWARISAN)

```python
"""
File: smartcash/ui/[domain]/[module]/handlers/config_updater.py
Deskripsi: Pembaruan konfigurasi dengan penanganan pewarisan
"""

from typing import Dict, Any

def update_[module]_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """PENTING: Perbarui UI dengan penanganan pewarisan"""
    # Ekstrak bagian dengan nilai bawaan yang aman (menangani pewarisan)
    [module]_config = config.get('[module]', {})
    performance_config = config.get('performance', {})
    
    safe_update = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Pemetaan field dengan validasi
    safe_update('option1_input', [module]_config.get('option1', 'default'))
    safe_update('option2_checkbox', [module]_config.get('option2', False))
    safe_update('worker_slider', min(max(performance_config.get('num_workers', 8), 1), 10))

def reset_[module]_ui(ui_components: Dict[str, Any]) -> None:
    """Atur ulang UI ke nilai bawaan"""
    try:
        from smartcash.ui.[domain].[module].handlers.defaults import get_default_[module]_config
        default_config = get_default_[module]_config()
        update_[module]_ui(ui_components, default_config)
    except Exception:
        _apply_hardcoded_defaults(ui_components)

def _apply_hardcoded_defaults(ui_components: Dict[str, Any]) -> None:
    """Fallback ke nilai bawaan hardcoded"""
    defaults = {'option1_input': 'default', 'worker_slider': 8}
    for key, value in defaults.items():
        if key in ui_components and hasattr(ui_components[key], 'value'):
            try:
                ui_components[key].value = value
            except Exception:
                pass
```

### 5. Template Handler (INTEGRASI PENANGAN KONFIGURASI)

```python
def setup_[module]_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Menyiapkan handler dengan integrasi penangan konfigurasi UI"""
    
    # PENTING: Siapkan penangan konfigurasi dengan logger UI
    config_handler = ui_components.get('config_handler')
    if config_handler and hasattr(config_handler, 'set_ui_components'):
        config_handler.set_ui_components(ui_components)
    
    # Siapkan handler lainnya...
    setup_config_handlers_fixed(ui_components, config)
    
    return ui_components

def setup_config_handlers_fixed(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """PENTING: Penangan konfigurasi dengan pencatatan log UI yang tepat"""
    
    def save_config(button=None):
        clear_outputs(ui_components)
        
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                handle_ui_error(ui_components, "❌ Penangan konfigurasi tidak tersedia")
                return
            
            # PENTING: Atur komponen UI untuk pencatatan log
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            success = config_handler.save_config(ui_components)
            # Pencatat log sudah ditangani di config_handler
            
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Gagal menyimpan: {str(e)}")
    
    def reset_config(button=None):
        clear_outputs(ui_components)
        
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                handle_ui_error(ui_components, "❌ Penangan konfigurasi tidak tersedia")
                return
            
            # PENTING: Atur komponen UI untuk pencatatan log
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            success = config_handler.reset_config(ui_components)
            # Pencatat log sudah ditangani di config_handler
            
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Gagal mengatur ulang: {str(e)}")
    
    # Hubungkan handler
    save_button = ui_components.get('save_button')
    reset_button = ui_components.get('reset_button')
    if save_button:
        save_button.on_click(save_config)
    if reset_button:
        reset_button.on_click(reset_config)
```

## 🐄 Daftar Periksa Implementasi

### Fase Persiapan
- [ ] Buat struktur direktori
- [ ] Implementasikan inisialisasi dengan pewarisan CommonInitializer
- [ ] **PENTING**: Tambahkan `_load_and_update_ui()` di inisialisator
- [ ] Siapkan penangan konfigurasi dengan pewarisan yang tepat

### Manajemen Konfigurasi (PERBAIKAN PENTING)
- [ ] **PENTING**: Implementasikan `_log_to_ui()` dengan fallback ke `log_to_accordion`
- [ ] **PENTING**: Tambahkan method `set_ui_components()` di penangan konfigurasi
- [ ] **PENTING**: Implementasikan `load_config()` dengan `_merge_configs()` untuk pewarisan
- [ ] **PENTING**: Tambahkan `_refresh_ui_after_save()` untuk pembaruan UI otomatis
- [ ] Implementasikan ekstraktor konfigurasi DRY dengan nilai bawaan sebagai dasar
- [ ] Implementasikan pembarui konfigurasi dengan ekstraksi aman dari pewarisan

### Integrasi Handler (PENTING)
- [ ] **PENTING**: Panggil `config_handler.set_ui_components()` di penyiapan handler
- [ ] **PENTING**: Gunakan pola `setup_config_handlers_fixed()`
- [ ] Siapkan penanganan error dengan pencatatan log UI

### Komponen UI
- [ ] Buat file komponen UI utama
- [ ] Implementasikan opsi input dengan tata letak responsif
- [ ] Siapkan pelacakan progress dengan dua level
- [ ] Tambahkan manajemen tombol yang tepat

## 💡 Pelajaran Penting

### 1. **Pemuatan Konfigurasi & Pembaruan UI**
- **HARUS** mengimplementasikan `_load_and_update_ui()` di inisialisator
- **HARUS** menangani pewarisan dengan `_merge_configs()`
- **HARUS** memanggil `config_handler.set_ui_components()` untuk pencatatan log

### 2. **Pencatatan Log ke UI**
- **HARUS** mengimplementasikan `_log_to_ui()` dengan fallback ke `log_to_accordion`
- **JANGAN PERNAH** hanya mengandalkan `print()` - log tidak akan muncul di UI

### 3. **Prinsip DRY**
- **SELALU** gunakan defaults.py sebagai struktur dasar
- **HANYA** perbarui nilai form di ekstraktor
- **HINDARI** menulis ulang seluruh struktur konfigurasi

### 4. **Pembaruan Otomatis**
- **HARUS** mengimplementasikan `_refresh_ui_after_save()`
- **HARUS** memuat ulang konfigurasi dari file setelah penyimpanan
- **HARUS** memanggil `update_ui()` dengan konfigurasi yang dimuat ulang

### 5. **Pencegahan Error**
- **SELALU** gunakan safe_update dengan try/catch
- **SELALU** validasi nilai dropdown sebelum penugasan
- **SELALU** tangani kunci yang hilang dengan `.get()` dan nilai default


