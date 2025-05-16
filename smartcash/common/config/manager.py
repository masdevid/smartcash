"""
File: smartcash/common/config/manager.py
Deskripsi: Manager konfigurasi dengan dukungan YAML, environment variables, dan dependency injection
"""

import os
import copy
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type, TypeVar, Callable, Tuple, List

# Import dari IO module
from smartcash.common.io import (
    load_json,
    save_json,
    load_yaml,
    save_yaml,
    load_config,
    save_config,
    ensure_dir
)

# Type variable untuk dependency injection
T = TypeVar('T')

class ConfigManager:
    """Manager untuk konfigurasi aplikasi dengan dukungan untuk loading dari file, environment variable overrides, dan dependency injection"""
    
    DEFAULT_CONFIG_DIR = 'configs'
    
    def __init__(self, base_dir: Optional[str] = None, config_file: Optional[str] = None, env_prefix: str = 'SMARTCASH_'):
        """Inisialisasi config manager dengan base directory, file konfigurasi utama, dan prefix environment variable"""
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        # Gunakan direktori configs/ di direktori saat ini untuk konsistensi dengan cell_utils.py
        self.config_dir = Path('configs')
        self.env_prefix = env_prefix
        self.config = {}
        self._dependencies = {}
        self._factory_functions = {}
        
        # Tambahan untuk persistensi dan observer pattern
        self.module_configs = {}  # Dictionary untuk menyimpan konfigurasi berbagai modul
        self.observers = {}       # Dictionary untuk menyimpan observer
        self.ui_components = {}   # Dictionary untuk menyimpan referensi UI components
        
        # Setup logger jika tersedia
        try:
            from smartcash.common.logger import get_logger
            self.logger = get_logger("config_manager")
        except ImportError:
            self.logger = None
            
        if config_file: 
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load konfigurasi dari file YAML/JSON, dengan resolve path relatif dan override environment variables"""
        config_path = self._resolve_config_path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")
        
        # Load konfigurasi menggunakan fungsi dari io module
        self.config = load_config(config_path, {})
        
        # Override dengan environment variables
        self._override_with_env_vars()
        
        return self.config
    
    def _resolve_config_path(self, config_file: str) -> Path:
        """Resolve path konfigurasi relatif atau absolut ke Path lengkap"""
        config_path = Path(config_file)
        
        # Check absolute path
        if config_path.is_absolute(): 
            if config_path.exists():
                return config_path
            
            # Jika di Colab, coba cari di direktori project
            is_colab = 'google.colab' in str(globals())
            if is_colab and '/content/' in str(config_path):
                # Coba cari di direktori smartcash/configs
                project_config_path = Path('smartcash') / 'configs' / config_path.name
                if project_config_path.exists():
                    return project_config_path
            
        # Check relative to config_dir
        if (self.config_dir / config_path).exists(): 
            return self.config_dir / config_path
        
        # Check relative to project root (smartcash/configs)
        project_config_path = Path('smartcash') / 'configs' / config_path.name
        if project_config_path.exists():
            return project_config_path
            
        # Check relative to current working directory
        if (Path.cwd() / config_path).exists(): 
            return Path.cwd() / config_path
        
        # Coba cari di berbagai lokasi umum
        common_paths = [
            Path('configs') / config_path.name,
            Path('smartcash/configs') / config_path.name,
            Path('/content/smartcash/configs') / config_path.name,
            Path('/content/configs') / config_path.name,
        ]
        
        for path in common_paths:
            if path.exists():
                return path
            
        # Default to config_dir
        return self.config_dir / config_path
    
    def _override_with_env_vars(self) -> None:
        """Override konfigurasi dengan environment variables menggunakan konvensi SMARTCASH_SECTION_KEY=value"""
        for env_name, env_value in os.environ.items():
            if not env_name.startswith(self.env_prefix): 
                continue
            
            # Konversi nama environment variable ke path config
            config_path = env_name[len(self.env_prefix):].lower().split('_')
            
            # Traverse & update config dict
            current = self.config
            for part in config_path[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Update nilai
            current[config_path[-1]] = self._parse_env_value(env_value)
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse nilai environment variable ke tipe yang sesuai (bool, number, list, string)"""
        # Boolean values
        if value.lower() in ('true', 'yes', '1'): 
            return True
        if value.lower() in ('false', 'no', '0'): 
            return False
            
        # Numbers
        try: 
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError: 
            pass
            
        # Lists (comma-separated values)
        if ',' in value: 
            return [self._parse_env_value(item.strip()) for item in value.split(',')]
            
        # Default: string
        return value
    
    def get(self, key: str, default=None) -> Any:
        """Ambil nilai konfigurasi dengan dot notation (e.g., 'model.img_size.width')"""
        parts = key.split('.')
        current = self.config
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
            
        return current
    
    def set(self, key: str, value: Any) -> None:
        """Set nilai konfigurasi dengan dot notation (e.g., 'model.img_size.width')"""
        parts = key.split('.')
        current = self.config
        
        # Traverse dan buat path jika perlu
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Set nilai
        current[parts[-1]] = value
    
    def merge_config(self, config: Union[Dict, str]) -> Dict[str, Any]:
        """Merge konfigurasi dari dict atau file dengan current config"""
        # Load dari file jika string
        if isinstance(config, str):
            config_path = self._resolve_config_path(config)
            loaded_config = load_config(config_path, {})
        else:
            loaded_config = config
        
        # Merge configs
        self._deep_merge(self.config, loaded_config)
        
        if self.logger:
            self.logger.info(f"✅ Konfigurasi berhasil digabungkan")
            
        return self.config
    
    def _deep_merge(self, target: Dict, source: Dict) -> None:
        """Deep merge dua dictionary secara rekursif"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def save_config(self, config_file: str, create_dirs: bool = True) -> bool:
        """Simpan konfigurasi ke file YAML/JSON"""
        try:
            config_path = Path(config_file)
            
            # Buat direktori jika perlu
            if create_dirs and not config_path.parent.exists():
                ensure_dir(config_path.parent)
            
            # Simpan berdasarkan ekstensi file
            if config_path.suffix.lower() in ('.yml', '.yaml'):
                save_yaml(self.config, config_path)
            elif config_path.suffix.lower() == '.json':
                save_json(self.config, config_path, pretty=True)
            else:
                # Default ke YAML
                yaml_path = f"{config_path}.yaml"
                save_yaml(self.config, yaml_path)
                
            if self.logger:
                self.logger.info(f"✅ Konfigurasi disimpan ke: {config_path}")
                
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
            return False
            
    # ===== Dependency Injection Methods =====
    
    def register(self, interface_type: Type[T], implementation: Type[T]) -> None:
        """Daftarkan implementasi untuk interface tertentu."""
        self._dependencies[interface_type] = implementation
    
    def register_instance(self, interface_type: Type[T], instance: T) -> None:
        """Daftarkan instance untuk interface tertentu (singleton)."""
        self._dependencies[interface_type] = instance
    
    def register_factory(self, interface_type: Type[T], factory: Callable[..., T]) -> None:
        """Daftarkan factory function untuk membuat implementasi."""
        self._factory_functions[interface_type] = factory
    
    def resolve(self, interface_type: Type[T], *args, **kwargs) -> T:
        """Resolve dependency untuk interface tertentu, dengan factory atau implementation class"""
        if interface_type in self._factory_functions:
            return self._factory_functions[interface_type](*args, **kwargs)
        if interface_type in self._dependencies:
            implementation = self._dependencies[interface_type]
            return implementation if not isinstance(implementation, type) else implementation(*args, **kwargs)
        raise KeyError(f"Tidak ada implementasi terdaftar untuk {interface_type.__name__}")

    def sync_with_drive(self, config_file: str, sync_strategy: str = 'drive_priority') -> Tuple[bool, str, Dict[str, Any]]:
        """
        Sinkronisasi file konfigurasi dengan Google Drive.
        
        Args:
            config_file: Path ke file konfigurasi
            sync_strategy: Strategi sinkronisasi ('merge', 'drive_priority', 'local_priority')
            
        Returns:
            Tuple (success, message, merged_config)
        """
        try:
            # Import modul config_sync
            from smartcash.common.config.sync import sync_config_with_drive
                
            # Panggil fungsi sinkronisasi dengan create_backup=True
            success, message, merged_config = sync_config_with_drive(
                config_file=config_file, 
                sync_strategy=sync_strategy, 
                create_backup=True,
                logger=self.logger
            )
            
            # Update config saat ini jika sukses
            if success and merged_config:
                self.config = merged_config
            return success, message, merged_config
            
        except ImportError:
            error_msg = f"❌ Module config_sync tidak tersedia"
            if self.logger:
                self.logger.error(error_msg)
            return False, error_msg, {}
        except Exception as e:
            error_msg = f"❌ Error saat sinkronisasi konfigurasi: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            return False, error_msg, {}
            
    def sync_to_drive(self, module_name: str) -> Tuple[bool, str]:
        """
        Sinkronisasi konfigurasi modul dengan Google Drive.
        
        Args:
            module_name: Nama modul yang akan disinkronkan
            
        Returns:
            Tuple (success, message)
        """
        if self.logger:
            self.logger.info(f"🔄 Memulai sinkronisasi konfigurasi {module_name} dengan Google Drive")
            
        try:
            # Pastikan konfigurasi modul ada
            if module_name not in self.module_configs:
                config = self.get_module_config(module_name)
                if not config:
                    if self.logger:
                        self.logger.warning(f"⚠️ Konfigurasi {module_name} tidak ditemukan untuk disinkronkan")
                    return False, f"Konfigurasi {module_name} tidak ditemukan"
            else:
                config = self.module_configs[module_name]
            
            # Debug log untuk melihat konfigurasi yang akan diupload
            if self.logger:
                self.logger.debug(f"🔍 Konfigurasi yang akan diupload ke Drive: {config}")
            
            # Tentukan nama file konfigurasi dengan pola xxx_config.yaml
            config_file = f"{module_name}_config.yaml"
            
            # Import modul config_sync
            from smartcash.common.config.sync import upload_config_to_drive
            
            # Simpan konfigurasi ke file lokal terlebih dahulu
            config_path = self._get_module_config_path(module_name)
            try:
                # Pastikan direktori ada
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                
                # Simpan konfigurasi ke file
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(config, f, default_flow_style=False)
                    
                if self.logger:
                    self.logger.debug(f"✅ Konfigurasi {module_name} berhasil disimpan ke file lokal")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"⚠️ Gagal menyimpan konfigurasi ke file lokal: {str(e)}")
                # Lanjutkan meskipun gagal menyimpan ke file lokal
            
            # Upload ke Google Drive
            # Pastikan config_path adalah string atau PathLike object, bukan dict
            success, message = upload_config_to_drive(str(config_path), config, self.logger)
            
            if success:
                if self.logger:
                    self.logger.info(f"✅ Konfigurasi {module_name} berhasil disinkronkan dengan Google Drive")
            else:
                if self.logger:
                    self.logger.warning(f"⚠️ Gagal menyinkronkan konfigurasi {module_name} dengan Google Drive: {message}")
            
            return success, message
        except ImportError:
            error_msg = f"❌ Module config_sync tidak tersedia untuk sinkronisasi dengan drive"
            if self.logger:
                self.logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"❌ Error saat menyinkronkan konfigurasi {module_name} dengan drive: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            import traceback
            if self.logger:
                self.logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            return False, error_msg

    def use_drive_as_source_of_truth(self) -> bool:
        """Sinkronisasi semua konfigurasi dengan Drive sebagai sumber kebenaran."""
        try:
            from smartcash.common.config.sync import sync_all_configs
            
            # Sinkronisasi semua konfigurasi dengan Drive sebagai prioritas
            results = sync_all_configs(
                sync_strategy='drive_priority',
                create_backup=True,
                logger=self.logger
            )
            
            # Muat ulang konfigurasi utama setelah sinkronisasi
            if "success" in results and results["success"]:
                for result in results["success"]:
                    if result["file"] == "base_config.yaml":
                        self.load_config("base_config.yaml")
                        break
            
            # Hitung jumlah sukses dan gagal
            success_count = len(results.get("success", []))
            failure_count = len(results.get("failure", []))
            
            # Log hasil operasi
            if self.logger:
                self.logger.info(f"🔄 Sinkronisasi selesai: {success_count} berhasil, {failure_count} gagal")
                if failure_count > 0:
                    for failure in results.get("failure", []): 
                        self.logger.warning(f"⚠️ Gagal sinkronisasi {failure.get('file', 'unknown')}: {failure.get('message', 'unknown error')}")
            
            return failure_count == 0
            
        except ImportError:
            if self.logger:
                self.logger.error("❌ Module config_sync tidak tersedia")
            return False
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error saat menggunakan Drive sebagai sumber kebenaran: {str(e)}")
            return False

    def get_drive_config_path(self, config_file: str = None) -> Optional[str]:
        """Dapatkan path konfigurasi di Google Drive."""
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            if not env_manager.is_drive_mounted:
                return None
                
            drive_configs_dir = env_manager.drive_path / 'configs'
            return str(drive_configs_dir / config_file) if config_file else str(drive_configs_dir)
            
        except Exception:
            return None
            
    def __getitem__(self, key):
        """Operator [] untuk mengakses konfigurasi."""
        return self.get(key)
        
    def __setitem__(self, key, value):
        """Operator [] untuk mengatur konfigurasi."""
        self.set(key, value)
        
    # ========== Metode untuk validasi parameter ==========
    
    def validate_param(self, value: Any, default_value: Any, 
                      valid_types: Optional[Union[type, List[type]]] = None, 
                      valid_values: Optional[List[Any]] = None) -> Any:
        """
        Validasi parameter konfigurasi.
        
        Args:
            value: Nilai yang akan divalidasi
            default_value: Nilai default jika validasi gagal
            valid_types: Tipe yang valid (single atau list)
            valid_values: List nilai yang valid
            
        Returns:
            Nilai yang valid atau default
        """
        # Validasi None
        if value is None:
            return default_value
        
        # Validasi tipe
        if valid_types:
            if not isinstance(valid_types, (list, tuple)):
                valid_types = [valid_types]
                
            if not any(isinstance(value, t) for t in valid_types):
                if self.logger:
                    self.logger.debug(f"⚠️ Validasi tipe gagal: {value} bukan {valid_types}, menggunakan default: {default_value}")
                return default_value
        
        # Validasi nilai
        if valid_values and value not in valid_values:
            if self.logger:
                self.logger.debug(f"⚠️ Validasi nilai gagal: {value} tidak dalam {valid_values}, menggunakan default: {default_value}")
            return default_value
        
        return value
    
    # ========== Metode untuk persistensi UI components ==========
    
    def _get_module_config_path(self, module_name: str) -> str:
        """
        Dapatkan path file konfigurasi untuk modul tertentu.
        
        Args:
            module_name: Nama modul
            
        Returns:
            Path file konfigurasi
        """
        # Gunakan direktori configs/ di direktori saat ini untuk konsistensi dengan cell_utils.py
        config_dir = 'configs'
        return os.path.join(config_dir, f"{module_name}_config.yaml")
    
    def get_module_config(self, module_name: str, default_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Dapatkan konfigurasi untuk modul tertentu.
        
        Args:
            module_name: Nama modul
            default_config: Konfigurasi default jika tidak ada yang tersimpan
            
        Returns:
            Dictionary konfigurasi
        """
        try:
            # Cek apakah sudah ada di cache
            if module_name in self.module_configs:
                return copy.deepcopy(self.module_configs[module_name])
            
            # Coba load dari file
            config_path = os.path.join(self.config_dir, f"{module_name}_config.yaml")
            
            if os.path.exists(config_path):
                try:
                    # Load konfigurasi dari file dengan penanganan error yang lebih baik
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f) or {}
                    
                    # Simpan ke cache
                    self.module_configs[module_name] = copy.deepcopy(config)
                    
                    if self.logger:
                        self.logger.info(f"✅ Konfigurasi {module_name} berhasil dimuat dari {config_path}")
                    
                    return copy.deepcopy(config)
                except Exception as yaml_error:
                    if self.logger:
                        self.logger.error(f"❌ Error saat memuat YAML {module_name}: {str(yaml_error)}")
                    # Jika gagal, gunakan default
                    if default_config is not None:
                        return copy.deepcopy(default_config)
                    else:
                        return {}
            else:
                # Jika file tidak ada, gunakan default
                if default_config is not None:
                    # Simpan default ke cache
                    self.module_configs[module_name] = copy.deepcopy(default_config)
                    
                    if self.logger:
                        self.logger.info(f"ℹ️ Menggunakan konfigurasi default untuk {module_name}")
                    
                    return copy.deepcopy(default_config)
                else:
                    # Return empty dict jika tidak ada default
                    if self.logger:
                        self.logger.warning(f"⚠️ Tidak ada konfigurasi untuk {module_name} dan tidak ada default")
                    
                    return {}
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error saat memuat konfigurasi {module_name}: {str(e)}")
            
            # Return default jika ada error
            if default_config is not None:
                return copy.deepcopy(default_config)
            else:
                return {}
    
    def reset_module_config(self, module_name: str, default_config: Dict[str, Any]) -> bool:
        """
        Reset konfigurasi modul ke nilai default.
        
        Args:
            module_name: Nama modul
            default_config: Konfigurasi default
            
        Returns:
            Boolean status keberhasilan
        """
        try:
            # Simpan konfigurasi default
            success = self.save_module_config(module_name, default_config)
            
            if self.logger:
                self.logger.info(f"✅ Konfigurasi {module_name} berhasil direset ke default")
                
            return success
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error saat reset konfigurasi {module_name}: {str(e)}")
            return False
    
    def save_module_config(self, module_name: str, config: Dict[str, Any]) -> bool:
        """
        Simpan konfigurasi untuk modul tertentu.
        
        Args:
            module_name: Nama modul
            config: Dictionary konfigurasi
            
        Returns:
            Boolean status keberhasilan
        """
        try:
            # Buat deep copy untuk mencegah modifikasi tidak sengaja
            config_copy = copy.deepcopy(config)
            
            # Simpan di cache
            self.module_configs[module_name] = config_copy
            
            # Simpan ke file
            config_path = os.path.join(self.config_dir, f"{module_name}_config.yaml")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Perbaikan: gunakan path langsung untuk save_yaml, bukan file object
            try:
                save_yaml(config_copy, config_path)
                if self.logger:
                    self.logger.info(f"✅ Konfigurasi {module_name} berhasil disimpan ke {config_path}")
            except Exception as save_error:
                if self.logger:
                    self.logger.error(f"❌ Error saat menyimpan YAML: {str(save_error)}")
                # Fallback: coba simpan dengan metode alternatif
                try:
                    with open(config_path, 'w', encoding='utf-8') as f:
                        yaml.dump(config_copy, f, default_flow_style=False, allow_unicode=True)
                    if self.logger:
                        self.logger.info(f"✅ Konfigurasi {module_name} berhasil disimpan dengan metode alternatif")
                except Exception as alt_error:
                    if self.logger:
                        self.logger.error(f"❌ Error saat menyimpan dengan metode alternatif: {str(alt_error)}")
                    raise
            
            # Notifikasi observer
            self.notify_observers(module_name, config_copy)
            
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error saat menyimpan konfigurasi {module_name}: {str(e)}")
            return False
    
    def register_ui_components(self, module_name: str, ui_components: Dict[str, Any]) -> None:
        """
        Register UI components untuk persistensi.
        
        Args:
            module_name: Nama modul
            ui_components: Dictionary komponen UI
        """
        self.ui_components[module_name] = ui_components
        
        if self.logger:
            self.logger.debug(f"🔗 UI components berhasil diregister untuk {module_name}")
    
    def get_ui_components(self, module_name: str) -> Dict[str, Any]:
        """
        Dapatkan UI components yang tersimpan.
        
        Args:
            module_name: Nama modul
            
        Returns:
            Dictionary komponen UI atau empty dict jika tidak ditemukan
        """
        return self.ui_components.get(module_name, {})
    
    # ========== Metode untuk observer pattern ==========
    
    def register_observer(self, module_name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register observer untuk notifikasi perubahan konfigurasi.
        
        Args:
            module_name: Nama modul
            callback: Fungsi callback yang akan dipanggil saat konfigurasi berubah
        """
        if module_name not in self.observers:
            self.observers[module_name] = []
        
        self.observers[module_name].append(callback)
        
        if self.logger:
            self.logger.debug(f"👁️ Observer berhasil diregister untuk {module_name}")
    
    def unregister_observer(self, module_name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Unregister observer.
        
        Args:
            module_name: Nama modul
            callback: Fungsi callback yang akan dihapus
        """
        if module_name in self.observers and callback in self.observers[module_name]:
            self.observers[module_name].remove(callback)
            
            if self.logger:
                self.logger.debug(f"👁️ Observer berhasil diunregister untuk {module_name}")
    
    def notify_observers(self, module_name: str, config: Dict[str, Any]) -> None:
        """
        Notifikasi semua observer tentang perubahan konfigurasi.
        
        Args:
            module_name: Nama modul
            config: Konfigurasi terbaru
        """
        if module_name in self.observers:
            for callback in self.observers[module_name]:
                try:
                    callback(copy.deepcopy(config))
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"⚠️ Error saat memanggil observer {module_name}: {str(e)}")

# Singleton instance
_config_manager = None

def get_config_manager(base_dir=None, config_file=None, env_prefix='SMARTCASH_'):
    """
    Dapatkan instance ConfigManager (singleton).
    
    Args:
        base_dir: Direktori dasar
        config_file: File konfigurasi utama
        env_prefix: Prefix untuk environment variables
        
    Returns:
        Instance singleton ConfigManager
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(base_dir, config_file, env_prefix)
    return _config_manager

# Tambahkan method get_instance sebagai staticmethod untuk kompatibilitas
ConfigManager.get_instance = staticmethod(get_config_manager)