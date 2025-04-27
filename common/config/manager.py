"""
File: smartcash/common/config/manager.py
Deskripsi: Manager konfigurasi dengan dukungan YAML, environment variables, dan dependency injection
"""

import os
import copy
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type, TypeVar, Callable, Tuple

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
        self.config_dir = self.base_dir / self.DEFAULT_CONFIG_DIR
        self.env_prefix = env_prefix
        self.config = {}
        self._dependencies = {}
        self._factory_functions = {}
        
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
        
        if self.logger:
            self.logger.info(f"✅ Konfigurasi dimuat dari: {config_path}")
            
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
                    if self.logger:
                        self.logger.info(f"📁 Menggunakan file konfigurasi dari: {project_config_path}")
                    return project_config_path
            
        # Check relative to config_dir
        if (self.config_dir / config_path).exists(): 
            return self.config_dir / config_path
        
        # Check relative to project root (smartcash/configs)
        project_config_path = Path('smartcash') / 'configs' / config_path.name
        if project_config_path.exists():
            if self.logger:
                self.logger.info(f"📁 Menggunakan file konfigurasi dari: {project_config_path}")
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
                if self.logger:
                    self.logger.info(f"📁 Menggunakan file konfigurasi dari: {path}")
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