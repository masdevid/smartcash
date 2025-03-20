"""
File: smartcash/common/config.py
Deskripsi: Manager konfigurasi dengan dukungan YAML, environment variables, dan dependency injection
"""

import os, yaml, json, copy
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type, TypeVar, Callable, Tuple, List

# Type variable untuk dependency injection
T = TypeVar('T')

class ConfigManager:
    """Manager untuk konfigurasi aplikasi dengan dukungan untuk loading dari file YAML/JSON, environment variable overrides, hierarchical configs, dan dependency injection"""
    
    DEFAULT_CONFIG_DIR = 'configs'
    
    def __init__(self, base_dir: Optional[str] = None, config_file: Optional[str] = None, env_prefix: str = 'SMARTCASH_'):
        """Inisialisasi config manager dengan base directory, file konfigurasi utama, dan prefix environment variable"""
        self.base_dir = Path(base_dir) if base_dir else Path.cwd(); self.config_dir = self.base_dir / self.DEFAULT_CONFIG_DIR
        self.env_prefix = env_prefix; self.config = {}; self._dependencies = {}; self._factory_functions = {}
        if config_file: self.load_config(config_file)  # Muat konfigurasi dari file jika disediakan
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load konfigurasi dari file YAML/JSON, dengan resolve path relatif dan override environment variables"""
        config_path = self._resolve_config_path(config_file)
        if not config_path.exists(): raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")
        
        # Load berdasarkan ekstensi file
        if config_path.suffix.lower() in ('.yml', '.yaml'): 
            with open(config_path, 'r', encoding='utf-8') as f: self.config = yaml.safe_load(f) or {}
        elif config_path.suffix.lower() == '.json': 
            with open(config_path, 'r', encoding='utf-8') as f: self.config = json.load(f)
        else: raise ValueError(f"Format file konfigurasi tidak didukung: {config_path.suffix}")
        
        self._override_with_env_vars()  # Override dengan environment variables
        return self.config
    
    def _resolve_config_path(self, config_file: str) -> Path:
        """Resolve path konfigurasi relatif atau absolut ke Path lengkap"""
        config_path = Path(config_file)
        if config_path.is_absolute(): return config_path  # Jika path absolut, gunakan langsung
        if (self.config_dir / config_path).exists(): return self.config_dir / config_path  # Jika di config_dir
        if (Path.cwd() / config_path).exists(): return Path.cwd() / config_path  # Jika di direktori kerja
        return self.config_dir / config_path  # Default ke config_dir
    
    def _override_with_env_vars(self) -> None:
        """Override konfigurasi dengan environment variables menggunakan konvensi SMARTCASH_SECTION_KEY=value"""
        for env_name, env_value in os.environ.items():
            if not env_name.startswith(self.env_prefix): continue  # Hanya proses var dengan prefix yang sesuai
            config_path = env_name[len(self.env_prefix):].lower().split('_')  # Convert ENV_VAR ke nested dict path
            
            # Traverse & update config dict
            current = self.config
            for part in config_path[:-1]:
                if part not in current: current[part] = {}
                current = current[part]
            
            current[config_path[-1]] = self._parse_env_value(env_value)  # Set nilai dengan auto type conversion
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse nilai environment variable ke tipe yang sesuai (bool, number, list, string)"""
        if value.lower() in ('true', 'yes', '1'): return True  # Boolean True
        elif value.lower() in ('false', 'no', '0'): return False  # Boolean False
            
        # Numbers
        try: return float(value) if '.' in value else int(value)
        except ValueError: pass
            
        # Lists (comma-separated values)
        if ',' in value: return [self._parse_env_value(item.strip()) for item in value.split(',')]
            
        return value  # Default: string
    
    def get(self, key: str, default=None) -> Any:
        """Ambil nilai konfigurasi dengan dot notation (e.g., 'model.img_size.width')"""
        parts = key.split('.'); current = self.config
        for part in parts:
            if not isinstance(current, dict) or part not in current: return default
            current = current[part]
        return current
    
    def set(self, key: str, value: Any) -> None:
        """Set nilai konfigurasi dengan dot notation (e.g., 'model.img_size.width')"""
        parts = key.split('.'); current = self.config
        for part in parts[:-1]:  # Traverse sampai level terakhir
            if part not in current: current[part] = {}
            current = current[part]
        current[parts[-1]] = value  # Set nilai
    
    def merge_config(self, config: Union[Dict, str]) -> Dict[str, Any]:
        """Merge konfigurasi dari dict atau file dengan current config"""
        # Load dari file jika string
        if isinstance(config, str):
            config_path = self._resolve_config_path(config)
            if config_path.suffix.lower() in ('.yml', '.yaml'):
                with open(config_path, 'r', encoding='utf-8') as f: config = yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f: config = json.load(f)
            else: raise ValueError(f"Format file tidak didukung: {config_path.suffix}")
        
        self._deep_merge(self.config, config)  # Deep merge config
        return self.config
    
    def _deep_merge(self, target: Dict, source: Dict) -> None:
        """Deep merge dua dictionary secara rekursif"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)  # Rekursif untuk nested dict
            else: target[key] = value  # Override atau tambahkan key baru
    
    def save_config(self, config_file: str, create_dirs: bool = True) -> None:
        """Simpan konfigurasi ke file YAML/JSON"""
        config_path = Path(config_file)
        if create_dirs and not config_path.parent.exists(): config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Simpan berdasarkan ekstensi
        if config_path.suffix.lower() in ('.yml', '.yaml'):
            with open(config_path, 'w', encoding='utf-8') as f: yaml.dump(self.config, f, default_flow_style=False)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f: json.dump(self.config, f, indent=2)
        else:  # Default ke YAML
            with open(f"{config_path}.yaml", 'w', encoding='utf-8') as f: yaml.dump(self.config, f, default_flow_style=False)
    
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
        if interface_type in self._factory_functions: return self._factory_functions[interface_type](*args, **kwargs)
        if interface_type in self._dependencies:
            implementation = self._dependencies[interface_type]
            return implementation if not isinstance(implementation, type) else implementation(*args, **kwargs)
        raise KeyError(f"Tidak ada implementasi terdaftar untuk {interface_type.__name__}")

    def sync_with_drive_enhanced(self, config_file: str, sync_strategy: str = 'drive_priority', backup: bool = True) -> Tuple[bool, str, Dict[str, Any]]:
        """Sinkronisasi file konfigurasi dengan Google Drive menggunakan strategi yang ditingkatkan."""
        try:
            from smartcash.common.config_sync import sync_config_with_drive
            logger = None; 
            try: 
                from smartcash.common.logger import get_logger; logger = get_logger("config_manager")
            except ImportError: pass
            
            # Panggil fungsi sinkronisasi yang ditingkatkan
            success, message, merged_config = sync_config_with_drive(
                config_file=config_file, sync_strategy=sync_strategy, create_backup=backup, logger=logger
            )
            
            # Update config saat ini jika sukses
            if success and merged_config: self.config = merged_config
            return success, message, merged_config
            
        except Exception as e:
            error_msg = f"❌ Error saat sinkronisasi konfigurasi: {str(e)}"
            if hasattr(self, '_logger') and self._logger: self._logger.error(error_msg)
            return False, error_msg, {}

    def use_drive_as_source_of_truth(self) -> bool:
        """Sinkronisasi semua konfigurasi dengan Drive sebagai sumber kebenaran."""
        try:
            from smartcash.common.config_sync import sync_all_configs
            logger = None
            try: 
                from smartcash.common.logger import get_logger; logger = get_logger("config_manager")
            except ImportError: pass
            
            # Sinkronisasi semua konfigurasi dengan Drive sebagai prioritas
            results = sync_all_configs(sync_strategy='drive_priority', create_backup=True, logger=logger)
            
            # Muat ulang konfigurasi utama setelah sinkronisasi
            if "success" in results and results["success"]:
                for result in results["success"]:
                    if result["file"] == "base_config.yaml":
                        self.load_config("base_config.yaml"); break
            
            # Hitung jumlah sukses dan gagal
            success_count = len(results.get("success", [])); failure_count = len(results.get("failure", []))
            
            # Log hasil operasi
            if logger:
                logger.info(f"🔄 Sinkronisasi selesai: {success_count} berhasil, {failure_count} gagal")
                if failure_count > 0:
                    for failure in results.get("failure", []): 
                        logger.warning(f"⚠️ Gagal sinkronisasi {failure.get('file', 'unknown')}: {failure.get('message', 'unknown error')}")
            
            return failure_count == 0  # Operasi dianggap sukses jika tidak ada kegagalan
            
        except Exception as e:
            error_msg = f"❌ Error saat menggunakan Drive sebagai sumber kebenaran: {str(e)}"
            if hasattr(self, '_logger') and self._logger: self._logger.error(error_msg)
            return False

    def get_drive_config_path(self, config_file: str = None) -> Optional[str]:
        """Dapatkan path konfigurasi di Google Drive."""
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            if not env_manager.is_drive_mounted: return None
            drive_configs_dir = env_manager.drive_path / 'configs'
            
            return str(drive_configs_dir / config_file) if config_file else str(drive_configs_dir)
            
        except Exception: return None
    
    def _merge_configs_smart(self, config1, config2):
        """Menggabungkan dua konfigurasi dengan strategi smart."""
        if config1 is None: return copy.deepcopy(config2)
        if config2 is None: return copy.deepcopy(config1)
        
        # Jika keduanya adalah dict, proses secara rekursif
        if isinstance(config1, dict) and isinstance(config2, dict):
            result = copy.deepcopy(config1)
            for key, value in config2.items():
                if key in result: result[key] = self._merge_configs_smart(result[key], value)
                else: result[key] = copy.deepcopy(value)
            return result
        
        # Jika keduanya adalah list, gabungkan dan hilangkan duplikat
        if isinstance(config1, list) and isinstance(config2, list):
            # Khusus untuk list sederhana, gabungkan dengan unik
            if all(not isinstance(x, (dict, list)) for x in config1 + config2): return list(set(config1 + config2))
            # Untuk list kompleks, gabungkan saja
            return copy.deepcopy(config1) + copy.deepcopy(config2)
        
        # Untuk nilai skalar, prioritaskan nilai yang tidak None/empty
        if config1 == "" or config1 is None or config1 == 0: return copy.deepcopy(config2)
        return copy.deepcopy(config1)
    
    def __getitem__(self, key): return self.get(key)
    def __setitem__(self, key, value): self.set(key, value)

# Singleton instance
_config_manager = None

def get_config_manager(base_dir=None, config_file=None, env_prefix='SMARTCASH_'):
    """Dapatkan instance ConfigManager (singleton)."""
    global _config_manager
    if _config_manager is None: _config_manager = ConfigManager(base_dir, config_file, env_prefix)
    return _config_manager