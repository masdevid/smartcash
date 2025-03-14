# smartcash/common/config.py
"""
File: smartcash/common/config.py
Deskripsi: Manager konfigurasi dengan dukungan YAML dan environment variables
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

class ConfigManager:
    """
    Manager untuk konfigurasi aplikasi dengan dukungan untuk:
    - Loading dari file YAML/JSON
    - Environment variable overrides
    - Hierarki konfigurasi
    """
    
    DEFAULT_CONFIG_DIR = 'config'
    
    def __init__(self, 
                base_dir: Optional[str] = None, 
                config_file: Optional[str] = None,
                env_prefix: str = 'SMARTCASH_'):
        """
        Inisialisasi config manager.
        
        Args:
            base_dir: Direktori root project
            config_file: Path file konfigurasi utama (relatif ke base_dir)
            env_prefix: Prefix untuk environment variables
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.config_dir = self.base_dir / self.DEFAULT_CONFIG_DIR
        self.env_prefix = env_prefix
        
        # Konfigurasi dasar
        self.config = {}
        
        # Muat konfigurasi dari file jika disediakan
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load konfigurasi dari file.
        
        Args:
            config_file: Nama file konfigurasi atau path lengkap
            
        Returns:
            Dictionary konfigurasi
        """
        # Tentukan path konfigurasi
        config_path = self._resolve_config_path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")
        
        # Load berdasarkan ekstensi file
        if config_path.suffix.lower() in ('.yml', '.yaml'):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Format file konfigurasi tidak didukung: {config_path.suffix}")
        
        # Override dengan environment variables
        self._override_with_env_vars()
        
        return self.config
    
    def _resolve_config_path(self, config_file: str) -> Path:
        """Resolve path konfigurasi relatif atau absolut."""
        config_path = Path(config_file)
        
        # Jika path absolut, gunakan langsung
        if config_path.is_absolute():
            return config_path
            
        # Jika config ada di config_dir, gunakan
        if (self.config_dir / config_path).exists():
            return self.config_dir / config_path
            
        # Jika config ada di direktori kerja, gunakan
        if (Path.cwd() / config_path).exists():
            return Path.cwd() / config_path
            
        # Default ke config_dir
        return self.config_dir / config_path
    
    def _override_with_env_vars(self) -> None:
        """Override konfigurasi dengan environment variables."""
        for env_name, env_value in os.environ.items():
            # Hanya proses var dengan prefix yang sesuai
            if not env_name.startswith(self.env_prefix):
                continue
                
            # Convert format ENV_VAR ke nested dict
            # contoh: SMARTCASH_MODEL_IMG_SIZE_WIDTH=640 -> config['model']['img_size']['width']=640
            config_path = env_name[len(self.env_prefix):].lower().split('_')
            
            # Traverse & update config dict
            current = self.config
            for part in config_path[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
                
            # Set nilai (dengan auto type conversion)
            key = config_path[-1]
            current[key] = self._parse_env_value(env_value)
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse nilai environment variable ke tipe yang sesuai."""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
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
        """
        Ambil nilai konfigurasi dengan dot notation.
        
        Args:
            key: Key dengan dot notation (e.g., 'model.img_size.width')
            default: Nilai default jika key tidak ditemukan
            
        Returns:
            Nilai konfigurasi atau default
        """
        parts = key.split('.')
        current = self.config
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
            
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        Set nilai konfigurasi dengan dot notation.
        
        Args:
            key: Key dengan dot notation (e.g., 'model.img_size.width')
            value: Nilai yang akan di-set
        """
        parts = key.split('.')
        current = self.config
        
        # Traverse sampai level terakhir
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Set nilai
        current[parts[-1]] = value
    
    def merge_config(self, config: Union[Dict, str]) -> Dict[str, Any]:
        """
        Merge konfigurasi dari dict atau file.
        
        Args:
            config: Dictionary konfigurasi atau path file
            
        Returns:
            Konfigurasi setelah merge
        """
        # Load dari file jika string
        if isinstance(config, str):
            config_path = self._resolve_config_path(config)
            
            if config_path.suffix.lower() in ('.yml', '.yaml'):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Format file tidak didukung: {config_path.suffix}")
        
        # Deep merge config
        self._deep_merge(self.config, config)
        return self.config
    
    def _deep_merge(self, target: Dict, source: Dict) -> None:
        """
        Deep merge dua dictionary.
        
        Args:
            target: Dictionary target
            source: Dictionary source
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Rekursif untuk nested dict
                self._deep_merge(target[key], value)
            else:
                # Override atau tambahkan key baru
                target[key] = value
    
    def save_config(self, config_file: str, create_dirs: bool = True) -> None:
        """
        Simpan konfigurasi ke file.
        
        Args:
            config_file: Path file untuk menyimpan konfigurasi
            create_dirs: Flag untuk create direktori jika belum ada
        """
        config_path = Path(config_file)
        
        # Buat direktori jika diperlukan
        if create_dirs and not config_path.parent.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Simpan berdasarkan ekstensi
        if config_path.suffix.lower() in ('.yml', '.yaml'):
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
        else:
            # Default ke YAML
            with open(f"{config_path}.yaml", 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
    
    def __getitem__(self, key):
        return self.get(key)
    
    def __setitem__(self, key, value):
        self.set(key, value)

# Singleton instance
_config_manager = None

def get_config_manager(base_dir=None, config_file=None, env_prefix='SMARTCASH_'):
    """
    Dapatkan instance ConfigManager (singleton).
    
    Args:
        base_dir: Direktori root project
        config_file: Path file konfigurasi utama
        env_prefix: Prefix untuk environment variables
        
    Returns:
        Instance ConfigManager
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(base_dir, config_file, env_prefix)
    return _config_manager