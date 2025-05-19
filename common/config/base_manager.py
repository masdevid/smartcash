"""
File: smartcash/common/config/base_manager.py
Deskripsi: Implementasi dasar untuk ConfigManager dengan fungsionalitas inti
"""

import os
import copy
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

from smartcash.common.io import (
    load_json,
    save_json,
    load_yaml,
    save_yaml,
    load_config,
    save_config,
    ensure_dir
)

from smartcash.common.config.singleton import Singleton

class BaseConfigManager(Singleton):
    """
    Implementasi dasar untuk ConfigManager dengan fungsionalitas inti seperti
    loading, saving, dan manipulasi konfigurasi
    """
    
    DEFAULT_CONFIG_DIR = 'configs'
    
    def __init__(self, base_dir: Optional[str] = None, config_file: Optional[str] = None, env_prefix: str = 'SMARTCASH_'):
        """
        Inisialisasi config manager dengan base directory, file konfigurasi utama, dan prefix environment variable
        
        Args:
            base_dir: Direktori dasar
            config_file: File konfigurasi utama
            env_prefix: Prefix untuk environment variables
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.config_dir = Path('configs')
        self.env_prefix = env_prefix
        self.config = {}
        self.module_configs = {}  # Dictionary untuk menyimpan konfigurasi berbagai modul
        
        # Setup logger jika tersedia
        try:
            from smartcash.common.logger import get_logger
            self._logger = get_logger("config_manager")
            # Hanya tampilkan log critical
            self._logger.setLevel("CRITICAL")
        except ImportError:
            self._logger = None
            
        if config_file: 
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load konfigurasi dari file YAML/JSON, dengan resolve path relatif dan override environment variables
        
        Args:
            config_file: Path ke file konfigurasi
            
        Returns:
            Dictionary konfigurasi
            
        Raises:
            FileNotFoundError: Jika file konfigurasi tidak ditemukan
        """
        config_path = self._resolve_config_path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")
        
        # Load konfigurasi menggunakan fungsi dari io module
        self.config = load_config(config_path, {})
        
        # Override dengan environment variables
        self._override_with_env_vars()
        
        return self.config
    
    def _resolve_config_path(self, config_file: str) -> Path:
        """
        Resolve path konfigurasi relatif atau absolut ke Path lengkap
        
        Args:
            config_file: Path ke file konfigurasi
            
        Returns:
            Path lengkap ke file konfigurasi
        """
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
        
        # Jika semua gagal, kembalikan path relatif terhadap config_dir
        return self.config_dir / config_path
    
    def _override_with_env_vars(self) -> None:
        """
        Override konfigurasi dengan environment variables menggunakan konvensi SMARTCASH_SECTION_KEY=value
        """
        for env_var, value in os.environ.items():
            if env_var.startswith(self.env_prefix):
                # Strip prefix dan split menjadi section dan key
                key_path = env_var[len(self.env_prefix):].lower().replace('_', '.')
                parsed_value = self._parse_env_value(value)
                
                # Set nilai di konfigurasi
                self.set(key_path, parsed_value)
    
    def _parse_env_value(self, value: str) -> Any:
        """
        Parse nilai environment variable ke tipe yang sesuai (bool, number, list, string)
        
        Args:
            value: Nilai string dari environment variable
            
        Returns:
            Nilai yang sudah diparse ke tipe yang sesuai
        """
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # List (comma-separated)
        if ',' in value:
            return [self._parse_env_value(item.strip()) for item in value.split(',')]
        
        # Default: string
        return value
    
    def get(self, key: str, default=None) -> Any:
        """
        Ambil nilai konfigurasi dengan dot notation (e.g., 'model.img_size.width')
        
        Args:
            key: Key dengan dot notation
            default: Nilai default jika key tidak ditemukan
            
        Returns:
            Nilai konfigurasi atau default jika tidak ditemukan
        """
        parts = key.split('.')
        config = self.config
        
        for part in parts:
            if isinstance(config, dict) and part in config:
                config = config[part]
            else:
                return default
                
        return config
    
    def set(self, key: str, value: Any) -> None:
        """
        Set nilai konfigurasi dengan dot notation (e.g., 'model.img_size.width')
        
        Args:
            key: Key dengan dot notation
            value: Nilai yang akan diset
        """
        parts = key.split('.')
        config = self.config
        
        for i, part in enumerate(parts[:-1]):
            if part not in config or not isinstance(config[part], dict):
                config[part] = {}
            config = config[part]
            
        config[parts[-1]] = value
    
    def merge_config(self, config: Union[Dict, str]) -> Dict[str, Any]:
        """
        Merge konfigurasi dari dict atau file dengan current config
        
        Args:
            config: Dictionary konfigurasi atau path ke file konfigurasi
            
        Returns:
            Dictionary konfigurasi yang sudah di-merge
        """
        if isinstance(config, str):
            config_path = self._resolve_config_path(config)
            if not config_path.exists():
                if self._logger:
                    self._logger.critical(f"File konfigurasi tidak ditemukan: {config_path}")
                return self.config
            
            config_data = load_config(config_path, {})
        else:
            config_data = config
            
        self._deep_merge(self.config, config_data)
        return self.config
    
    def _deep_merge(self, target: Dict, source: Dict) -> None:
        """
        Deep merge dua dictionary secara rekursif
        
        Args:
            target: Dictionary target
            source: Dictionary source yang akan di-merge ke target
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)
    
    def save_config(self, config_file: str, create_dirs: bool = True) -> bool:
        """
        Simpan konfigurasi ke file YAML/JSON
        
        Args:
            config_file: Path ke file konfigurasi
            create_dirs: Buat direktori jika belum ada
            
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            config_path = Path(config_file)
            
            # Buat direktori jika belum ada
            if create_dirs and not config_path.parent.exists():
                ensure_dir(config_path.parent)
            
            # Simpan konfigurasi
            save_config(config_path, self.config)
            return True
        except Exception as e:
            if self._logger:
                self._logger.critical(f"Error saat menyimpan konfigurasi: {str(e)}")
            return False
    
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
            Nilai yang sudah divalidasi atau default jika validasi gagal
        """
        # Jika value None, gunakan default
        if value is None:
            return default_value
        
        # Validasi tipe
        if valid_types:
            if not isinstance(valid_types, list):
                valid_types = [valid_types]
                
            if not any(isinstance(value, t) for t in valid_types):
                return default_value
        
        # Validasi nilai
        if valid_values and value not in valid_values:
            return default_value
            
        return value
    
    def __getitem__(self, key):
        """
        Operator [] untuk mengakses konfigurasi.
        """
        return self.get(key)
    
    def __setitem__(self, key, value):
        """
        Operator [] untuk mengatur konfigurasi.
        """
        self.set(key, value)
