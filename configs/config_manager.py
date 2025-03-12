# File: smartcash/config/config_manager.py
# Author: Alfrida Sabar
# Deskripsi: Manager untuk konfigurasi aplikasi dengan validasi dan pengelolaan yang robust

import os
import yaml
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import copy
import re

from smartcash.utils.logger import global_logger
from smartcash.exceptions.base import ConfigError

class ConfigManager:
    """
    Manager terpusat untuk pengelolaan konfigurasi aplikasi.
    Mendukung loading dari file YAML, validasi, dan overriding.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        defaults: Optional[Dict[str, Any]] = None,
        env_prefix: str = "SMARTCASH_"
    ):
        """
        Inisialisasi ConfigManager.
        
        Args:
            config_path: Path ke file konfigurasi (opsional)
            defaults: Konfigurasi default (opsional)
            env_prefix: Prefix untuk variabel lingkungan
        """
        self.logger = global_logger("config_manager")
        self.env_prefix = env_prefix
        
        # Konfigurasi default minimal
        self._defaults = {
            'app_name': "SmartCash",
            'version': "1.0.0",
            'data_dir': "data",
            'output_dir': "runs/train",
            'logs_dir': "logs",
            'layers': ['banknote', 'nominal', 'security'],
            'model': {
                'backbone': "efficientnet_b4",
                'framework': "YOLOv5"
            }
        }
        
        # Update defaults jika disediakan
        if defaults:
            self._update_dict(self._defaults, defaults)
        
        # Konfigurasi aktif
        self._config = copy.deepcopy(self._defaults)
        
        # Load dari file jika disediakan
        if config_path:
            self.load_from_file(config_path)
        
        # Override dari environment variables
        self._override_from_env()
        
        self.logger.info(f"🔧 ConfigManager diinisialisasi dengan {config_path or 'konfigurasi default'}")
    
    def load_from_file(self, config_path: str) -> Dict[str, Any]:
        """
        Load konfigurasi dari file YAML.
        
        Args:
            config_path: Path ke file konfigurasi
            
        Returns:
            Konfigurasi yang telah dimuat
            
        Raises:
            ConfigError: Jika file tidak ditemukan atau tidak valid
        """
        try:
            path = Path(config_path)
            
            if not path.exists():
                raise ConfigError(f"File konfigurasi tidak ditemukan: {config_path}")
                
            with open(path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
                
            # Update konfigurasi aktif
            self._update_dict(self._config, loaded_config)
            
            self.logger.info(f"✅ Konfigurasi berhasil dimuat dari {path}")
            return self._config
            
        except yaml.YAMLError as e:
            error_msg = f"File konfigurasi tidak valid: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            raise ConfigError(error_msg)
            
        except Exception as e:
            error_msg = f"Gagal memuat konfigurasi: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            raise ConfigError(error_msg)
    
    def _update_dict(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """
        Update dictionary secara rekursif.
        
        Args:
            base: Dictionary dasar
            update: Dictionary update
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._update_dict(base[key], value)
            else:
                base[key] = value
    
    def _override_from_env(self) -> None:
        """Override konfigurasi dari variabel lingkungan."""
        env_vars = {k: v for k, v in os.environ.items() if k.startswith(self.env_prefix)}
        
        for key, value in env_vars.items():
            # Remove prefix dan konversi ke lowercase
            config_key = key[len(self.env_prefix):].lower()
            
            # Handle nested keys dengan format SMARTCASH_MODEL_BACKBONE
            if '_' in config_key:
                parts = config_key.split('_')
                
                # Build nested dict path
                current = self._config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    elif not isinstance(current[part], dict):
                        # Jika key ada tapi bukan dict, ubah ke dict
                        current[part] = {}
                    current = current[part]
                
                # Set nilai akhir
                current[parts[-1]] = self._parse_env_value(value)
            else:
                # Set nilai langsung
                self._config[config_key] = self._parse_env_value(value)
    
    def _parse_env_value(self, value: str) -> Any:
        """
        Parse nilai dari variabel lingkungan ke tipe yang sesuai.
        
        Args:
            value: Nilai string dari variabel lingkungan
            
        Returns:
            Nilai yang telah dikonversi ke tipe yang sesuai
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
            else:
                return int(value)
        except ValueError:
            pass
            
        # List (comma separated)
        if ',' in value:
            return [self._parse_env_value(item.strip()) for item in value.split(',')]
            
        # JSON
        if value.startswith('{') or value.startswith('['):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
                
        # Default: string
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Dapatkan nilai konfigurasi dengan dot notation.
        
        Args:
            key: Key konfigurasi dengan dot notation (e.g. 'model.backbone')
            default: Nilai default jika tidak ditemukan
            
        Returns:
            Nilai konfigurasi
        """
        if '.' not in key:
            return self._config.get(key, default)
            
        parts = key.split('.')
        value = self._config
        
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set nilai konfigurasi dengan dot notation.
        
        Args:
            key: Key konfigurasi dengan dot notation
            value: Nilai yang akan di-set
        """
        if '.' not in key:
            self._config[key] = value
            return
            
        parts = key.split('.')
        current = self._config
        
        # Navigasi ke parent
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
            
        # Set nilai
        current[parts[-1]] = value
    
    def get_config(self) -> Dict[str, Any]:
        """
        Dapatkan seluruh konfigurasi aktif.
        
        Returns:
            Dictionary konfigurasi
        """
        return copy.deepcopy(self._config)
    
    def save_to_file(self, output_path: str) -> str:
        """
        Simpan konfigurasi ke file YAML.
        
        Args:
            output_path: Path output
            
        Returns:
            Path file yang disimpan
        """
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
                
            self.logger.info(f"✅ Konfigurasi disimpan ke {path}")
            return str(path)
            
        except Exception as e:
            error_msg = f"Gagal menyimpan konfigurasi: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            raise ConfigError(error_msg)
    
    def validate(self, schema: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validasi konfigurasi dengan schema.
        
        Args:
            schema: Schema validasi (opsional)
            
        Returns:
            True jika valid
            
        Raises:
            ConfigError: Jika konfigurasi tidak valid
        """
        # Validasi dasar tanpa schema
        if schema is None:
            # Pastikan konfigurasi minimal sudah ada
            required_keys = ['app_name', 'version', 'data_dir', 'output_dir']
            missing_keys = [key for key in required_keys if key not in self._config]
            
            if missing_keys:
                raise ConfigError(f"Konfigurasi tidak valid: key tidak ditemukan {missing_keys}")
                
            return True
            
        # TODO: Implement full schema validation
        # Validasi dengan schema lengkap dapat diimplementasikan di sini
        
        return True

# Singleton config manager (lazy initialization)
_config_manager = None

def get_config_manager(
    config_path: Optional[str] = None,
    defaults: Optional[Dict[str, Any]] = None
) -> ConfigManager:
    """
    Dapatkan instance ConfigManager singleton.
    
    Args:
        config_path: Path ke file konfigurasi (opsional)
        defaults: Konfigurasi default (opsional)
        
    Returns:
        Instance ConfigManager
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path, defaults)
    elif config_path is not None:
        # Reload konfigurasi jika path berbeda
        _config_manager.load_from_file(config_path)
        
    return _config_manager