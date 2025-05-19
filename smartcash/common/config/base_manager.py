"""
File: smartcash/common/config/base_manager.py
Deskripsi: Base class untuk pengelolaan konfigurasi
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from smartcash.common.io import load_config, save_config
from smartcash.common.constants.core import DEFAULT_CONFIG_DIR

class BaseConfigManager:
    """
    Base class untuk pengelolaan konfigurasi
    """
    
    def __init__(self, base_dir: str, config_file: str, logger: Optional[Any] = None):
        """
        Inisialisasi base config manager
        
        Args:
            base_dir: Base directory untuk konfigurasi
            config_file: Path ke file konfigurasi
            logger: Logger instance (opsional)
        """
        self._base_dir = Path(base_dir)
        self._config_file = config_file
        self._logger = logger
        self._config = {}
    
    @property
    def base_dir(self) -> Path:
        """
        Get base directory
        
        Returns:
            Path ke base directory
        """
        return self._base_dir
    
    @base_dir.setter
    def base_dir(self, value: str) -> None:
        """
        Set base directory
        
        Args:
            value: Path ke base directory
        """
        self._base_dir = Path(value)
    
    @property
    def config_file(self) -> str:
        """
        Get config file path
        
        Returns:
            Path ke file konfigurasi
        """
        return self._config_file
    
    @config_file.setter
    def config_file(self, value: str) -> None:
        """
        Set config file path
        
        Args:
            value: Path ke file konfigurasi
        """
        self._config_file = value
    
    @property
    def config(self) -> Dict[str, Any]:
        """
        Get current configuration
        
        Returns:
            Dictionary konfigurasi
        """
        return self._config
    
    def load_config(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Load konfigurasi dari file
        
        Args:
            config_file: Path ke file konfigurasi (opsional)
            
        Returns:
            Dictionary konfigurasi
        """
        if config_file is None:
            config_file = self._config_file
        
        # Resolve path konfigurasi
        config_path = self._resolve_config_path(config_file)
        
        # Load konfigurasi jika file ada
        if config_path.exists():
            try:
                self._config = load_config(config_path)
                if self._logger:
                    self._logger.info(f"Konfigurasi berhasil dimuat dari: {config_path}")
                return self._config
            except Exception as e:
                if self._logger:
                    self._logger.error(f"Error saat memuat konfigurasi: {str(e)}")
                raise
        
        # Jika file tidak ada, kembalikan konfigurasi kosong
        if self._logger:
            self._logger.warning(f"File konfigurasi tidak ditemukan: {config_path}")
        return {}
    
    def save_config(self, config: Dict[str, Any], config_file: Optional[str] = None) -> bool:
        """
        Simpan konfigurasi ke file
        
        Args:
            config: Konfigurasi yang akan disimpan
            config_file: Path ke file konfigurasi (opsional)
            
        Returns:
            True jika berhasil, False jika gagal
        """
        if config_file is None:
            config_file = self._config_file
        
        try:
            # Resolve path konfigurasi
            config_path = self._resolve_config_path(config_file)
            
            # Simpan konfigurasi
            save_config(config_path, config)
            self._config = config
            
            if self._logger:
                self._logger.info(f"Konfigurasi berhasil disimpan ke: {config_path}")
            return True
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error saat menyimpan konfigurasi: {str(e)}")
            return False
    
    def _resolve_config_path(self, config_file: str) -> Path:
        """
        Resolve path konfigurasi
        
        Args:
            config_file: Path ke file konfigurasi
            
        Returns:
            Path ke file konfigurasi
        """
        # Jika path absolut, gunakan path tersebut
        if os.path.isabs(config_file):
            return Path(config_file)
        
        # Jika path relatif, gabungkan dengan base path
        return self._base_dir / DEFAULT_CONFIG_DIR / config_file
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Dapatkan nilai konfigurasi
        
        Args:
            key: Key konfigurasi
            default: Nilai default jika key tidak ditemukan
            
        Returns:
            Nilai konfigurasi
        """
        return self._config.get(key, default)
    
    def set_config_value(self, key: str, value: Any) -> None:
        """
        Set nilai konfigurasi
        
        Args:
            key: Key konfigurasi
            value: Nilai konfigurasi
        """
        self._config[key] = value
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update konfigurasi
        
        Args:
            config: Dictionary konfigurasi
        """
        self._config.update(config)
