"""
File: smartcash/common/config/colab_manager.py
Deskripsi: Pengelolaan konfigurasi untuk Google Colab dengan integrasi Google Drive
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from smartcash.common.config.drive_manager import DriveConfigManager
from smartcash.common.constants.paths import COLAB_PATH, DRIVE_PATH
from smartcash.common.constants.core import DEFAULT_CONFIG_DIR, APP_NAME

class ColabConfigManager(DriveConfigManager):
    """
    Pengelolaan konfigurasi untuk Google Colab dengan integrasi Google Drive
    """
    
    def __init__(self, *args, **kwargs):
        """
        Inisialisasi Colab config manager
        """
        super().__init__(*args, **kwargs)
        self._setup_colab_environment()
    
    def _setup_colab_environment(self) -> None:
        """
        Setup environment Colab dan mount Google Drive
        """
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Set base config path di Google Drive
            self._drive_base_path = Path(f"{DRIVE_PATH}/{DEFAULT_CONFIG_DIR}")
            self._drive_base_path.mkdir(parents=True, exist_ok=True)
            
            # Set local config path
            self._local_base_path = Path(f"{COLAB_PATH}/{APP_NAME}/{DEFAULT_CONFIG_DIR}")
            self._local_base_path.mkdir(parents=True, exist_ok=True)
            
            if self._logger:
                self._logger.info("Google Drive mounted dan environment Colab disiapkan")
        except ImportError:
            if self._logger:
                self._logger.error("Tidak dapat mengimpor google.colab")
            raise
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error saat setup Colab environment: {str(e)}")
            raise
    
    def _resolve_config_path(self, config_file: str) -> Path:
        """
        Resolve path konfigurasi untuk environment Colab
        
        Args:
            config_file: Path ke file konfigurasi
            
        Returns:
            Path ke file konfigurasi
        """
        # Jika path absolut, gunakan path tersebut
        if os.path.isabs(config_file):
            return Path(config_file)
        
        # Jika path relatif, gabungkan dengan base path
        return self._local_base_path / config_file
    
    def _get_drive_config_path(self, config_file: str) -> Path:
        """
        Dapatkan path konfigurasi di Google Drive
        
        Args:
            config_file: Path ke file konfigurasi lokal
            
        Returns:
            Path ke file konfigurasi di Google Drive
        """
        # Jika path absolut, konversi ke relatif
        if os.path.isabs(config_file):
            config_file = os.path.basename(config_file)
        
        return self._drive_base_path / config_file
    
    def sync_with_drive(self, config_file: str, sync_strategy: str = 'drive_priority') -> Tuple[bool, str, Dict[str, Any]]:
        """
        Sinkronisasi file konfigurasi dengan Google Drive di environment Colab
        
        Args:
            config_file: Path ke file konfigurasi
            sync_strategy: Strategi sinkronisasi ('merge', 'drive_priority', 'local_priority')
            
        Returns:
            Tuple (success, message, merged_config)
        """
        try:
            # Resolve path konfigurasi
            local_path = self._resolve_config_path(config_file)
            drive_path = self._get_drive_config_path(config_file)
            
            # Load konfigurasi lokal jika ada
            local_config = {}
            if local_path.exists():
                local_config = self._load_config_file(local_path)
            
            # Load konfigurasi dari Drive jika ada
            drive_config = {}
            if drive_path.exists():
                drive_config = self._load_config_file(drive_path)
            
            # Merge konfigurasi berdasarkan strategi
            merged_config = self._merge_configs(local_config, drive_config, sync_strategy)
            
            # Simpan konfigurasi yang sudah di-merge
            if sync_strategy in ['merge', 'local_priority']:
                self._save_config_file(local_path, merged_config)
            
            if sync_strategy in ['merge', 'drive_priority']:
                self._save_config_file(drive_path, merged_config)
            
            return True, "Sinkronisasi berhasil", merged_config
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error saat sinkronisasi dengan Drive: {str(e)}")
            return False, str(e), {}
    
    def _merge_configs(self, local_config: Dict[str, Any], drive_config: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """
        Merge konfigurasi lokal dan Drive berdasarkan strategi
        
        Args:
            local_config: Konfigurasi lokal
            drive_config: Konfigurasi dari Drive
            strategy: Strategi merge
            
        Returns:
            Konfigurasi yang sudah di-merge
        """
        if strategy == 'drive_priority':
            return {**local_config, **drive_config}
        elif strategy == 'local_priority':
            return {**drive_config, **local_config}
        else:  # merge
            return {**local_config, **drive_config}
    
    def _save_config_file(self, path: Path, config: Dict[str, Any]) -> None:
        """
        Simpan konfigurasi ke file
        
        Args:
            path: Path ke file konfigurasi
            config: Konfigurasi yang akan disimpan
        """
        from smartcash.common.io import save_config
        save_config(path, config) 