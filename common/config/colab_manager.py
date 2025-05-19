"""
File: smartcash/common/config/colab_manager.py
Deskripsi: Pengelolaan konfigurasi untuk Google Colab dengan integrasi Google Drive
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

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
        self._config_files = self._get_config_files()
    
    def _setup_colab_environment(self) -> None:
        """
        Setup environment Colab dan mount Google Drive
        """
        try:
            # Cek apakah Google Drive sudah terhubung
            drive_connected = False
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                drive_connected = True
                if self._logger:
                    self._logger.info("Google Drive berhasil di-mount")
            except ImportError:
                if self._logger:
                    self._logger.warning("Google Drive tidak tersedia, menggunakan konfigurasi lokal")
            except Exception as e:
                if self._logger:
                    self._logger.warning(f"Gagal mount Google Drive: {str(e)}")
            
            # Set base config path di Google Drive jika terhubung
            if drive_connected:
                self._drive_base_path = Path(DRIVE_PATH) / APP_NAME / DEFAULT_CONFIG_DIR
                self._drive_base_path.mkdir(parents=True, exist_ok=True)
            
            # Set local config path
            self._local_base_path = Path(COLAB_PATH) / APP_NAME / DEFAULT_CONFIG_DIR
            self._local_base_path.mkdir(parents=True, exist_ok=True)
            
            # Inisialisasi konfigurasi pertama kali
            self._initialize_first_time_config(drive_connected)
            
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error saat setup Colab environment: {str(e)}")
            raise
    
    def _initialize_first_time_config(self, drive_connected: bool) -> None:
        """
        Inisialisasi konfigurasi untuk pertama kali
        
        Args:
            drive_connected: Apakah Google Drive terhubung
        """
        # Path sumber konfigurasi default
        default_config_path = Path('/content/smartcash/configs')
        
        # Jika Google Drive terhubung, cek konfigurasi di Drive
        if drive_connected:
            drive_config_path = Path(DRIVE_PATH) / APP_NAME / DEFAULT_CONFIG_DIR
            if drive_config_path.exists():
                # Copy dari Drive ke local jika ada
                self._copy_config_files(drive_config_path, self._local_base_path)
                if self._logger:
                    self._logger.info("Konfigurasi diinisialisasi dari Google Drive")
                return
        
        # Jika tidak ada di Drive atau Drive tidak terhubung, cek konfigurasi default
        if default_config_path.exists():
            # Copy dari default ke local
            self._copy_config_files(default_config_path, self._local_base_path)
            if self._logger:
                self._logger.info("Konfigurasi diinisialisasi dari default config")
            
            # Jika Drive terhubung, copy ke Drive juga
            if drive_connected:
                self._copy_config_files(default_config_path, self._drive_base_path)
                if self._logger:
                    self._logger.info("Konfigurasi default disalin ke Google Drive")
    
    def _copy_config_files(self, src_path: Path, dst_path: Path) -> None:
        """
        Copy file konfigurasi dari satu lokasi ke lokasi lain
        
        Args:
            src_path: Path sumber
            dst_path: Path tujuan
        """
        try:
            # Pastikan direktori tujuan ada
            dst_path.mkdir(parents=True, exist_ok=True)
            
            # Copy semua file .yaml dan .yml
            for ext in ['.yaml', '.yml']:
                for src_file in src_path.glob(f'*{ext}'):
                    if src_file.is_file() and not src_file.name.startswith('.'):
                        dst_file = dst_path / src_file.name
                        shutil.copy2(src_file, dst_file)
                        if self._logger:
                            self._logger.debug(f"File {src_file.name} disalin ke {dst_path}")
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error saat menyalin file konfigurasi: {str(e)}")
            raise
    
    def _get_config_files(self) -> List[str]:
        """
        Dapatkan daftar file konfigurasi yang tersedia
        
        Returns:
            List nama file konfigurasi
        """
        config_files = []
        for ext in ['.yaml', '.yml']:
            config_files.extend([
                f.name for f in self._local_base_path.glob(f'*{ext}')
                if f.is_file() and not f.name.startswith('.')
            ])
        return sorted(config_files)
    
    def get_available_configs(self) -> List[str]:
        """
        Dapatkan daftar konfigurasi yang tersedia
        
        Returns:
            List nama konfigurasi
        """
        return self._config_files
    
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
    
    def sync_all_configs(self, sync_strategy: str = 'drive_priority') -> Dict[str, Tuple[bool, str]]:
        """
        Sinkronisasi semua file konfigurasi dengan Google Drive
        
        Args:
            sync_strategy: Strategi sinkronisasi ('merge', 'drive_priority', 'local_priority')
            
        Returns:
            Dictionary dengan hasil sinkronisasi untuk setiap file
        """
        results = {}
        for config_file in self._config_files:
            success, message, _ = self.sync_with_drive(config_file, sync_strategy)
            results[config_file] = (success, message)
        return results
    
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
        save_config(str(path), config) 