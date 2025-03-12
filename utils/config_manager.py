"""
File: smartcash/utils/config_manager.py
Author: Refactored
Deskripsi: Centralized configuration manager yang sudah direfactor dengan pattern singleton
           untuk memastikan konsistensi dalam loading, saving, dan manipulasi konfigurasi
           di semua cell notebook.
"""

import os
import yaml
import pickle
import copy
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

class ConfigManager:
    """
    Centralized manager untuk konfigurasi SmartCash dengan pattern singleton.
    
    Menyediakan interface konsisten untuk:
    - Loading konfigurasi dari berbagai sumber
    - Saving konfigurasi dengan backup
    - Accessing dan updating nilai konfigurasi (termasuk deep updates)
    - Sinkronisasi konfigurasi ke Google Drive
    """
    
    # Singleton instance dan lock
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implementasi singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, base_dir: Optional[str] = None, logger = None):
        """
        Initialize ConfigManager.
        
        Args:
            base_dir: Base directory yang berisi configs/ dan lainnya
            logger: Logger untuk mencatat aktivitas
        """
        # Skip jika sudah diinisialisasi
        if getattr(self, '_initialized', False):
            return
            
        self.base_dir = Path(base_dir) if base_dir else Path(os.getcwd())
        self.config_dir = self.base_dir / 'configs'
        self.logger = logger
        self.config = {}
        
        # Buat direktori konfigurasi jika belum ada
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Default file paths
        self.default_config_path = self.config_dir / 'base_config.yaml'
        self.experiment_config_path = self.config_dir / 'experiment_config.yaml'
        self.pickle_path = self.base_dir / 'config.pkl'
        
        # Google Drive integration
        self.is_colab = self._detect_colab()
        self.drive_mounted = False
        self.drive_path = None
        if self.is_colab:
            self._check_drive_mounted()
            
        self._initialized = True
    
    def _detect_colab(self) -> bool:
        """Deteksi jika berjalan di Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _check_drive_mounted(self) -> bool:
        """Periksa apakah Google Drive sudah di-mount."""
        if os.path.exists('/content/drive/MyDrive'):
            self.drive_mounted = True
            self.drive_path = Path('/content/drive/MyDrive/SmartCash')
            os.makedirs(self.drive_path, exist_ok=True)
            os.makedirs(self.drive_path / 'configs', exist_ok=True)
            return True
        return False
    
    def mount_drive(self) -> bool:
        """Mount Google Drive jika diperlukan."""
        if not self.is_colab:
            return False
            
        if self.drive_mounted:
            return True
            
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            
            self.drive_path = Path('/content/drive/MyDrive/SmartCash')
            os.makedirs(self.drive_path, exist_ok=True)
            os.makedirs(self.drive_path / 'configs', exist_ok=True)
            
            self.drive_mounted = True
            
            if self.logger:
                self.logger.info("✅ Google Drive berhasil di-mount")
                
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error saat mount Google Drive: {str(e)}")
            return False
    
    @classmethod
    def get_instance(cls, base_dir: Optional[str] = None, logger = None) -> 'ConfigManager':
        """
        Dapatkan instance singleton.
        
        Args:
            base_dir: Base directory
            logger: Logger
            
        Returns:
            Instance ConfigManager
        """
        return cls(base_dir, logger)
    
    @classmethod
    def load_config(cls, 
                filename: Optional[str] = None,
                fallback_to_pickle: bool = True,
                default_config: Optional[Dict[str, Any]] = None,
                logger: Optional[Any] = None,
                use_singleton: bool = True) -> Dict[str, Any]:
        """
        Muat konfigurasi dari file yaml atau pickle dengan prioritas yang jelas.
        
        Args:
            filename: Nama file konfigurasi (optional)
            fallback_to_pickle: Flag untuk memuat dari pickle jika yaml tidak ada
            default_config: Konfigurasi default jika tidak ada file yang ditemukan
            logger: Optional logger untuk pesan log
            use_singleton: Gunakan singleton instance untuk menyimpan config
            
        Returns:
            Dictionary konfigurasi
        """
        # Jika use_singleton, gunakan atau buat instance
        cm = None
        if use_singleton:
            cm = cls.get_instance(logger=logger)
            
            # Return config yang sudah ada jika tidak kosong
            if cm.config and not filename:
                return cm.config
        
        config = {}
        
        # Definisikan file yang akan dicoba dimuat
        files_to_try = []
        if filename:
            # If a full path is provided
            if os.path.isabs(filename) or '/' in filename:
                files_to_try.append(filename)
            else:
                # If just a filename, append to configs directory
                files_to_try.append(os.path.join('configs', filename))
        
        # Add default files to try
        files_to_try.extend([
            'configs/experiment_config.yaml',
            'configs/training_config.yaml',
            'configs/base_config.yaml'
        ])
        
        # Coba drive path jika tersedia
        if cm and cm.is_colab and cm.drive_mounted:
            for file_path in files_to_try:
                drive_path = os.path.join('/content/drive/MyDrive/SmartCash', file_path)
                if os.path.exists(drive_path):
                    files_to_try.insert(0, drive_path)  # Add to start of list
        
        # Coba memuat dari file yaml
        for file_path in files_to_try:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        config = yaml.safe_load(f)
                    if logger:
                        logger.info(f"📝 Konfigurasi dimuat dari {file_path}")
                    
                    # Simpan ke singleton jika digunakan
                    if use_singleton and cm:
                        cm.config = config
                        
                    return config
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Gagal memuat konfigurasi dari {file_path}: {str(e)}")
        
        # Coba memuat dari pickle jika fallback_to_pickle=True
        if fallback_to_pickle:
            pickle_files = ['config.pkl']
            
            # Tambahkan drive path jika tersedia
            if cm and cm.is_colab and cm.drive_mounted:
                pickle_files.insert(0, str(cm.drive_path / 'config.pkl'))
                
            for pickle_path in pickle_files:
                if os.path.exists(pickle_path):
                    try:
                        with open(pickle_path, 'rb') as f:
                            config = pickle.load(f)
                        if logger:
                            logger.info(f"📝 Konfigurasi dimuat dari {pickle_path}")
                        
                        # Simpan ke singleton jika digunakan
                        if use_singleton and cm:
                            cm.config = config
                            
                        return config
                    except Exception as e:
                        if logger:
                            logger.warning(f"⚠️ Gagal memuat konfigurasi dari {pickle_path}: {str(e)}")
        
        # Gunakan konfigurasi default jika semua gagal
        if default_config:
            if logger:
                logger.warning("⚠️ Menggunakan konfigurasi default")
                
            # Simpan ke singleton jika digunakan
            if use_singleton and cm:
                cm.config = default_config
                
            return default_config
        
        # Jika tidak ada default_config dan semua gagal, kembalikan dictionary kosong
        if logger:
            logger.warning("⚠️ Tidak ada konfigurasi yang dimuat, mengembalikan dictionary kosong")
            
        # Simpan ke singleton jika digunakan
        if use_singleton and cm:
            cm.config = {}
            
        return {}
    
    def save_config(self, 
                  config: Optional[Dict[str, Any]] = None,
                  filename: Optional[str] = None,
                  backup: bool = True,
                  sync_to_drive: bool = True) -> bool:
        """
        Simpan konfigurasi ke file dengan backup opsional.
        
        Args:
            config: Konfigurasi yang akan disimpan (gunakan self.config jika None)
            filename: Nama file (opsional)
            backup: Buat backup sebelum menyimpan
            sync_to_drive: Sinkronkan ke Google Drive jika tersedia
            
        Returns:
            Bool sukses/gagal
        """
        # Gunakan config instance jika tidak disediakan
        if config is None:
            config = self.config
            
        # Gunakan default filename jika tidak disediakan
        if filename is None:
            filename = self.default_config_path
        else:
            # Tambahkan base path jika bukan path absolut
            if not os.path.isabs(filename):
                filename = self.config_dir / filename
        
        # Pastikan direktori ada
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Buat backup jika diperlukan
        if backup and os.path.exists(filename):
            backup_path = f"{filename}.bak"
            try:
                shutil.copy2(filename, backup_path)
                if self.logger:
                    self.logger.info(f"📝 Backup konfigurasi dibuat: {backup_path}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"⚠️ Gagal membuat backup: {str(e)}")
        
        # Simpan ke file
        try:
            # Simpan ke YAML
            with open(filename, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Simpan juga ke pickle untuk backup
            pickle_path = str(Path(filename).with_suffix('.pkl'))
            with open(pickle_path, 'wb') as f:
                pickle.dump(config, f)
                
            if self.logger:
                self.logger.info(f"💾 Konfigurasi disimpan ke {filename}")
                
            # Sinkronkan ke drive jika diperlukan
            if sync_to_drive and self.is_colab and self.drive_mounted:
                self.sync_to_drive(config, os.path.basename(filename))
                
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
            return False
    
    def sync_to_drive(self, 
                    config: Optional[Dict[str, Any]] = None,
                    filename: str = 'base_config.yaml') -> bool:
        """
        Sinkronkan konfigurasi ke Google Drive.
        
        Args:
            config: Konfigurasi yang akan disimpan (gunakan self.config jika None)
            filename: Nama file konfigurasi
            
        Returns:
            Bool sukses/gagal
        """
        if not self.is_colab:
            if self.logger:
                self.logger.warning("⚠️ Tidak berjalan di Google Colab, sync_to_drive dilewati")
            return False
            
        if not self.drive_mounted:
            mounted = self.mount_drive()
            if not mounted:
                if self.logger:
                    self.logger.warning("⚠️ Google Drive tidak ter-mount, sync_to_drive gagal")
                return False
        
        # Gunakan config instance jika tidak disediakan
        if config is None:
            config = self.config
        
        # Pastikan direktori ada
        drive_config_dir = self.drive_path / 'configs'
        os.makedirs(drive_config_dir, exist_ok=True)
        
        # Simpan ke drive
        try:
            # Simpan ke YAML
            drive_config_path = drive_config_dir / filename
            with open(drive_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Simpan juga ke pickle untuk backup
            pickle_path = drive_config_dir / Path(filename).with_suffix('.pkl')
            with open(pickle_path, 'wb') as f:
                pickle.dump(config, f)
                
            if self.logger:
                self.logger.info(f"💾 Konfigurasi disinkronkan ke Google Drive: {drive_config_path}")
                
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error saat menyinkronkan ke Drive: {str(e)}")
            return False
    
    def load_from_drive(self, filename: str = 'base_config.yaml') -> Dict[str, Any]:
        """
        Muat konfigurasi dari Google Drive.
        
        Args:
            filename: Nama file konfigurasi
            
        Returns:
            Dict konfigurasi
        """
        if not self.is_colab:
            if self.logger:
                self.logger.warning("⚠️ Tidak berjalan di Google Colab, load_from_drive dilewati")
            return {}
            
        if not self.drive_mounted:
            mounted = self.mount_drive()
            if not mounted:
                if self.logger:
                    self.logger.warning("⚠️ Google Drive tidak ter-mount, load_from_drive gagal")
                return {}
        
        # Coba load dari drive
        try:
            drive_config_path = self.drive_path / 'configs' / filename
            
            if drive_config_path.exists():
                with open(drive_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Update instance config
                self.config = config
                
                if self.logger:
                    self.logger.info(f"📝 Konfigurasi dimuat dari Google Drive: {drive_config_path}")
                    
                return config
            
            # Coba load dari pickle jika yaml tidak ada
            pickle_path = self.drive_path / 'configs' / Path(filename).with_suffix('.pkl')
            if pickle_path.exists():
                with open(pickle_path, 'rb') as f:
                    config = pickle.load(f)
                
                # Update instance config
                self.config = config
                
                if self.logger:
                    self.logger.info(f"📝 Konfigurasi dimuat dari Google Drive: {pickle_path}")
                    
                return config
                
            if self.logger:
                self.logger.warning(f"⚠️ File konfigurasi tidak ditemukan di Google Drive: {drive_config_path}")
                
            return {}
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error saat load dari Drive: {str(e)}")
            return {}
    
    def update_config(self, 
                     updates: Dict[str, Any], 
                     save: bool = True, 
                     filename: Optional[str] = None,
                     sync_to_drive: bool = True) -> Dict[str, Any]:
        """
        Update konfigurasi dengan nilai baru secara rekursif.
        
        Args:
            updates: Dict dengan nilai yang akan diupdate
            save: Flag untuk menyimpan perubahan ke file
            filename: Nama file untuk penyimpanan (opsional)
            sync_to_drive: Sinkronkan ke Google Drive jika tersedia
            
        Returns:
            Dict konfigurasi yang sudah diupdate
        """
        # Deep update config
        self._deep_update(self.config, updates)
        
        # Simpan jika diperlukan
        if save:
            self.save_config(self.config, filename, sync_to_drive=sync_to_drive)
        
        return self.config
    
    def get_config(self) -> Dict[str, Any]:
        """
        Dapatkan konfigurasi saat ini.
        
        Returns:
            Dict konfigurasi
        """
        return self.config
    
    def _deep_update(self, target: Dict, source: Dict) -> None:
        """
        Update nested dictionary secara rekursif.
        
        Args:
            target: Dict tujuan yang akan diupdate
            source: Dict sumber dengan nilai-nilai baru
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value

# Fungsi helper untuk mempermudah akses ke singleton manager
def get_config_manager(logger = None) -> ConfigManager:
    """
    Dapatkan instance singleton ConfigManager.
    
    Args:
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Instance ConfigManager
    """
    return ConfigManager.get_instance(logger=logger)