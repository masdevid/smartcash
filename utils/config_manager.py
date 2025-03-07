"""
File: smartcash/utils/config_manager.py
Author: Alfrida Sabar
Deskripsi: Centralized configuration manager untuk memastikan konsistensi dalam loading, saving, dan manipulasi konfigurasi.
"""

import os
import yaml
import pickle
import copy
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

class ConfigManager:
    """
    Centralized manager untuk konfigurasi SmartCash.
    
    Menyediakan interface konsisten untuk:
    - Loading konfigurasi dari berbagai sumber
    - Saving konfigurasi dengan backup
    - Accessing dan updating nilai konfigurasi (termasuk deep updates)
    """
    
    def __init__(self, base_dir: Optional[str] = None, logger = None):
        """
        Initialize ConfigManager.
        
        Args:
            base_dir: Base directory yang berisi configs/ dan lainnya
            logger: Logger untuk mencatat aktivitas
        """
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
    
    def load(self, 
            filename: Optional[str] = None, 
            fallback_to_pickle: bool = True, 
            use_defaults: bool = True) -> Dict[str, Any]:
        """
        Load konfigurasi dari file dengan strategi fallback.
        
        Args:
            filename: Path lengkap atau nama file konfigurasi (optional)
            fallback_to_pickle: Menggunakan pickle sebagai backup jika file yaml tidak ditemukan
            use_defaults: Menggunakan konfigurasi default jika tidak ada file atau pickle ditemukan
            
        Returns:
            Dictionary konfigurasi
        """
        # Reset config
        self.config = {}
        
        # Definisikan file yang akan dicoba dimuat
        files_to_try = []
        
        if filename:
            # Jika full path diberikan
            if os.path.isabs(filename) or '/' in filename:
                files_to_try.append(filename)
            # Jika hanya nama file, tambahkan ke config_dir
            else:
                files_to_try.append(str(self.config_dir / filename))
        
        # Tambahkan default paths
        files_to_try.extend([
            str(self.experiment_config_path),
            str(self.default_config_path)
        ])
        
        # Coba memuat dari file yaml
        for file_path in files_to_try:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        self.config = yaml.safe_load(f)
                    if self.logger:
                        self.logger.info(f"📝 Konfigurasi dimuat dari {file_path}")
                    break
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"⚠️ Gagal memuat konfigurasi dari {file_path}: {str(e)}")
        
        # Fallback ke pickle jika diminta
        if not self.config and fallback_to_pickle and os.path.exists(self.pickle_path):
            try:
                with open(self.pickle_path, 'rb') as f:
                    self.config = pickle.load(f)
                if self.logger:
                    self.logger.info(f"📝 Konfigurasi dimuat dari {self.pickle_path}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"⚠️ Gagal memuat konfigurasi dari pickle: {str(e)}")
        
        # Fallback ke default jika belum ada yang