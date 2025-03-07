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
    
    def load_config(filename: Optional[str] = None,
                fallback_to_pickle: bool = True,
                default_config: Optional[Dict[str, Any]] = None,
                logger: Optional[Any] = None) -> Dict[str, Any]:
        """
        Muat konfigurasi dari file yaml atau pickle dengan prioritas yang jelas.
        
        Args:
            filename: Nama file konfigurasi (optional)
            fallback_to_pickle: Flag untuk memuat dari pickle jika yaml tidak ada
            default_config: Konfigurasi default jika tidak ada file yang ditemukan
            logger: Optional logger untuk pesan log
            
        Returns:
            Dictionary konfigurasi
        """
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
        
        # Coba memuat dari file yaml
        for file_path in files_to_try:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        config = yaml.safe_load(f)
                    if logger:
                        logger.info(f"📝 Konfigurasi dimuat dari {file_path}")
                    return config
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Gagal memuat konfigurasi dari {file_path}: {str(e)}")
        
        # Coba memuat dari pickle jika fallback_to_pickle=True
        if fallback_to_pickle and os.path.exists('config.pkl'):
            try:
                with open('config.pkl', 'rb') as f:
                    config = pickle.load(f)
                if logger:
                    logger.info("📝 Konfigurasi dimuat dari config.pkl")
                return config
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Gagal memuat konfigurasi dari config.pkl: {str(e)}")
        
        # Gunakan konfigurasi default jika semua gagal
        if default_config:
            if logger:
                logger.warning("⚠️ Menggunakan konfigurasi default")
            return default_config
        
        # Jika tidak ada default_config dan semua gagal, kembalikan dictionary kosong
        if logger:
            logger.warning("⚠️ Tidak ada konfigurasi yang dimuat, mengembalikan dictionary kosong")
        return {}