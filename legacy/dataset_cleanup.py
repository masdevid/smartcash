"""
File: smartcash/handlers/dataset_cleanup.py
Author: Alfrida Sabar
Deskripsi: Handler untuk membersihkan dataset, menghapus file augmentasi atau file yang tidak valid
"""

import yaml
from pathlib import Path
from typing import Dict, Optional

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.dataset.dataset_cleaner import DatasetCleaner

class DatasetCleanupHandler:
    """
    Handler untuk membersihkan dan memvalidasi dataset.
    Bekerja sebagai adapter untuk DatasetCleaner.
    """
    
    def __init__(
        self,
        config_path: str,
        data_dir: str = "data",
        backup_dir: Optional[str] = "backup",
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi dataset cleanup handler.
        
        Args:
            config_path: Path ke file konfigurasi
            data_dir: Direktori dataset
            backup_dir: Direktori backup (opsional)
            logger: Logger kustom
        """
        self.logger = logger or SmartCashLogger(__name__)
        self.config = self._load_config(config_path)
        self.data_dir = Path(data_dir)
        self.backup_dir = Path(backup_dir) if backup_dir else None
        
        # Buat instance DatasetCleaner
        self.cleaner = DatasetCleaner(
            config=self.config,
            data_dir=self.data_dir,
            logger=self.logger
        )
        
        # Statistik pembersihan (menggunakan dari cleaner)
        self.stats = self.cleaner.stats
        
    def _load_config(self, config_path: str) -> Dict:
        """
        Load konfigurasi pembersihan.
        
        Args:
            config_path: Path ke file konfigurasi
            
        Returns:
            Dict konfigurasi
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Default cleanup patterns jika tidak ada di config
        if 'cleanup' not in config:
            config['cleanup'] = {
                'augmentation_patterns': [
                    r'aug_.*',
                    r'.*_augmented.*',
                    r'.*_modified.*'
                ],
                'ignored_patterns': [
                    r'.*\.gitkeep',
                    r'.*\.DS_Store'
                ]
            }
            
        return config
        
    def cleanup(
        self,
        augmented_only: bool = True,
        create_backup: bool = True
    ) -> Dict:
        """
        Bersihkan dataset.
        
        Args:
            augmented_only: Hanya hapus file hasil augmentasi
            create_backup: Buat backup sebelum menghapus
            
        Returns:
            Dict statistik pembersihan
        """
        return self.cleaner.cleanup(
            augmented_only=augmented_only,
            create_backup=create_backup,
            backup_dir=self.backup_dir
        )