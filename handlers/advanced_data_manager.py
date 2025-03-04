# File: smartcash/handlers/advanced_data_manager.py
# Author: Alfrida Sabar
# Deskripsi: Handler data dengan dukungan preprocessing dan augmentasi

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.data_handler import DataHandler
from smartcash.utils.dataset_augmentation_utils import DatasetProcessor

class AdvancedDataManager:
    """Pengelola data SmartCash dengan dukungan preprocessing dan augmentasi."""
    
    def __init__(self, config, logger=None):
        """
        Inisialisasi data manager.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger untuk output
        """
        self.config = config
        self.logger = logger or SmartCashLogger("advanced_data_manager")
        
        # Inisialisasi DataHandler dasar
        self.data_handler = DataHandler(config=config)
        
        # Setup data directory
        self.data_dir = Path(config.get('data_dir', 'data'))
        
        # Inisialisasi dataset processor
        self.dataset_processor = DatasetProcessor(
            data_dir=str(self.data_dir),
            logger=self.logger
        )
    
    def get_dataloaders(self, batch_size=None):
        """
        Dapatkan DataLoader untuk training, validation, dan testing.
        
        Args:
            batch_size: Ukuran batch (jika None, ambil dari config)
            
        Returns:
            Dict DataLoader untuk train, val, dan test
        """
        batch_size = batch_size or self.config.get('training', {}).get('batch_size', 16)
        num_workers = min(4, self.config.get('model', {}).get('workers', 4))
        
        self.logger.info(f"🔄 Mempersiapkan dataloader dengan batch size {batch_size}, workers {num_workers}")
        
        # Dapatkan dataloaders
        train_loader = self.data_handler.get_train_loader(
            batch_size=batch_size, 
            num_workers=num_workers
        )
        
        val_loader = self.data_handler.get_val_loader(
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        test_loader = self.data_handler.get_test_loader(
            batch_size=batch_size,
            num_workers=2  # Kurangi workers untuk pengujian
        )
        
        dataloaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        
        for name, loader in dataloaders.items():
            self.logger.info(f"📦 {name.capitalize()} dataset: {len(loader)} batches")
            
        return dataloaders
    
    def augment_data(self, split='train', augmentation_type='combined', num_workers=4):
        """
        Augmentasi data untuk satu split dataset.
        
        Args:
            split: Split dataset yang akan diaugmentasi ('train', 'valid', 'test')
            augmentation_type: Tipe augmentasi ('position', 'lighting', 'combined')
            num_workers: Jumlah worker untuk multiprocessing
            
        Returns:
            Dict hasil augmentasi
        """
        self.logger.info(f"🔄 Memulai augmentasi data {split} dengan tipe {augmentation_type}")
        
        # Jalankan augmentasi
        stats = self.dataset_processor.augment_dataset(
            split=split,
            augmentation_type=augmentation_type,
            num_workers=num_workers
        )
        
        return stats
    
    def clean_augmented_data(self, splits=None):
        """
        Bersihkan file hasil augmentasi.
        
        Args:
            splits: List split yang akan dibersihkan (jika None, semua split)
            
        Returns:
            Dict statistik pembersihan
        """
        if splits is None:
            splits = ['train', 'valid', 'test']
            
        self.logger.info(f"🧹 Membersihkan data hasil augmentasi untuk {', '.join(splits)}")
        
        # Jalankan pembersihan
        stats = self.dataset_processor.clean_augmented_data(splits=splits)
        
        return stats
    
    def get_dataset_stats(self, detailed=False):
        """
        Dapatkan statistik dataset.
        
        Args:
            detailed: Jika True, tampilkan statistik lebih detail
            
        Returns:
            DataFrame statistik
        """
        self.logger.info("📊 Mengumpulkan statistik dataset...")
        
        # Dapatkan statistik dari processor
        stats = self.dataset_processor.get_dataset_stats()
        
        # Konversi ke DataFrame
        df_data = []
        for split, split_stats in stats.items():
            row = {'Split': split.capitalize()}
            row.update(split_stats)
            df_data.append(row)
            
        df = pd.DataFrame(df_data)
        
        # Tambahkan total
        total_row = {'Split': 'Total'}
        for col in df.columns:
            if col != 'Split':
                total_row[col] = df[col].sum()
        
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
        
        if detailed:
            self.logger.info(f"📊 Statistik Dataset:\n{df}")
            
        return df
    
    def split_dataset(self, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
        """
        Split dataset menjadi train, valid, dan test.
        
        Args:
            train_ratio: Rasio data training
            valid_ratio: Rasio data validasi
            test_ratio: Rasio data testing
            
        Returns:
            Dict statistik split
        """
        self.logger.info(
            f"🔄 Memulai split dataset dengan rasio "
            f"train={train_ratio}, valid={valid_ratio}, test={test_ratio}"
        )
        
        # Jalankan split
        stats = self.dataset_processor.split_dataset(
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio
        )
        
        return stats