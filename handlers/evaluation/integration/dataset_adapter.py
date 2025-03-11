# File: smartcash/handlers/evaluation/integration/dataset_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Adapter untuk akses dataset dan pembuatan dataloader untuk evaluasi

import os
import torch
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from torch.utils.data import DataLoader

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.handlers.dataset_manager import DatasetManager

class DatasetAdapter:
    """
    Adapter untuk DatasetManager.
    Menyediakan antarmuka untuk akses dataset dan pembuatan dataloader untuk evaluasi.
    """
    
    def __init__(
        self, 
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi DatasetAdapter.
        
        Args:
            config: Konfigurasi untuk evaluasi
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.logger = logger or get_logger("dataset_adapter")
        
        # Setup DatasetManager
        try:
            self.dataset_manager = DatasetManager(
                config=config,
                data_dir=config.get('data_dir', 'data')
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal menginisialisasi DatasetManager: {str(e)}")
            self.dataset_manager = None
        
        # Konfigurasi dataset
        self.dataset_config = self.config.get('data', {})
        self.train_dir = self.dataset_config.get('train_dir', 'data/train')
        self.valid_dir = self.dataset_config.get('valid_dir', 'data/valid')
        self.test_dir = self.dataset_config.get('test_dir', 'data/test')
        
        # Konfigurasi dataloader
        self.dataloader_config = self.dataset_config.get('preprocessing', {})
        self.batch_size = self.dataloader_config.get('batch_size', 16)
        self.num_workers = self.dataloader_config.get('num_workers', 4)
        self.img_size = self.dataloader_config.get('img_size', [640, 640])
        
        self.logger.debug(f"🔧 DatasetAdapter diinisialisasi (batch_size={self.batch_size}, num_workers={self.num_workers})")
    
    def get_dataset(
        self,
        dataset_path: str,
        split: str = 'test'
    ) -> torch.utils.data.Dataset:
        """
        Dapatkan dataset untuk evaluasi.
        
        Args:
            dataset_path: Path ke dataset
            split: Split dataset ('train', 'valid', 'test')
            
        Returns:
            Dataset PyTorch
            
        Raises:
            FileNotFoundError: Jika dataset tidak ditemukan
            RuntimeError: Jika gagal memuat dataset
        """
        try:
            # Cek keberadaan direktori
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"❌ Dataset tidak ditemukan: {dataset_path}")
            
            self.logger.info(f"🔄 Memuat dataset dari {dataset_path}...")
            
            # Gunakan DatasetManager untuk memuat dataset
            if self.dataset_manager:
                dataset = self.dataset_manager.get_dataset(split)
                self.logger.info(f"✅ Dataset berhasil dimuat: {len(dataset)} sampel")
                return dataset
            else:
                # Fallback ke implementasi manual jika tidak ada DatasetManager
                raise NotImplementedError("DatasetManager tidak tersedia, implementasi manual belum diimplementasikan")
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memuat dataset: {str(e)}")
            raise RuntimeError(f"Gagal memuat dataset: {str(e)}")
    
    def get_dataloader(
        self,
        dataset_path: str,
        split: str = 'test',
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        shuffle: bool = False
    ) -> DataLoader:
        """
        Dapatkan dataloader untuk evaluasi.
        
        Args:
            dataset_path: Path ke dataset
            split: Split dataset ('train', 'valid', 'test')
            batch_size: Ukuran batch
            num_workers: Jumlah worker
            shuffle: Acak dataset
            
        Returns:
            DataLoader PyTorch
        """
        # Gunakan nilai default jika tidak disediakan
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.num_workers
        
        try:
            # Dapatkan dataset
            dataset = self.get_dataset(dataset_path, split)
            
            # Buat dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                pin_memory=True,
                drop_last=False
            )
            
            self.logger.info(
                f"✅ DataLoader berhasil dibuat: "
                f"{len(dataloader)} batch, "
                f"batch_size={batch_size}, "
                f"num_workers={num_workers}"
            )
            
            return dataloader
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat dataloader: {str(e)}")
            raise RuntimeError(f"Gagal membuat dataloader: {str(e)}")
    
    def get_eval_loader(
        self,
        dataset_path: Optional[str] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None
    ) -> DataLoader:
        """
        Buat dataloader untuk evaluasi.
        
        Args:
            dataset_path: Path ke dataset (opsional, gunakan test_dir dari config)
            batch_size: Ukuran batch (opsional)
            num_workers: Jumlah worker (opsional)
            
        Returns:
            DataLoader untuk evaluasi
        """
        # Gunakan test_dir dari config jika dataset_path tidak disediakan
        dataset_path = dataset_path or self.test_dir
        
        return self.get_dataloader(
            dataset_path=dataset_path,
            split='test',
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False
        )
    
    def get_dataset_info(self, dataset_path: str) -> Dict[str, Any]:
        """
        Dapatkan informasi tentang dataset.
        
        Args:
            dataset_path: Path ke dataset
            
        Returns:
            Dictionary berisi informasi dataset
        """
        try:
            # Cek keberadaan direktori
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"❌ Dataset tidak ditemukan: {dataset_path}")
            
            # Coba dapatkan info dengan DatasetManager
            if self.dataset_manager:
                # Gunakan split yang sesuai berdasarkan path
                if dataset_path.endswith('/train') or dataset_path == self.train_dir:
                    split = 'train'
                elif dataset_path.endswith('/valid') or dataset_path == self.valid_dir:
                    split = 'valid'
                else:
                    split = 'test'
                
                try:
                    # Gunakan dataset manager untuk mendapatkan statistik
                    stats = self.dataset_manager.get_split_statistics()
                    split_stats = stats.get(split, {})
                    return {
                        'path': dataset_path,
                        'split': split,
                        'num_samples': split_stats.get('num_samples', 0),
                        'num_classes': split_stats.get('num_classes', 0),
                        'class_distribution': split_stats.get('class_distribution', {}),
                        'image_sizes': split_stats.get('image_sizes', [])
                    }
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal mendapatkan statistik dataset: {str(e)}")
            
            # Fallback ke penghitungan manual
            images_dir = os.path.join(dataset_path, 'images')
            labels_dir = os.path.join(dataset_path, 'labels')
            
            num_images = len(os.listdir(images_dir)) if os.path.exists(images_dir) else 0
            num_labels = len(os.listdir(labels_dir)) if os.path.exists(labels_dir) else 0
            
            return {
                'path': dataset_path,
                'num_images': num_images,
                'num_labels': num_labels,
                'has_images_dir': os.path.exists(images_dir),
                'has_labels_dir': os.path.exists(labels_dir)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal mendapatkan info dataset: {str(e)}")
            # Return info minimal
            return {
                'path': dataset_path,
                'error': str(e)
            }
    
    def verify_dataset(self, dataset_path: str) -> bool:
        """
        Verifikasi keberadaan dan struktur dataset.
        
        Args:
            dataset_path: Path ke dataset
            
        Returns:
            True jika dataset valid
        """
        try:
            # Cek keberadaan direktori
            if not os.path.exists(dataset_path):
                self.logger.warning(f"⚠️ Dataset tidak ditemukan: {dataset_path}")
                return False
            
            # Cek struktur
            images_dir = os.path.join(dataset_path, 'images')
            labels_dir = os.path.join(dataset_path, 'labels')
            
            if not os.path.exists(images_dir):
                self.logger.warning(f"⚠️ Direktori images tidak ditemukan: {images_dir}")
                return False
            
            if not os.path.exists(labels_dir):
                self.logger.warning(f"⚠️ Direktori labels tidak ditemukan: {labels_dir}")
                return False
            
            # Cek isi
            num_images = len(os.listdir(images_dir))
            num_labels = len(os.listdir(labels_dir))
            
            if num_images == 0:
                self.logger.warning(f"⚠️ Tidak ada gambar di {images_dir}")
                return False
            
            if num_labels == 0:
                self.logger.warning(f"⚠️ Tidak ada label di {labels_dir}")
                return False
            
            self.logger.info(f"✅ Dataset valid: {num_images} gambar, {num_labels} label")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal memverifikasi dataset: {str(e)}")
            return False