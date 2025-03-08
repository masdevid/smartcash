# File: smartcash/handlers/dataset/dataset_loader.py
# Author: Alfrida Sabar
# Deskripsi: Loader untuk dataset dari berbagai sumber

import torch
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from torch.utils.data import DataLoader

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.cache import CacheManager
from smartcash.handlers.dataset.multilayer_dataset import MultilayerDataset
from smartcash.handlers.dataset.dataset_transformer import DatasetTransformer
from smartcash.handlers.dataset.collate_functions import multilayer_collate_fn, flat_collate_fn

class DatasetLoader:
    """
    Loader untuk dataset dari berbagai sumber.
    Menyediakan DataLoader dengan konfigurasi yang sesuai.
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cache_size_gb: float = 1.0,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi DatasetLoader.
        
        Args:
            config: Konfigurasi dataset
            data_dir: Direktori dataset (opsional)
            cache_dir: Direktori cache (opsional)
            cache_size_gb: Ukuran cache dalam GB
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.logger = logger or SmartCashLogger(__name__)
        
        # Setup paths
        self.data_dir = Path(data_dir or config.get('data_dir', 'data'))
        self.img_size = tuple(config.get('model', {}).get('img_size', [640, 640]))
        self.batch_size = config.get('model', {}).get('batch_size', 16)
        self.num_workers = config.get('model', {}).get('workers', 4)
        
        # Aktifkan layer sesuai konfigurasi
        self.active_layers = config.get('layers', ['banknote'])
        
        # Inisialisasi cache
        self.cache = CacheManager(
            cache_dir=cache_dir or config.get('data', {}).get('preprocessing', {}).get('cache_dir', '.cache/smartcash'),
            max_size_gb=cache_size_gb,
            logger=self.logger
        )
        
        # Inisialisasi transformer
        self.transformer = DatasetTransformer(config, self.img_size, logger)
        
        self.logger.info(
            f"🔧 DatasetLoader diinisialisasi:\n"
            f"   • Data dir: {self.data_dir}\n"
            f"   • Img size: {self.img_size}\n"
            f"   • Layers: {self.active_layers}"
        )
    
    def get_dataset(
        self,
        split: str,
        transform=None,
        require_all_layers: bool = False
    ) -> MultilayerDataset:
        """
        Dapatkan dataset untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            transform: Transformasi kustom (opsional)
            require_all_layers: Jika True, hanya gambar dengan semua layer yang akan digunakan
            
        Returns:
            Dataset yang siap digunakan
        """
        # Tentukan path data
        split_path = self._get_split_path(split)
        
        # Tentukan transformasi
        if transform is None:
            transform = self.transformer.get_transform(split)
        
        # Buat dataset
        dataset = MultilayerDataset(
            data_path=split_path,
            img_size=self.img_size,
            mode=split,
            transform=transform,
            layers=self.active_layers,
            require_all_layers=require_all_layers,
            logger=self.logger
        )
        
        self.logger.info(
            f"📊 Dataset '{split}' dibuat dengan {len(dataset)} sampel"
        )
        
        return dataset
    
    def get_dataloader(
        self,
        split: str,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        shuffle: Optional[bool] = None,
        transform=None,
        require_all_layers: bool = False,
        pin_memory: bool = True,
        flat_targets: bool = False
    ) -> DataLoader:
        """
        Dapatkan dataloader untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            batch_size: Ukuran batch
            num_workers: Jumlah worker
            shuffle: Apakah shuffle dataset
            transform: Transformasi kustom
            require_all_layers: Jika True, hanya gambar dengan semua layer yang akan digunakan
            pin_memory: Apakah pin memory untuk GPU
            flat_targets: Jika True, targets akan digabung menjadi satu tensor
            
        Returns:
            DataLoader untuk split yang diminta
        """
        # Gunakan nilai default
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.num_workers
        shuffle = shuffle if shuffle is not None else (split == 'train')
        
        # Dapatkan dataset
        dataset = self.get_dataset(
            split=split,
            transform=transform,
            require_all_layers=require_all_layers
        )
        
        # Pilih collate function berdasarkan format target
        collate_fn = flat_collate_fn if flat_targets else multilayer_collate_fn
        
        # Buat dataloader
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            collate_fn=collate_fn,
            drop_last=(split == 'train')  # Drop batch terakhir hanya untuk training
        )
        
        self.logger.info(
            f"🔄 DataLoader '{split}' dibuat:\n"
            f"   • Batch size: {batch_size}\n"
            f"   • Num workers: {num_workers}\n"
            f"   • Shuffle: {shuffle}\n"
            f"   • Samples: {len(dataset)}\n"
            f"   • Batches: {len(loader)}"
        )
        
        return loader
    
    def get_all_dataloaders(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        require_all_layers: bool = False,
        flat_targets: bool = False
    ) -> Dict[str, DataLoader]:
        """
        Dapatkan semua dataloader (train, val, test) sekaligus.
        
        Args:
            batch_size: Ukuran batch
            num_workers: Jumlah worker
            require_all_layers: Jika True, hanya gambar dengan semua layer yang akan digunakan
            flat_targets: Jika True, targets akan digabung menjadi satu tensor
            
        Returns:
            Dict berisi dataloader untuk setiap split
        """
        start_time = time.time()
        
        # Siapkan dataloader untuk setiap split
        dataloaders = {}
        for split in ['train', 'val', 'test']:
            split_path = self._get_split_path(split)
            
            # Skip jika direktori tidak ada
            if not split_path.exists():
                self.logger.info(f"⚠️ Split '{split}' dilewati karena direktori tidak ada: {split_path}")
                continue
                
            # Buat dataloader
            dataloaders[split] = self.get_dataloader(
                split=split,
                batch_size=batch_size,
                num_workers=num_workers,
                require_all_layers=require_all_layers,
                flat_targets=flat_targets
            )
        
        elapsed_time = time.time() - start_time
        self.logger.success(
            f"✅ Semua dataloader dibuat dalam {elapsed_time:.2f} detik:\n"
            f"   • Train: {len(dataloaders.get('train', [])) if 'train' in dataloaders else 'N/A'} batches\n"
            f"   • Val: {len(dataloaders.get('val', [])) if 'val' in dataloaders else 'N/A'} batches\n"
            f"   • Test: {len(dataloaders.get('test', [])) if 'test' in dataloaders else 'N/A'} batches"
        )
        
        return dataloaders
    
    def get_train_loader(self, **kwargs) -> DataLoader:
        """
        Dapatkan dataloader untuk training.
        
        Args:
            **kwargs: Argumen yang diteruskan ke get_dataloader
            
        Returns:
            DataLoader untuk training
        """
        return self.get_dataloader('train', **kwargs)
    
    def get_val_loader(self, **kwargs) -> DataLoader:
        """
        Dapatkan dataloader untuk validasi.
        
        Args:
            **kwargs: Argumen yang diteruskan ke get_dataloader
            
        Returns:
            DataLoader untuk validasi
        """
        return self.get_dataloader('val', **kwargs)
    
    def get_test_loader(self, **kwargs) -> DataLoader:
        """
        Dapatkan dataloader untuk testing.
        
        Args:
            **kwargs: Argumen yang diteruskan ke get_dataloader
            
        Returns:
            DataLoader untuk testing
        """
        return self.get_dataloader('test', **kwargs)
    
    def _get_split_path(self, split: str) -> Path:
        """
        Dapatkan path untuk split dataset.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            
        Returns:
            Path ke direktori split dataset
        """
        # Normalisasi nama split
        if split in ('val', 'validation'):
            split = 'valid'
            
        # Cek konfigurasi khusus untuk path split
        split_paths = self.config.get('data', {}).get('local', {})
        if split in split_paths:
            return Path(split_paths[split])
            
        # Fallback ke path default
        return self.data_dir / split
    
    def get_dataset_stats(self) -> Dict:
        """
        Dapatkan statistik dataset.
        
        Returns:
            Dict berisi statistik dataset
        """
        stats = {}
        
        for split in ['train', 'valid', 'test']:
            split_path = self._get_split_path(split)
            
            # Skip jika direktori tidak ada
            if not split_path.exists():
                stats[split] = {'error': f"Direktori tidak ditemukan: {split_path}"}
                continue
                
            try:
                # Buat dataset sementara tanpa transformasi
                temp_dataset = MultilayerDataset(
                    data_path=split_path,
                    img_size=self.img_size,
                    mode=split,
                    transform=None,
                    layers=self.active_layers,
                    logger=self.logger
                )
                
                # Hitung distribusi layer
                layer_stats = temp_dataset.get_layer_statistics()
                
                # Hitung distribusi kelas
                class_stats = temp_dataset.get_class_statistics()
                
                stats[split] = {
                    'total_samples': len(temp_dataset),
                    'valid_samples': len(temp_dataset.valid_samples),
                    'layer_distribution': layer_stats,
                    'class_distribution': class_stats
                }
                
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal mendapatkan statistik untuk {split}: {str(e)}")
                stats[split] = {'error': str(e)}
        
        return stats