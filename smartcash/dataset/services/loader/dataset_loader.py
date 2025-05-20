"""
File: smartcash/dataset/services/loader/dataset_loader.py
Deskripsi: Layanan untuk loading dataset dan pembuatan dataloader
"""

import torch
from pathlib import Path
from typing import Dict, Optional, Any, List, Union

from torch.utils.data import DataLoader
from smartcash.common.logger import get_logger
from smartcash.dataset.components.datasets.multilayer_dataset import MultilayerDataset
from smartcash.dataset.utils.transform.image_transform import ImageTransformer
from smartcash.dataset.components.collate.multilayer_collate import multilayer_collate_fn
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS, DEFAULT_IMG_SIZE

class DatasetLoaderService:
    """Service untuk loading dataset dan pembuatan dataloader."""
    
    def __init__(self, config: Dict, data_dir: str, logger=None):
        """
        Inisialisasi DatasetLoaderService.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori data
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = logger or get_logger()
        
        # Setup parameter default
        self.img_size = tuple(config.get('model', {}).get('input_size', DEFAULT_IMG_SIZE))
        self.batch_size = config.get('training', {}).get('batch_size', 16)
        self.num_workers = config.get('model', {}).get('workers', 4)
        
        # Inisialisasi transformer
        self.transformer = ImageTransformer(self.config, self.img_size, self.logger)
        
        self.logger.info(f"🔄 DatasetLoaderService diinisialisasi dengan ukuran gambar: {self.img_size}")
    
    def get_dataset(self, split: str, transform=None, require_all_layers: bool = False) -> MultilayerDataset:
        """
        Dapatkan dataset untuk split tertentu.
        
        Args:
            split: Split dataset yang diinginkan ('train', 'valid', 'test')
            transform: Transformasi kustom (opsional)
            require_all_layers: Apakah memerlukan semua layer dalam setiap gambar
            
        Returns:
            Instance dari MultilayerDataset
        """
        # Normalisasi nama split
        if split in ('val', 'validation'):
            split = 'valid'
            
        # Tentukan path split
        split_path = self._get_split_path(split)
        
        # Dapatkan transformasi yang sesuai
        transform = transform or self.transformer.get_transform(split)
        
        # Buat dan return dataset
        dataset = MultilayerDataset(
            data_path=split_path,
            img_size=self.img_size,
            mode=split,
            transform=transform,
            require_all_layers=require_all_layers,
            logger=self.logger,
            config=self.config
        )
        
        self.logger.info(f"📊 Dataset '{split}' dibuat dengan {len(dataset)} sampel")
        return dataset
    
    def get_dataloader(self, split: str, batch_size: Optional[int] = None, 
                     num_workers: Optional[int] = None, shuffle: Optional[bool] = None,
                     transform=None, require_all_layers: bool = False,
                     pin_memory: bool = True, flat_targets: bool = False) -> DataLoader:
        """
        Dapatkan dataloader untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            batch_size: Ukuran batch (opsional)
            num_workers: Jumlah worker (opsional)
            shuffle: Flag untuk mengacak data (opsional)
            transform: Transformasi kustom (opsional)
            require_all_layers: Apakah memerlukan semua layer dalam setiap gambar
            pin_memory: Flag untuk pin memory (untuk GPU)
            flat_targets: Apakah menggunakan format target yang flat
            
        Returns:
            Instance dari DataLoader
        """
        # Gunakan nilai default jika parameter tidak disediakan
        batch_size = batch_size or self.batch_size
        num_workers = num_workers or self.num_workers
        shuffle = shuffle if shuffle is not None else (split == 'train')
        
        # Dapatkan dataset
        dataset = self.get_dataset(
            split=split,
            transform=transform,
            require_all_layers=require_all_layers
        )
        
        # Pilih collate function yang sesuai
        if flat_targets:
            from smartcash.dataset.components.collate.multilayer_collate import flat_collate_fn
            collate_fn = flat_collate_fn
        else:
            collate_fn = multilayer_collate_fn
        
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
    
    def get_all_dataloaders(self, batch_size: Optional[int] = None, 
                           num_workers: Optional[int] = None, **kwargs) -> Dict[str, DataLoader]:
        """
        Dapatkan semua dataloader untuk semua split.
        
        Args:
            batch_size: Ukuran batch (opsional)
            num_workers: Jumlah worker (opsional)
            **kwargs: Parameter tambahan untuk dataloader
            
        Returns:
            Dictionary berisi dataloader untuk setiap split
        """
        import time
        start_time = time.time()
        dataloaders = {}
        
        for split in DEFAULT_SPLITS:
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
                **kwargs
            )
        
        elapsed_time = time.time() - start_time
        self.logger.success(
            f"✅ Semua dataloader dibuat dalam {elapsed_time:.2f} detik:\n"
            f"   • Train: {len(dataloaders.get('train', [])) if 'train' in dataloaders else 'N/A'} batches\n"
            f"   • Valid: {len(dataloaders.get('valid', [])) if 'valid' in dataloaders else 'N/A'} batches\n"
            f"   • Test: {len(dataloaders.get('test', [])) if 'test' in dataloaders else 'N/A'} batches"
        )
        return dataloaders
    
    def _get_split_path(self, split: str) -> Path:
        """
        Dapatkan path untuk split dataset tertentu.
        
        Args:
            split: Split dataset
            
        Returns:
            Path ke direktori split
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