"""
File: smartcash/dataset/services/loader/preprocessed_dataset_loader.py
Deskripsi: Loader untuk dataset yang sudah dipreprocessing sebelumnya
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

from smartcash.common.logger import get_logger
from smartcash.dataset.components.datasets.multilayer_dataset import MultilayerDataset
from smartcash.dataset.preprocessor.core.preprocessing_manager import PreprocessingManager


class PreprocessedDatasetLoader:
    """
    Loader untuk dataset yang sudah dipreprocessing sebelumnya.
    Meningkatkan kecepatan loading untuk training dan evaluasi.
    """
    
    def __init__(
        self,
        preprocessed_dir: Union[str, Path] = "data/preprocessed",
        fallback_to_raw: bool = True,
        auto_preprocess: bool = True,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi loader dataset preprocessed.
        
        Args:
            preprocessed_dir: Direktori dataset preprocessed
            fallback_to_raw: Gunakan dataset asli jika preprocessed tidak tersedia
            auto_preprocess: Otomatis lakukan preprocessing jika diperlukan
            config: Konfigurasi tambahan
            logger: Logger untuk mencatat aktivitas
        """
        self.logger = logger or get_logger()
        self.preprocessed_dir = Path(preprocessed_dir)
        self.fallback_to_raw = fallback_to_raw
        self.auto_preprocess = auto_preprocess
        
        # Default config
        self.default_config = {
            'raw_dataset_dir': 'data/dataset',
            'img_size': (640, 640),
            'batch_size': 16,
            'num_workers': 4,
            'pin_memory': True,
            'multilayer': True
        }
        
        # Merge konfigurasi
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # Inisialisasi preprocessor untuk keperluan fallback
        if self.fallback_to_raw or self.auto_preprocess:
            self.preprocessor = PreprocessingManager(
                config={
                    'img_size': self.config['img_size'],
                    'preprocessed_dir': str(self.preprocessed_dir),
                    'dataset_dir': self.config['raw_dataset_dir']
                },
                logger=self.logger
            )
            
        self.logger.info(f"📦 PreprocessedDatasetLoader diinisialisasi (preprocessed_dir: {self.preprocessed_dir})")
    
    def get_dataset(
        self,
        split: str,
        require_all_layers: bool = False,
        transform: Optional[Callable] = None
    ) -> torch.utils.data.Dataset:
        """
        Dapatkan dataset untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            require_all_layers: Membutuhkan semua layer dalam dataset
            transform: Transformasi tambahan setelah loading
            
        Returns:
            Dataset yang sudah dimuat
            
        Raises:
            ValueError: Jika dataset tidak tersedia dan fallback dinonaktifkan
        """
        # Cek apakah dataset preprocessed tersedia
        split_dir = self.preprocessed_dir / split
        
        # Trigger preprocessing jika perlu
        if not self._is_split_available(split):
            if self.auto_preprocess:
                self.logger.info(f"🔄 Otomatis melakukan preprocessing untuk {split}")
                self.preprocessor.preprocess_dataset(split=split)
            elif self.fallback_to_raw:
                self.logger.warning(
                    f"⚠️ Dataset preprocessed untuk {split} tidak tersedia. "
                    f"Menggunakan dataset asli."
                )
                return self._get_raw_dataset(split, require_all_layers, transform)
            else:
                raise ValueError(
                    f"Dataset preprocessed untuk {split} tidak tersedia dan "
                    f"fallback_to_raw=False"
                )
        
        # Buat dataset dari data preprocessed
        return PreprocessedMultilayerDataset(
            root_dir=split_dir,
            img_size=self.config['img_size'],
            require_all_layers=require_all_layers,
            transform=transform,
            logger=self.logger
        )
    
    def get_dataloader(
        self,
        split: str,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        shuffle: bool = None,
        require_all_layers: bool = False,
        transform: Optional[Callable] = None
    ) -> torch.utils.data.DataLoader:
        """
        Dapatkan dataloader untuk split tertentu.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            batch_size: Ukuran batch
            num_workers: Jumlah worker untuk loading
            shuffle: Acak urutan data
            require_all_layers: Membutuhkan semua layer dalam dataset
            transform: Transformasi tambahan setelah loading
            
        Returns:
            DataLoader yang sudah dikonfigurasi
        """
        # Gunakan default jika tidak disediakan
        batch_size = batch_size or self.config['batch_size']
        num_workers = num_workers or self.config['num_workers']
        
        # Shuffle default ke True untuk training, False untuk lainnya
        if shuffle is None:
            shuffle = (split == 'train')
        
        # Dapatkan dataset
        dataset = self.get_dataset(
            split=split,
            require_all_layers=require_all_layers,
            transform=transform
        )
        
        # Buat dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.config['pin_memory'],
            drop_last=(split == 'train')  # Drop incomplete batch hanya saat training
        )
        
        self.logger.info(
            f"🔄 Dataloader untuk {split} siap: "
            f"{len(dataset)} sampel, {batch_size} batch_size, {num_workers} workers"
        )
        
        return dataloader
    
    def get_all_dataloaders(
        self,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        require_all_layers: bool = False,
        transform: Optional[Callable] = None
    ) -> Dict[str, torch.utils.data.DataLoader]:
        """
        Dapatkan dataloader untuk semua split.
        
        Args:
            batch_size: Ukuran batch
            num_workers: Jumlah worker untuk loading
            require_all_layers: Membutuhkan semua layer dalam dataset
            transform: Transformasi tambahan setelah loading
            
        Returns:
            Dictionary dengan semua dataloader
        """
        dataloaders = {}
        
        for split in ['train', 'val', 'test']:
            try:
                dataloader = self.get_dataloader(
                    split=split,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    require_all_layers=require_all_layers,
                    transform=transform
                )
                dataloaders[split] = dataloader
            except Exception as e:
                self.logger.error(f"❌ Gagal membuat dataloader untuk {split}: {str(e)}")
        
        return dataloaders
    
    def ensure_preprocessed(
        self,
        splits: List[str] = ['train', 'val', 'test'],
        force_reprocess: bool = False
    ) -> Dict[str, Dict[str, int]]:
        """
        Pastikan dataset sudah dipreprocessing.
        
        Args:
            splits: List split yang akan diproses
            force_reprocess: Paksa proses ulang meskipun sudah ada
            
        Returns:
            Statistik hasil preprocessing
        """
        if not hasattr(self, 'preprocessor'):
            self.preprocessor = DatasetPreprocessor(
                config={
                    'img_size': self.config['img_size'],
                    'preprocessed_dir': str(self.preprocessed_dir),
                    'dataset_dir': self.config['raw_dataset_dir']
                },
                logger=self.logger
            )
        
        results = {}
        for split in splits:
            if force_reprocess or not self._is_split_available(split):
                result = self.preprocessor.preprocess_dataset(split=split, force_reprocess=force_reprocess)
                results[split] = result
            else:
                self.logger.info(f"✓ Dataset preprocessed untuk {split} sudah tersedia")
                
        return results
    
    def _is_split_available(self, split: str) -> bool:
        """
        Cek apakah split dataset preprocessed tersedia.
        
        Args:
            split: Split dataset
            
        Returns:
            True jika tersedia, False jika tidak
        """
        split_dir = self.preprocessed_dir / split
        if not split_dir.exists():
            return False
            
        images_dir = split_dir / 'images'
        if not images_dir.exists():
            return False
            
        # Cek apakah ada minimal 1 file
        return len(list(images_dir.glob('*.npy'))) > 0 or len(list(images_dir.glob('*.jpg'))) > 0
    
    def _get_raw_dataset(
        self,
        split: str,
        require_all_layers: bool = False,
        transform: Optional[Callable] = None
    ) -> torch.utils.data.Dataset:
        """
        Gunakan dataset asli (fallback).
        
        Args:
            split: Split dataset
            require_all_layers: Membutuhkan semua layer
            transform: Transformasi tambahan
            
        Returns:
            Dataset asli
        """
        # Perlu import di sini untuk menghindari circular import
        from smartcash.dataset.services.loader.multilayer_loader import MultilayerDataset
        
        raw_dir = Path(self.config['raw_dataset_dir']) / split
        
        return MultilayerDataset(
            root_dir=raw_dir,
            img_size=self.config['img_size'],
            require_all_layers=require_all_layers,
            transform=transform
        )
    

class PreprocessedMultilayerDataset(torch.utils.data.Dataset):
    """
    Dataset untuk data yang sudah dipreprocessing dengan dukungan multilayer.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        img_size: Tuple[int, int] = (640, 640),
        require_all_layers: bool = False,
        transform: Optional[Callable] = None,
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi dataset.
        
        Args:
            root_dir: Direktori root dataset preprocessed
            img_size: Ukuran gambar
            require_all_layers: Membutuhkan semua layer dalam dataset
            transform: Transformasi tambahan setelah loading
            logger: Logger untuk mencatat aktivitas
        """
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.require_all_layers = require_all_layers
        self.transform = transform
        self.logger = logger or get_logger()
        
        # Verifikasi direktori
        self.images_dir = self.root_dir / 'images'
        self.labels_dir = self.root_dir / 'labels'
        
        if not self.images_dir.exists():
            raise ValueError(f"Direktori gambar tidak ditemukan: {self.images_dir}")
            
        if not self.labels_dir.exists():
            raise ValueError(f"Direktori label tidak ditemukan: {self.labels_dir}")
        
        # Daftar semua file gambar
        self.image_files = sorted(list(self.images_dir.glob('*.npy')))
        
        # Tambahkan format gambar lain jika tidak ada .npy
        if not self.image_files:
            self.image_files = sorted(list(self.images_dir.glob('*.jpg')) + 
                                    list(self.images_dir.glob('*.jpeg')) + 
                                    list(self.images_dir.glob('*.png')))
        
        # Filter gambar yang memiliki label
        self.valid_items = []
        for img_path in self.image_files:
            label_path = self.labels_dir / img_path.with_suffix('.txt').name
            if label_path.exists():
                self.valid_items.append((img_path, label_path))
        
        self.logger.info(f"📦 Loaded {len(self.valid_items)} valid items from {root_dir}")
    
    def __len__(self) -> int:
        """Dapatkan jumlah item dalam dataset."""
        return len(self.valid_items)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Dapatkan satu item dari dataset.
        
        Args:
            idx: Indeks item
            
        Returns:
            Dictionary dengan gambar dan label
        """
        img_path, label_path = self.valid_items[idx]
        
        # Load preprocessed image (npy format)
        if img_path.suffix == '.npy':
            # Load gambar yang sudah dipreprocessing
            try:
                img = np.load(str(img_path))
                
                # Convert to tensor
                img_tensor = torch.from_numpy(img).float()
                
                # Ubah format ke channel first jika perlu
                if img_tensor.shape[-1] == 3:  # Jika format HWC
                    img_tensor = img_tensor.permute(2, 0, 1)
            except Exception as e:
                self.logger.error(f"❌ Error saat loading {img_path}: {str(e)}")
                # Fallback ke gambar kosong
                img_tensor = torch.zeros((3, *self.img_size))
        else:
            # Load gambar normal dan lakukan preprocessing on-the-fly
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            
            # Normalisasi
            img = img.astype(np.float32) / 255.0
            
            # Convert to tensor dan ubah ke channel first
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        
        # Load label
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            # Parse label YOLO format
            targets = []
            for line in lines:
                values = line.strip().split()
                if len(values) >= 5:  # class, x, y, w, h
                    class_id = int(values[0])
                    x, y, w, h = map(float, values[1:5])
                    targets.append([class_id, x, y, w, h])
            
            # Convert targets ke tensor
            targets = torch.tensor(targets)
        except Exception as e:
            self.logger.error(f"❌ Error saat loading label {label_path}: {str(e)}")
            # Fallback ke tensor kosong
            targets = torch.zeros((0, 5))
        
        # Terapkan transformasi tambahan jika ada
        if self.transform:
            img_tensor, targets = self.transform(img_tensor, targets)
        
        return {
            'image': img_tensor,
            'targets': targets,
            'img_path': str(img_path),
            'label_path': str(label_path)
        }