"""
File: smartcash/dataset/services/loader/preprocessed_dataset_loader.py
Deskripsi: Updated loader untuk dataset yang sudah dipreprocessing dengan integrasi service layer baru
"""

import os
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor.utils.preprocessing_factory import PreprocessingFactory


class PreprocessedDatasetLoader:
    """
    Updated loader untuk dataset yang sudah dipreprocessing dengan service layer integration.
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
        Inisialisasi loader dataset preprocessed dengan service layer baru.
        
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
        
        # Default config untuk compatibility dengan service layer baru
        self.default_config = {
            'data': {'dir': 'data'},
            'preprocessing': {
                'img_size': [640, 640],
                'normalize': True,
                'output_dir': str(self.preprocessed_dir),
                'num_workers': 4
            },
            'batch_size': 16,
            'pin_memory': True,
            'multilayer': True
        }
        
        # Merge konfigurasi
        self.config = self.default_config.copy()
        if config:
            # Update preprocessing config properly
            if 'preprocessing' in config:
                self.config['preprocessing'].update(config['preprocessing'])
            if 'data' in config:
                self.config['data'].update(config['data'])
            # Direct config updates
            for key in ['batch_size', 'pin_memory', 'multilayer']:
                if key in config:
                    self.config[key] = config[key]
        
        # Initialize service layer components untuk auto preprocessing
        self._preprocessing_manager = None
        
        self.logger.info(f"📦 PreprocessedDatasetLoader initialized dengan service layer (dir: {self.preprocessed_dir})")
    
    def get_dataset(
        self,
        split: str,
        require_all_layers: bool = False,
        transform: Optional[Callable] = None
    ) -> torch.utils.data.Dataset:
        """
        Dapatkan dataset untuk split tertentu dengan fallback mechanism.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            require_all_layers: Membutuhkan semua layer dalam dataset
            transform: Transformasi tambahan setelah loading
            
        Returns:
            Dataset yang sudah dimuat
        """
        # Normalize split name (val -> valid)
        normalized_split = 'valid' if split == 'val' else split
        
        # Check apakah dataset preprocessed tersedia
        if not self._is_split_available(normalized_split):
            if self.auto_preprocess:
                self.logger.info(f"🔄 Auto preprocessing untuk {normalized_split}")
                self._ensure_preprocessing(normalized_split)
            elif self.fallback_to_raw:
                self.logger.warning(
                    f"⚠️ Dataset preprocessed untuk {normalized_split} tidak tersedia. "
                    f"Menggunakan dataset asli."
                )
                return self._get_raw_dataset(normalized_split, require_all_layers, transform)
            else:
                raise ValueError(
                    f"Dataset preprocessed untuk {normalized_split} tidak tersedia dan "
                    f"fallback_to_raw=False"
                )
        
        # Load dataset dari preprocessed data
        return PreprocessedDataset(
            root_dir=self.preprocessed_dir / normalized_split,
            img_size=tuple(self.config['preprocessing']['img_size']),
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
            split: Split dataset ('train', 'valid', 'test')
            batch_size: Ukuran batch
            num_workers: Jumlah worker untuk loading
            shuffle: Acak urutan data
            require_all_layers: Membutuhkan semua layer dalam dataset
            transform: Transformasi tambahan setelah loading
            
        Returns:
            DataLoader yang sudah dikonfigurasi
        """
        # Use defaults dari config
        batch_size = batch_size or self.config['batch_size']
        num_workers = num_workers or self.config['preprocessing']['num_workers']
        
        # Shuffle default: True untuk training, False untuk lainnya
        if shuffle is None:
            shuffle = (split in ['train', 'training'])
        
        # Get dataset
        dataset = self.get_dataset(
            split=split,
            require_all_layers=require_all_layers,
            transform=transform
        )
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.config['pin_memory'],
            drop_last=(split in ['train', 'training'])
        )
        
        self.logger.info(
            f"🔄 Dataloader {split}: {len(dataset)} samples, batch={batch_size}, workers={num_workers}"
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
        
        Returns:
            Dictionary dengan semua dataloader
        """
        dataloaders = {}
        
        for split in ['train', 'valid', 'test']:
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
                self.logger.warning(f"⚠️ Skip dataloader {split}: {str(e)}")
        
        return dataloaders
    
    def ensure_preprocessed(
        self,
        splits: List[str] = ['train', 'valid', 'test'],
        force_reprocess: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Pastikan dataset sudah dipreprocessing menggunakan service layer baru.
        
        Args:
            splits: List split yang akan diproses
            force_reprocess: Paksa proses ulang meskipun sudah ada
            
        Returns:
            Hasil preprocessing per split
        """
        results = {}
        
        for split in splits:
            try:
                if force_reprocess or not self._is_split_available(split):
                    result = self._ensure_preprocessing(split, force_reprocess)
                    results[split] = result
                else:
                    self.logger.info(f"✅ Dataset preprocessed {split} sudah tersedia")
                    results[split] = {'success': True, 'message': 'Already available'}
            except Exception as e:
                self.logger.error(f"❌ Error preprocessing {split}: {str(e)}")
                results[split] = {'success': False, 'error': str(e)}
        
        return results
    
    def _ensure_preprocessing(self, split: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """Ensure preprocessing menggunakan service layer baru."""
        try:
            # Create preprocessing manager jika belum ada
            if not self._preprocessing_manager:
                self._preprocessing_manager = PreprocessingFactory.create_preprocessing_manager(
                    self.config, self.logger
                )
            
            # Coordinate preprocessing untuk split
            result = self._preprocessing_manager.coordinate_preprocessing(
                split=split,
                force_reprocess=force_reprocess
            )
            
            if result['success']:
                self.logger.success(f"✅ Preprocessing {split} berhasil: {result.get('total_images', 0)} gambar")
            else:
                self.logger.error(f"❌ Preprocessing {split} gagal: {result.get('message')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error dalam preprocessing {split}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _is_split_available(self, split: str) -> bool:
        """
        Check apakah split dataset preprocessed tersedia.
        
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
        
        # Check minimal ada 1 file gambar
        image_extensions = ['.jpg', '.jpeg', '.png', '.npy']
        for ext in image_extensions:
            if list(images_dir.glob(f'*{ext}')):
                return True
        
        return False
    
    def _get_raw_dataset(
        self,
        split: str,
        require_all_layers: bool = False,
        transform: Optional[Callable] = None
    ) -> torch.utils.data.Dataset:
        """
        Fallback ke dataset asli (raw).
        
        Args:
            split: Split dataset
            require_all_layers: Membutuhkan semua layer
            transform: Transformasi tambahan
            
        Returns:
            Dataset asli
        """
        # Import here untuk avoid circular import
        try:
            from smartcash.dataset.services.loader.multilayer_loader import MultilayerDataset
            
            raw_dir = Path(self.config['data']['dir']) / split
            
            return MultilayerDataset(
                root_dir=raw_dir,
                img_size=tuple(self.config['preprocessing']['img_size']),
                require_all_layers=require_all_layers,
                transform=transform
            )
        except ImportError:
            # Fallback ke basic dataset jika MultilayerDataset tidak tersedia
            return PreprocessedDataset(
                root_dir=Path(self.config['data']['dir']) / split,
                img_size=tuple(self.config['preprocessing']['img_size']),
                require_all_layers=require_all_layers,
                transform=transform,
                logger=self.logger
            )


class PreprocessedDataset(torch.utils.data.Dataset):
    """
    Updated dataset untuk data yang sudah dipreprocessing dengan dukungan multilayer.
    Compatible dengan output preprocessor service layer baru.
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
        
        # Verify directories
        self.images_dir = self.root_dir / 'images'
        self.labels_dir = self.root_dir / 'labels'
        
        if not self.images_dir.exists():
            raise ValueError(f"Images directory tidak ditemukan: {self.images_dir}")
        
        if not self.labels_dir.exists():
            self.logger.warning(f"⚠️ Labels directory tidak ditemukan: {self.labels_dir}")
            self.labels_dir = None
        
        # Get all image files (support multiple formats)
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.npy']:
            self.image_files.extend(sorted(self.images_dir.glob(f'*{ext}')))
        
        # Filter gambar yang memiliki label (jika ada labels_dir)
        self.valid_items = []
        for img_path in self.image_files:
            if self.labels_dir:
                label_path = self.labels_dir / img_path.with_suffix('.txt').name
                if label_path.exists():
                    self.valid_items.append((img_path, label_path))
                elif not self.require_all_layers:
                    # Allow image without label untuk inference
                    self.valid_items.append((img_path, None))
            else:
                # No labels directory, use image only
                self.valid_items.append((img_path, None))
        
        if not self.valid_items:
            raise ValueError(f"Tidak ada valid items ditemukan di {root_dir}")
        
        self.logger.info(f"📦 Loaded {len(self.valid_items)} items dari {root_dir}")
    
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
        
        # Load image
        img_tensor = self._load_image(img_path)
        
        # Load label jika ada
        targets = self._load_label(label_path) if label_path else torch.zeros((0, 5))
        
        # Apply transform jika ada
        if self.transform:
            img_tensor, targets = self.transform(img_tensor, targets)
        
        return {
            'image': img_tensor,
            'targets': targets,
            'img_path': str(img_path),
            'label_path': str(label_path) if label_path else None
        }
    
    def _load_image(self, img_path: Path) -> torch.Tensor:
        """Load gambar dengan support multiple formats."""
        try:
            if img_path.suffix == '.npy':
                # Load preprocessed numpy array
                img = np.load(str(img_path))
                img_tensor = torch.from_numpy(img).float()
                
                # Ensure channel first format
                if len(img_tensor.shape) == 3 and img_tensor.shape[-1] == 3:
                    img_tensor = img_tensor.permute(2, 0, 1)
            else:
                # Load regular image dan preprocess on-the-fly
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError(f"Cannot load image: {img_path}")
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                
                # Normalize
                img = img.astype(np.float32) / 255.0
                
                # Convert to tensor (HWC -> CHW)
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            
            return img_tensor
            
        except Exception as e:
            self.logger.error(f"❌ Error loading {img_path}: {str(e)}")
            # Return zero tensor sebagai fallback
            return torch.zeros((3, *self.img_size))
    
    def _load_label(self, label_path: Path) -> torch.Tensor:
        """Load label dalam format YOLO."""
        try:
            if not label_path.exists():
                return torch.zeros((0, 5))
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            targets = []
            for line in lines:
                values = line.strip().split()
                if len(values) >= 5:  # class, x, y, w, h
                    class_id = int(values[0])
                    x, y, w, h = map(float, values[1:5])
                    targets.append([class_id, x, y, w, h])
            
            return torch.tensor(targets, dtype=torch.float32)
            
        except Exception as e:
            self.logger.error(f"❌ Error loading label {label_path}: {str(e)}")
            return torch.zeros((0, 5))