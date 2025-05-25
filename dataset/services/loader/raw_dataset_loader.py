"""
File: smartcash/dataset/services/loader/raw_dataset_loader.py
Deskripsi: SRP loader untuk raw dataset tanpa preprocessing
"""

from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Union, Any

from smartcash.common.logger import get_logger
from smartcash.dataset.components.datasets.multilayer_dataset import MultilayerDataset
from smartcash.dataset.utils.transform.image_transform import ImageTransformer
from smartcash.dataset.components.collate.multilayer_collate import multilayer_collate_fn
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS, DEFAULT_IMG_SIZE


class RawDatasetLoader:
    """SRP loader untuk raw dataset dengan minimal dependencies."""
    
    def __init__(self, data_dir: Union[str, Path], config: Dict[str, Any], logger=None):
        self.data_dir = Path(data_dir)
        self.config = config
        self.logger = logger or get_logger()
        
        # Extract config values
        self.img_size = tuple(config.get('model', {}).get('input_size', DEFAULT_IMG_SIZE))
        self.batch_size = config.get('training', {}).get('batch_size', 16)
        self.num_workers = config.get('model', {}).get('workers', 4)
        
        # Initialize transformer
        self.transformer = ImageTransformer(config, self.img_size, logger)
        
        self.logger.info(f"🔄 RawDatasetLoader initialized: {self.img_size}")
    
    def get_dataset(self, split: str, transform=None, require_all_layers: bool = False) -> MultilayerDataset:
        """Get raw dataset for split."""
        split = 'valid' if split in ('val', 'validation') else split
        split_path = self._get_split_path(split)
        transform = transform or self.transformer.get_transform(split)
        
        dataset = MultilayerDataset(
            data_path=split_path,
            img_size=self.img_size,
            mode=split,
            transform=transform,
            require_all_layers=require_all_layers,
            logger=self.logger,
            config=self.config
        )
        
        self.logger.info(f"📊 Raw dataset '{split}': {len(dataset)} samples")
        return dataset
    
    def get_dataloader(self, split: str, **kwargs) -> DataLoader:
        """Get dataloader for split."""
        batch_size = kwargs.get('batch_size', self.batch_size)
        num_workers = kwargs.get('num_workers', self.num_workers)
        shuffle = kwargs.get('shuffle', split == 'train')
        
        dataset = self.get_dataset(
            split=split,
            transform=kwargs.get('transform'),
            require_all_layers=kwargs.get('require_all_layers', False)
        )
        
        # Choose collate function
        collate_fn = kwargs.get('collate_fn', multilayer_collate_fn)
        if kwargs.get('flat_targets', False):
            from smartcash.dataset.components.collate.multilayer_collate import flat_collate_fn
            collate_fn = flat_collate_fn
        
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=kwargs.get('pin_memory', True),
            collate_fn=collate_fn,
            drop_last=(split == 'train')
        )
        
        self.logger.info(f"🔄 DataLoader '{split}': {len(loader)} batches")
        return loader
    
    def get_all_dataloaders(self, **kwargs) -> Dict[str, DataLoader]:
        """Get dataloaders for all available splits."""
        dataloaders = {}
        
        for split in DEFAULT_SPLITS:
            split_path = self._get_split_path(split)
            if not split_path.exists():
                self.logger.info(f"⚠️ Skip '{split}': {split_path}")
                continue
            
            dataloaders[split] = self.get_dataloader(split=split, **kwargs)
        
        self.logger.success(f"✅ Created {len(dataloaders)} dataloaders")
        return dataloaders
    
    def _get_split_path(self, split: str) -> Path:
        """Get path for dataset split."""
        split = 'valid' if split in ('val', 'validation') else split
        
        # Check config specific paths
        split_paths = self.config.get('data', {}).get('local', {})
        if split in split_paths:
            return Path(split_paths[split])
        
        return self.data_dir / split