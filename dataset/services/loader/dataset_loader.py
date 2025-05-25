"""
File: smartcash/dataset/services/loader/dataset_loader.py
Deskripsi: Refactored main dataset loader service dengan reduced duplication dan SRP
"""

import time
from pathlib import Path
from typing import Dict, Optional, Any, Union

from smartcash.common.logger import get_logger
from smartcash.dataset.services.loader.loader_factory import DatasetLoaderFactory
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS


class DatasetLoaderService:
    """Main service untuk dataset loading dengan factory pattern dan reduced duplication."""
    
    def __init__(self, config: Dict, data_dir: str, logger=None):
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = logger or get_logger()
        
        # Initialize appropriate loader based on availability
        self._primary_loader = self._initialize_primary_loader()
        self._cache_manager = None
        
        self.logger.info(f"🔄 DatasetLoaderService initialized: {data_dir}")
    
    def get_dataset(self, split: str, **kwargs):
        """Get dataset using primary loader."""
        return self._primary_loader.get_dataset(split=split, **kwargs)
    
    def get_dataloader(self, split: str, **kwargs):
        """Get dataloader using primary loader."""
        return self._primary_loader.get_dataloader(split=split, **kwargs)
    
    def get_all_dataloaders(self, **kwargs) -> Dict[str, Any]:
        """Get all dataloaders with timing info."""
        start_time = time.time()
        dataloaders = self._primary_loader.get_all_dataloaders(**kwargs)
        elapsed = time.time() - start_time
        
        self.logger.success(f"✅ All dataloaders ready in {elapsed:.2f}s")
        return dataloaders
    
    def get_batch_generator(self, split: str, **kwargs):
        """Get optimized batch generator for split."""
        dataset = self.get_dataset(split=split, **kwargs)
        return DatasetLoaderFactory.create_batch_generator(
            dataset=dataset,
            batch_size=kwargs.get('batch_size', 16),
            shuffle=kwargs.get('shuffle', split == 'train'),
            num_workers=kwargs.get('num_workers', 4)
        )
    
    def enable_caching(self, max_ram_gb: float = 2.0, max_disk_gb: float = 10.0):
        """Enable dataset caching for performance."""
        if not self._cache_manager:
            self._cache_manager = DatasetLoaderFactory.create_cache_manager(
                max_ram_gb=max_ram_gb,
                max_disk_gb=max_disk_gb,
                logger=self.logger
            )
            self.logger.info(f"💾 Caching enabled: {max_ram_gb}GB RAM, {max_disk_gb}GB disk")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics if caching enabled."""
        if self._cache_manager:
            return self._cache_manager.get_stats()
        return {'caching_enabled': False}
    
    def clear_cache(self):
        """Clear dataset cache."""
        if self._cache_manager:
            self._cache_manager.clear()
            self.logger.info("🧹 Dataset cache cleared")
    
    def _initialize_primary_loader(self):
        """Initialize primary loader based on data availability."""
        # Try preprocessed first
        preprocessed_dir = self.data_dir / "preprocessed"
        if self._has_preprocessed_data(preprocessed_dir):
            self.logger.info("📦 Using preprocessed dataset loader")
            return DatasetLoaderFactory.create_preprocessed_loader(
                preprocessed_dir=preprocessed_dir,
                config=self.config,
                logger=self.logger
            )
        
        # Fallback to raw
        self.logger.info("📊 Using raw dataset loader")
        return DatasetLoaderFactory.create_raw_loader(
            data_dir=self.data_dir,
            config=self.config,
            logger=self.logger
        )
    
    def _has_preprocessed_data(self, preprocessed_dir: Path) -> bool:
        """Check if preprocessed data exists."""
        if not preprocessed_dir.exists():
            return False
        
        # Check if any split has images
        for split in DEFAULT_SPLITS:
            split_dir = preprocessed_dir / split / 'images'
            if split_dir.exists() and any(split_dir.glob('*')):
                return True
        
        return False