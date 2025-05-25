"""
File: smartcash/dataset/services/loader/loader_factory.py
Deskripsi: Factory untuk creating dataset loaders dengan SRP compliance
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.dataset.services.loader.preprocessed_dataset_loader import PreprocessedDatasetLoader
from smartcash.dataset.services.loader.raw_dataset_loader import RawDatasetLoader
from smartcash.dataset.services.loader.batch_generator import BatchGenerator
from smartcash.dataset.services.loader.cache_manager import DatasetCacheManager


class DatasetLoaderFactory:
    """Factory untuk creating different types of dataset loaders."""
    
    @staticmethod
    def create_preprocessed_loader(
        preprocessed_dir: Union[str, Path] = "data/preprocessed",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None
    ) -> PreprocessedDatasetLoader:
        """Create preprocessed dataset loader."""
        return PreprocessedDatasetLoader(
            preprocessed_dir=preprocessed_dir,
            fallback_to_raw=True,
            auto_preprocess=True,
            config=config,
            logger=logger or get_logger()
        )
    
    @staticmethod
    def create_raw_loader(
        data_dir: Union[str, Path],
        config: Dict[str, Any],
        logger: Optional[Any] = None
    ) -> RawDatasetLoader:
        """Create raw dataset loader."""
        return RawDatasetLoader(
            data_dir=data_dir,
            config=config,
            logger=logger or get_logger()
        )
    
    @staticmethod
    def create_batch_generator(
        dataset,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 2,
        **kwargs
    ) -> BatchGenerator:
        """Create optimized batch generator."""
        return BatchGenerator(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )
    
    @staticmethod
    def create_cache_manager(
        cache_dir: Optional[str] = None,
        max_ram_gb: float = 2.0,
        max_disk_gb: float = 10.0,
        logger: Optional[Any] = None
    ) -> DatasetCacheManager:
        """Create dataset cache manager."""
        return DatasetCacheManager(
            cache_dir=cache_dir,
            max_ram_usage_gb=max_ram_gb,
            max_disk_usage_gb=max_disk_gb,
            logger=logger or get_logger()
        )
    
    @staticmethod
    def create_unified_loader(
        data_dir: Union[str, Path],
        config: Dict[str, Any],
        prefer_preprocessed: bool = True,
        logger: Optional[Any] = None
    ):
        """Create unified loader yang automatically choose preprocessed or raw."""
        if prefer_preprocessed:
            try:
                return DatasetLoaderFactory.create_preprocessed_loader(
                    preprocessed_dir=Path(data_dir) / "preprocessed",
                    config=config,
                    logger=logger
                )
            except Exception:
                logger = logger or get_logger()
                logger.warning("⚠️ Preprocessed loader failed, fallback to raw")
                return DatasetLoaderFactory.create_raw_loader(
                    data_dir=data_dir,
                    config=config,
                    logger=logger
                )
        else:
            return DatasetLoaderFactory.create_raw_loader(
                data_dir=data_dir,
                config=config,
                logger=logger
            )