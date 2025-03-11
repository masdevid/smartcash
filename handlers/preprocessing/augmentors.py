"""
File: smartcash/handlers/preprocessing/augmentors.py
Author: Alfrida Sabar
Deskripsi: Komponen untuk augmentasi dataset menggunakan utils/augmentation.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path

from smartcash.utils.logger import get_logger
from smartcash.utils.augmentation import AugmentationManager
from smartcash.utils.observer import EventTopics
from smartcash.utils.observer.observer_manager import ObserverManager
from smartcash.utils.environment_manager import EnvironmentManager


class DatasetAugmentor:
    """Komponen untuk augmentasi dataset."""
    
    def __init__(self, config=None, logger=None, env_manager=None):
        self.config = config or {}
        self.logger = logger or get_logger("DatasetAugmentor")
        self.env_manager = env_manager or EnvironmentManager(logger=self.logger)
        self._augmentor = None
        
        # Setup observer manager
        self.observer_manager = ObserverManager(auto_register=True)
        self.observer_manager.create_logging_observer(
            event_types=[EventTopics.PREPROCESSING_START, EventTopics.PREPROCESSING_END, EventTopics.PREPROCESSING_ERROR],
            log_level="debug"
        )
    
    def _lazy_init_augmentor(self):
        """Lazy initialize augmentor."""
        if self._augmentor is None:
            output_dir = self.config.get('data_dir', 'data')
            resolved_output_dir = self.env_manager.get_path(output_dir) if self.env_manager else Path(output_dir)
            num_workers = self.config.get('data', {}).get('preprocessing', {}).get('num_workers', 4)
            self._augmentor = AugmentationManager(
                config=self.config,
                output_dir=str(resolved_output_dir),
                logger=self.logger,
                num_workers=num_workers
            )
    
    def augment(self, split='train', augmentation_types=None, **kwargs):
        """Augmentasi dataset pada split tertentu."""
        self._lazy_init_augmentor()
        
        if augmentation_types is None:
            augmentation_types = ['combined', 'lighting']
        
        self.logger.start(f"🎨 Memulai augmentasi: {split}, tipe: {augmentation_types}")
        
        try:
            self.observer_manager.create_simple_observer(
                event_type=EventTopics.PREPROCESSING_END,
                callback=lambda *args, **kw: self.logger.success(
                    f"✅ Augmentasi {split} selesai: {len(kw.get('result', {}).get('augmented_images', []))} gambar dibuat"
                ),
                name=f"AugmentationEnd_{split}"
            )
            
            stats = self._augmentor.augment_dataset(
                split=split,
                augmentation_types=augmentation_types,
                **kwargs
            )
            
            return {
                'status': 'success', 
                'augmentation_stats': stats, 
                'split': split
            }
        
        except Exception as e:
            self.logger.error(f"❌ Augmentasi gagal: {str(e)}")
            return {
                'status': 'error', 
                'error': str(e), 
                'split': split
            }