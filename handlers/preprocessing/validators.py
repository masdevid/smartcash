"""
File: smartcash/handlers/preprocessing/validators.py
Author: Alfrida Sabar
Deskripsi: Komponen untuk validasi dataset menggunakan utils/dataset.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from smartcash.utils.logger import get_logger
from smartcash.utils.dataset import EnhancedDatasetValidator
from smartcash.utils.observer import EventTopics
from smartcash.utils.observer.observer_manager import ObserverManager
from smartcash.utils.environment_manager import EnvironmentManager


class DatasetValidator:
    """Komponen untuk validasi dataset."""
    
    def __init__(self, config=None, logger=None, env_manager=None):
        self.config = config or {}
        self.logger = logger or get_logger("DatasetValidator")
        self.env_manager = env_manager or EnvironmentManager(logger=self.logger)
        self._validator = None
        
        # Setup observer manager
        self.observer_manager = ObserverManager(auto_register=True)
        self.observer_manager.create_logging_observer(
            event_types=[EventTopics.PREPROCESSING_START, EventTopics.PREPROCESSING_END, EventTopics.PREPROCESSING_ERROR],
            log_level="debug"
        )
    
    def _lazy_init_validator(self):
        """Lazy initialize validator."""
        if self._validator is None:
            data_dir = self.config.get('data_dir', 'data')
            resolved_data_dir = self.env_manager.get_path(data_dir) if self.env_manager else Path(data_dir)
            self._validator = EnhancedDatasetValidator(
                config=self.config,
                data_dir=str(resolved_data_dir),
                logger=self.logger
            )
    
    def validate(self, split='train', **kwargs):
        """Validasi dataset pada split tertentu."""
        self._lazy_init_validator()
        
        self.logger.start(f"🔍 Memulai validasi: {split}")
        
        try:
            self.observer_manager.create_simple_observer(
                event_type=EventTopics.PREPROCESSING_END,
                callback=lambda *args, **kw: self.logger.success(
                    f"✅ Validasi {split} selesai: "
                    f"{kw.get('result', {}).get('valid_images', 0)}/{kw.get('result', {}).get('total_images', 0)} "
                    f"gambar valid"
                ),
                name=f"ValidationEnd_{split}"
            )
            
            results = self._validator.validate_dataset(split=split, **kwargs)
            
            return {
                'status': 'success', 
                'validation_stats': results, 
                'split': split
            }
        
        except Exception as e:
            self.logger.error(f"❌ Validasi gagal: {str(e)}")
            return {
                'status': 'error', 
                'error': str(e), 
                'split': split
            }