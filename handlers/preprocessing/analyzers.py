"""
File: smartcash/handlers/preprocessing/analyzers.py
Author: Alfrida Sabar
Deskripsi: Komponen untuk analisis dataset menggunakan utils/dataset.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from smartcash.utils.logger import get_logger
from smartcash.utils.dataset import EnhancedDatasetValidator
from smartcash.utils.observer import EventDispatcher, EventTopics
from smartcash.utils.environment_manager import EnvironmentManager


class DatasetAnalyzer:
    """Komponen untuk analisis dataset."""
    
    def __init__(self, config=None, logger=None, env_manager=None):
        self.config = config or {}
        self.logger = logger or get_logger("DatasetAnalyzer")
        self.env_manager = env_manager or EnvironmentManager(logger=self.logger)
        self._validator = None
    
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
    
    def analyze(self, split='train', **kwargs):
        """Analisis dataset pada split tertentu."""
        self._lazy_init_validator()
        
        self.logger.start(f"📊 Memulai analisis dataset: {split}")
        
        try:
            EventDispatcher.notify(EventTopics.PREPROCESSING_START, self, split=split)
            
            analysis = self._validator.analyze_dataset(split=split, **kwargs)
            
            EventDispatcher.notify(EventTopics.PREPROCESSING_END, self, split=split, result=analysis)
            
            self.logger.success(f"✅ Analisis split '{split}' selesai")
            
            return {
                'status': 'success', 
                'analysis': analysis, 
                'split': split
            }
        
        except Exception as e:
            self.logger.error(f"❌ Analisis gagal: {str(e)}")
            EventDispatcher.notify(EventTopics.PREPROCESSING_ERROR, self, error=str(e))
            
            return {
                'status': 'error', 
                'error': str(e), 
                'split': split
            }