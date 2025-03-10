"""
File: smartcash/handlers/preprocessing/validators.py
Author: Alfrida Sabar
Deskripsi: Komponen untuk validasi dataset menggunakan utils/dataset.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from smartcash.utils.logger import get_logger
from smartcash.utils.dataset import EnhancedDatasetValidator
from smartcash.utils.observer import EventDispatcher, EventTopics
from smartcash.utils.environment_manager import EnvironmentManager


class DatasetValidator:
    """Komponen untuk validasi dataset."""
    
    def __init__(self, config=None, logger=None, env_manager=None):
        self.config = config or {}
        self.logger = logger or get_logger("DatasetValidator")
        self.env_manager = env_manager or EnvironmentManager(logger=self.logger)
        self._validator = None
        
    def validate(self, split='train', fix_issues=False, move_invalid=False, visualize=True, sample_size=0, **kwargs):
        """Validasi dataset pada split tertentu."""
        # Lazy-load validator
        if self._validator is None:
            data_dir = self.config.get('data_dir', 'data')
            resolved_data_dir = self.env_manager.get_path(data_dir) if self.env_manager else Path(data_dir)
            self._validator = EnhancedDatasetValidator(
                config=self.config,
                data_dir=str(resolved_data_dir),
                logger=self.logger
            )
            
        self.logger.start(f"🔍 Memulai validasi dataset untuk split '{split}'")
        EventDispatcher.notify(EventTopics.VALIDATION_EVENT, self, operation="validate", split=split)
        
        # Lakukan validasi
        results = self._validator.validate_dataset(
            split=split,
            fix_issues=fix_issues,
            move_invalid=move_invalid,
            visualize=visualize,
            sample_size=sample_size,
            **kwargs
        )
        
        # Format hasil
        valid_percent = (results['valid_images'] / results['total_images'] * 100) if results['total_images'] > 0 else 0
        self.logger.success(
            f"✅ Validasi split '{split}' selesai: {results['valid_images']}/{results['total_images']} "
            f"gambar valid ({valid_percent:.1f}%)"
        )
        
        EventDispatcher.notify(EventTopics.VALIDATION_EVENT, self, operation="validate_complete", 
                              split=split, result=results)
        
        return {
            'status': 'success' if results['valid_images'] == results['total_images'] else 'warning',
            'validation_stats': results,
            'split': split
        }
    
    def fix_dataset(self, split='train', fix_coordinates=True, fix_labels=True, fix_images=False, backup=True, **kwargs):
        """Perbaiki masalah dataset."""
        if self._validator is None:
            data_dir = self.config.get('data_dir', 'data')
            resolved_data_dir = self.env_manager.get_path(data_dir) if self.env_manager else Path(data_dir)
            self._validator = EnhancedDatasetValidator(
                config=self.config,
                data_dir=str(resolved_data_dir),
                logger=self.logger
            )
            
        self.logger.start(f"🔧 Memulai perbaikan dataset untuk split '{split}'")
        EventDispatcher.notify(EventTopics.VALIDATION_EVENT, self, operation="fix", split=split)
        
        # Lakukan perbaikan
        results = self._validator.fix_dataset(
            split=split,
            fix_coordinates=fix_coordinates,
            fix_labels=fix_labels,
            fix_images=fix_images,
            backup=backup,
            **kwargs
        )
        
        self.logger.success(
            f"✅ Perbaikan split '{split}' selesai: {results['fixed_labels']} label dan "
            f"{results['fixed_coordinates']} koordinat diperbaiki"
        )
        
        EventDispatcher.notify(EventTopics.VALIDATION_EVENT, self, operation="fix_complete", 
                             split=split, result=results)
        
        return {
            'status': 'success',
            'fix_stats': results,
            'split': split
        }