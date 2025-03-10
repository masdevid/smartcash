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
        
    def analyze(self, split='train', sample_size=0, analyze_class_distribution=True, 
               analyze_image_sizes=True, analyze_bounding_boxes=True, **kwargs):
        """Analisis dataset pada split tertentu."""
        # Lazy-load validator
        if self._validator is None:
            data_dir = self.config.get('data_dir', 'data')
            resolved_data_dir = self.env_manager.get_path(data_dir) if self.env_manager else Path(data_dir)
            self._validator = EnhancedDatasetValidator(
                config=self.config,
                data_dir=str(resolved_data_dir),
                logger=self.logger
            )
            
        self.logger.start(f"📊 Memulai analisis dataset untuk split '{split}'")
        EventDispatcher.notify(EventTopics.VALIDATION_EVENT, self, operation="analyze", split=split)
        
        # Lakukan analisis melalui validator yang ada
        analysis = self._validator.analyze_dataset(
            split=split,
            sample_size=sample_size,
            analyze_class_distribution=analyze_class_distribution,
            analyze_image_sizes=analyze_image_sizes,
            analyze_bounding_boxes=analyze_bounding_boxes,
            **kwargs
        )
        
        # Log insights penting
        if 'class_balance' in analysis:
            imbalance = analysis['class_balance'].get('imbalance_score', 0)
            self.logger.info(f"📊 Ketidakseimbangan kelas: {imbalance:.2f}/10")
            
        if 'image_size_distribution' in analysis:
            dominant_size = analysis['image_size_distribution'].get('dominant_size', 'unknown')
            self.logger.info(f"📊 Ukuran gambar dominan: {dominant_size}")
            
        if 'bbox_stats' in analysis:
            avg_objects = analysis['bbox_stats'].get('average_objects_per_image', 0)
            self.logger.info(f"📊 Rata-rata {avg_objects:.1f} objek per gambar")
        
        self.logger.success(f"✅ Analisis split '{split}' selesai")
        
        EventDispatcher.notify(EventTopics.VALIDATION_EVENT, self, operation="analyze_complete", 
                             split=split, result=analysis)
        
        return {
            'status': 'success',
            'analysis': analysis,
            'split': split
        }
    
    def analyze_class_balance(self, split='train', **kwargs):
        """Analisis keseimbangan kelas pada split tertentu."""
        return self.analyze(
            split=split, 
            analyze_class_distribution=True,
            analyze_image_sizes=False,
            analyze_bounding_boxes=False,
            **kwargs
        )
    
    def analyze_image_sizes(self, split='train', **kwargs):
        """Analisis ukuran gambar pada split tertentu."""
        return self.analyze(
            split=split, 
            analyze_class_distribution=False,
            analyze_image_sizes=True,
            analyze_bounding_boxes=False,
            **kwargs
        )
    
    def analyze_bounding_boxes(self, split='train', **kwargs):
        """Analisis bounding box pada split tertentu."""
        return self.analyze(
            split=split, 
            analyze_class_distribution=False,
            analyze_image_sizes=False,
            analyze_bounding_boxes=True,
            **kwargs
        )