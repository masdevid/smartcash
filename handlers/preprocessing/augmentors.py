"""
File: smartcash/handlers/preprocessing/augmentors.py
Author: Alfrida Sabar
Deskripsi: Komponen untuk augmentasi dataset menggunakan utils/augmentation.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path

from smartcash.utils.logger import get_logger
from smartcash.utils.augmentation import AugmentationManager
from smartcash.utils.observer import EventDispatcher, EventTopics
from smartcash.utils.environment_manager import EnvironmentManager


class DatasetAugmentor:
    """Komponen untuk augmentasi dataset."""
    
    def __init__(self, config=None, logger=None, env_manager=None):
        self.config = config or {}
        self.logger = logger or get_logger("DatasetAugmentor")
        self.env_manager = env_manager or EnvironmentManager(logger=self.logger)
        self._augmentor = None
        
    def augment(self, split='train', augmentation_types=None, num_variations=3, 
               output_prefix='aug', resume=True, validate_results=True, **kwargs):
        """Augmentasi dataset pada split tertentu."""
        # Lazy-load augmentor
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
            
        if augmentation_types is None:
            augmentation_types = ['combined', 'lighting']
            
        self.logger.start(
            f"🎨 Memulai augmentasi untuk split '{split}' dengan "
            f"jenis: {', '.join(augmentation_types)}"
        )
        EventDispatcher.notify(EventTopics.AUGMENTATION_EVENT, self, operation="augment", split=split)
        
        # Lakukan augmentasi
        stats = self._augmentor.augment_dataset(
            split=split,
            augmentation_types=augmentation_types,
            num_variations=num_variations,
            output_prefix=output_prefix,
            resume=resume,
            validate_results=validate_results,
            **kwargs
        )
        
        self.logger.success(
            f"✅ Augmentasi selesai: {stats['augmented']} gambar dihasilkan "
            f"dalam {stats['duration']:.2f} detik"
        )
        
        EventDispatcher.notify(EventTopics.AUGMENTATION_EVENT, self, operation="augment_complete", 
                              split=split, result=stats)
        
        return {
            'status': 'success',
            'augmentation_stats': stats,
            'split': split
        }
    
    def augment_with_combinations(self, split='train', combinations=None, 
                                 base_output_prefix='aug', resume=True, validate_results=True, **kwargs):
        """Augmentasi dataset dengan kombinasi parameter kustom."""
        # Buat kombinasi default jika tidak diberikan
        if combinations is None:
            # Ambil parameter dari config
            degrees = self.config.get('training', {}).get('degrees', 30)
            translate = self.config.get('training', {}).get('translate', 0.1)
            scale = self.config.get('training', {}).get('scale', 0.5)
            fliplr = self.config.get('training', {}).get('fliplr', 0.5)
            hsv_h = self.config.get('training', {}).get('hsv_h', 0.015)
            hsv_s = self.config.get('training', {}).get('hsv_s', 0.7)
            hsv_v = self.config.get('training', {}).get('hsv_v', 0.4)
            
            combinations = [
                {
                    'augmentation_types': ['combined'],
                    'num_variations': 2,
                    'output_prefix': f"{base_output_prefix}_combined",
                    'degrees': degrees,
                    'translate': translate,
                    'scale': scale,
                    'fliplr': fliplr
                },
                {
                    'augmentation_types': ['lighting'],
                    'num_variations': 2,
                    'output_prefix': f"{base_output_prefix}_lighting",
                    'hsv_h': hsv_h,
                    'hsv_s': hsv_s,
                    'hsv_v': hsv_v
                }
            ]
        
        self.logger.start(f"🎨 Memulai augmentasi dengan {len(combinations)} kombinasi parameter")
        total_augmented = 0
        results = []
        
        # Eksekusi setiap kombinasi
        for i, combo in enumerate(combinations):
            EventDispatcher.notify(EventTopics.PREPROCESSING_PROGRESS, self, 
                                  operation="augmentation_combinations", current=i, total=len(combinations))
            combo_params = {**kwargs, **combo}
            self.logger.info(f"🔄 Kombinasi {i+1}/{len(combinations)}: {combo['augmentation_types']}")
            result = self.augment(split=split, resume=resume, validate_results=validate_results, **combo_params)
            results.append(result)
            total_augmented += result['augmentation_stats']['augmented']
        
        self.logger.success(
            f"✅ Augmentasi kombinasi selesai: {total_augmented} total gambar dari {len(combinations)} kombinasi"
        )
        
        return {
            'status': 'success',
            'combination_results': results,
            'total_augmented': total_augmented,
            'combinations_count': len(combinations),
            'split': split
        }