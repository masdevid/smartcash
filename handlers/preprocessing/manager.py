"""
File: smartcash/handlers/preprocessing/manager.py
Author: Alfrida Sabar
Deskripsi: Manager utama preprocessing yang mengintegrasikan semua komponen preprocessing.
"""

import time
from typing import Dict, Any, Optional, List

from smartcash.utils.logger import get_logger
from smartcash.utils.observer import EventTopics
from smartcash.utils.observer.observer_manager import ObserverManager
from smartcash.utils.environment_manager import EnvironmentManager

from smartcash.handlers.preprocessing.validators import DatasetValidator
from smartcash.handlers.preprocessing.augmentors import DatasetAugmentor
from smartcash.handlers.preprocessing.analyzers import DatasetAnalyzer


class PreprocessingManager:
    """Manager utama untuk preprocessing dataset."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger=None, base_dir=None):
        self.config = config or {}
        self.logger = logger or get_logger("PreprocessingManager")
        
        # Setup environment
        self.env_manager = EnvironmentManager(base_dir=base_dir, logger=self.logger)
        if self.env_manager.is_colab:
            self.env_manager.mount_drive()
            if self.env_manager.is_drive_mounted:
                self.env_manager.create_symlinks()
        
        # Inisialisasi komponen
        self.validator = DatasetValidator(config=self.config, logger=self.logger, env_manager=self.env_manager)
        self.augmentor = DatasetAugmentor(config=self.config, logger=self.logger, env_manager=self.env_manager)
        self.analyzer = DatasetAnalyzer(config=self.config, logger=self.logger, env_manager=self.env_manager)
        
        # Setup observer
        self.observer_manager = ObserverManager(auto_register=True)
        self.observer_manager.create_logging_observer(
            event_types=[EventTopics.PREPROCESSING_START, EventTopics.PREPROCESSING_END, EventTopics.PREPROCESSING_ERROR],
            log_level="debug"
        )
    
    def run_full_pipeline(self, splits=None, validate_dataset=True, fix_issues=False, 
                         augment_data=False, analyze_dataset=True):
        """Jalankan pipeline preprocessing lengkap."""
        if splits is None:
            splits = ['train', 'valid', 'test']
            
        start_time = time.time()
        results = {'status': 'success', 'validation': {}, 'augmentation': {}, 'analysis': {}}
        
        try:
            # Validasi dataset
            if validate_dataset:
                for split in splits:
                    results['validation'][split] = self.validator.validate(split=split, fix_issues=fix_issues)
                    
            # Analisis dataset
            if analyze_dataset:
                for split in splits:
                    results['analysis'][split] = self.analyzer.analyze(split=split)
                    
            # Augmentasi dataset (hanya train)
            if augment_data and 'train' in splits:
                results['augmentation']['train'] = self.augmentor.augment(split='train')
                
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
            self.logger.error(f"❌ Pipeline preprocessing gagal: {str(e)}")
            
        results['elapsed'] = time.time() - start_time
        
        # Tambahkan observer untuk log akhir pipeline
        self.observer_manager.create_simple_observer(
            event_type=EventTopics.PREPROCESSING_END,
            callback=lambda *args, **kw: self.logger.success(f"✅ Preprocessing selesai dalam {results['elapsed']:.2f} detik"),
            name="PreprocessingEnd"
        )
        
        return results