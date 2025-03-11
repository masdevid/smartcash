"""
File: smartcash/handlers/preprocessing/manager.py
Deskripsi: Manager preprocessing dengan observer minimal
"""

from typing import Dict, Any, Optional, List
import time

from smartcash.utils.logger import get_logger
from smartcash.utils.observer import (
    EventDispatcher, 
    EventTopics, 
    ObserverManager
)
from smartcash.utils.environment_manager import EnvironmentManager
from smartcash.utils.dataset.dataset_utils import DEFAULT_SPLITS

from smartcash.handlers.preprocessing.validators import DatasetValidator
from smartcash.handlers.preprocessing.augmentors import DatasetAugmentor
from smartcash.handlers.preprocessing.analyzers import DatasetAnalyzer


class PreprocessingManager:
    """Manager utama preprocessing dengan observer minimal."""
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None, 
        logger = None, 
        base_dir: Optional[str] = None
    ):
        """Inisialisasi PreprocessingManager."""
        self.config = config or {}
        self.logger = logger or get_logger("PreprocessingManager")
        
        # Setup komponen
        self.env_manager = EnvironmentManager(base_dir=base_dir, logger=self.logger)
        self.validator = DatasetValidator(
            config=self.config, 
            logger=self.logger, 
            env_manager=self.env_manager
        )
        self.augmentor = DatasetAugmentor(
            config=self.config, 
            logger=self.logger, 
            env_manager=self.env_manager
        )
        self.analyzer = DatasetAnalyzer(
            config=self.config, 
            logger=self.logger, 
            env_manager=self.env_manager
        )
        
        # Setup observer
        self.observer_manager = ObserverManager()
        self._register_default_observers()
    
    def _register_default_observers(self):
        """Registrasi observer default."""
        self.observer_manager.create_logging_observer(
            event_types=[
                EventTopics.PREPROCESSING_START, 
                EventTopics.PREPROCESSING_END,
                EventTopics.PREPROCESSING_PROGRESS
            ],
            log_level="debug"
        )
    
    def run_full_pipeline(
        self, 
        splits: Optional[List[str]] = None, 
        validate_dataset: bool = True, 
        fix_issues: bool = False,
        augment_data: bool = False, 
        analyze_dataset: bool = True
    ) -> Dict[str, Any]:
        """Jalankan pipeline preprocessing lengkap."""
        splits = splits or DEFAULT_SPLITS
        
        EventDispatcher.notify(
            event_type=EventTopics.PREPROCESSING_START, 
            sender=self,
            operation="full_pipeline",
            splits=splits
        )
        
        results = {
            'status': 'success', 
            'validation': {}, 
            'augmentation': {}, 
            'analysis': {}
        }
        
        start_time = time.time()
        
        try:
            # Validasi dataset
            if validate_dataset:
                for split in splits:
                    results['validation'][split] = self.validator.validate(
                        split=split, 
                        fix_issues=fix_issues
                    )
            
            # Analisis dataset
            if analyze_dataset:
                for split in splits:
                    results['analysis'][split] = self.analyzer.analyze(split=split)
            
            # Augmentasi dataset (hanya untuk train)
            if augment_data and 'train' in splits:
                results['augmentation']['train'] = self.augmentor.augment(split='train')
            
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
        
        # Tambahkan waktu eksekusi
        results['elapsed'] = time.time() - start_time
        
        # Kirim event akhir
        EventDispatcher.notify(
            event_type=EventTopics.PREPROCESSING_END, 
            sender=self,
            operation="full_pipeline", 
            results=results
        )
        
        return results
    
    # Metode wrapper untuk operasi individual
    def validate(self, split='train', **kwargs):
        """Validasi dataset pada split tertentu."""
        return self.validator.validate(split=split, **kwargs)
    
    def augment(self, split='train', **kwargs):
        """Augmentasi dataset pada split tertentu."""
        return self.augmentor.augment(split=split, **kwargs)
    
    def analyze(self, split='train', **kwargs):
        """Analisis dataset pada split tertentu."""
        return self.analyzer.analyze(split=split, **kwargs)