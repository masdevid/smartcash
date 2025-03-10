# File: smartcash/handlers/dataset/dataset_utils_adapter.py
# Deskripsi: Adapter untuk integrasi utils/dataset ke handlers/dataset

from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from smartcash.utils.observer import EventTopics, notify
from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.dataset import (
    EnhancedDatasetValidator, 
    DatasetAnalyzer,
    DatasetFixer
)
from smartcash.utils.dataset.dataset_utils import DatasetUtils
from smartcash.utils.augmentation import AugmentationManager

class DatasetUtilsAdapter:
    """Adapter untuk mengintegrasikan komponen utils/dataset dengan handlers/dataset."""
    
    def __init__(
        self, 
        config: Dict, 
        data_dir: Optional[str] = None, 
        logger: Optional[SmartCashLogger] = None
    ):
        """Inisialisasi adapter."""
        self.config = config
        self.data_dir = data_dir or config.get('data_dir', 'data')
        self.logger = logger or SmartCashLogger(__name__)
        self._components = {}  # Cache komponen untuk lazy loading
        
        self.logger.info(f"🔄 DatasetUtilsAdapter diinisialisasi untuk: {self.data_dir}")
    
    def _get_component(self, name: str, factory_func):
        """Helper untuk lazy loading komponen."""
        if name not in self._components:
            self._components[name] = factory_func()
        return self._components[name]
        
    @property
    def validator(self) -> EnhancedDatasetValidator:
        """Lazy load validator."""
        return self._get_component('validator', lambda: EnhancedDatasetValidator(
            config=self.config,
            data_dir=self.data_dir,
            logger=self.logger
        ))
    
    @property
    def analyzer(self) -> DatasetAnalyzer:
        """Lazy load analyzer."""
        return self._get_component('analyzer', lambda: DatasetAnalyzer(
            config=self.config,
            data_dir=self.data_dir,
            logger=self.logger
        ))
    
    @property
    def fixer(self) -> DatasetFixer:
        """Lazy load fixer."""
        return self._get_component('fixer', lambda: DatasetFixer(
            config=self.config,
            data_dir=self.data_dir,
            logger=self.logger
        ))
    
    @property
    def augmentor(self) -> AugmentationManager:
        """Lazy load augmentor."""
        return self._get_component('augmentor', lambda: AugmentationManager(
            config=self.config,
            output_dir=self.data_dir,
            logger=self.logger,
            num_workers=self.config.get('model', {}).get('workers', 4)
        ))
    
    @property
    def utils(self) -> DatasetUtils:
        """Lazy load utils."""
        return self._get_component('utils', lambda: DatasetUtils(
            config=self.config,
            data_dir=self.data_dir,
            logger=self.logger
        ))
    
    def validate_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """Validasi dataset menggunakan EnhancedDatasetValidator."""
        notify(EventTopics.VALIDATION_EVENT, self, action="start", split=split, **kwargs)
        
        try:
            result = self.validator.validate_dataset(split=split, **kwargs)
            notify(EventTopics.VALIDATION_EVENT, self, action="complete", split=split, result=result)
            return result
        except Exception as e:
            notify(EventTopics.VALIDATION_EVENT, self, action="error", split=split, error=str(e))
            self.logger.error(f"❌ Validasi dataset gagal: {str(e)}")
            raise
    
    def analyze_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """Analisis dataset menggunakan DatasetAnalyzer."""
        notify(EventTopics.PREPROCESSING, self, action="analyze", split=split, **kwargs)
        
        try:
            result = self.analyzer.analyze_dataset(split=split, **kwargs)
            notify(EventTopics.PREPROCESSING, self, action="analyze_complete", split=split, result=result)
            return result
        except Exception as e:
            notify(EventTopics.PREPROCESSING, self, action="analyze_error", split=split, error=str(e))
            self.logger.error(f"❌ Analisis dataset gagal: {str(e)}")
            raise
    
    def fix_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """Perbaiki masalah dataset menggunakan DatasetFixer."""
        notify(EventTopics.PREPROCESSING, self, action="fix", split=split, **kwargs)
        
        try:
            result = self.fixer.fix_dataset(split=split, **kwargs)
            notify(EventTopics.PREPROCESSING, self, action="fix_complete", split=split, result=result)
            return result
        except Exception as e:
            notify(EventTopics.PREPROCESSING, self, action="fix_error", split=split, error=str(e))
            self.logger.error(f"❌ Perbaikan dataset gagal: {str(e)}")
            raise
    
    def augment_dataset(self, **kwargs) -> Dict[str, Any]:
        """Augmentasi dataset menggunakan AugmentationManager."""
        split = kwargs.get('split', 'train')
        aug_types = kwargs.get('augmentation_types', ['combined'])
        
        notify(EventTopics.AUGMENTATION_EVENT, self, 
               action="start", split=split, types=aug_types)
        
        try:
            result = self.augmentor.augment_dataset(**kwargs)
            notify(EventTopics.AUGMENTATION_EVENT, self, action="complete", split=split, result=result)
            return result
        except Exception as e:
            notify(EventTopics.AUGMENTATION_EVENT, self, action="error", split=split, error=str(e))
            self.logger.error(f"❌ Augmentasi dataset gagal: {str(e)}")
            raise
    
    def split_dataset(self, **kwargs) -> Dict:
        """Split dataset menggunakan DatasetUtils."""
        notify(EventTopics.PREPROCESSING, self, action="split", **kwargs)
        
        try:
            result = self.utils.split_dataset(**kwargs)
            notify(EventTopics.PREPROCESSING, self, action="split_complete", result=result)
            return result
        except Exception as e:
            notify(EventTopics.PREPROCESSING, self, action="split_error", error=str(e))
            self.logger.error(f"❌ Split dataset gagal: {str(e)}")
            raise