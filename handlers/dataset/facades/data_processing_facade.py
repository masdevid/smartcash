# File: smartcash/handlers/dataset/facades/data_processing_facade.py
# Author: Alfrida Sabar
# Deskripsi: Facade untuk operasi pemrosesan data dengan adapter pattern

from typing import Dict, Optional, Any

from smartcash.utils.observer import EventTopics, notify
from smartcash.handlers.dataset.facades.dataset_base_facade import DatasetBaseFacade
from smartcash.handlers.dataset.dataset_utils_adapter import DatasetUtilsAdapter
from smartcash.handlers.dataset.core.dataset_balancer import DatasetBalancer


class DataProcessingFacade(DatasetBaseFacade):
    """Facade untuk operasi pemrosesan dataset (validasi, augmentasi, balancing)."""
    
    @property
    def utils_adapter(self) -> DatasetUtilsAdapter:
        """Akses ke adapter utils/dataset."""
        return self._get_component('utils_adapter', lambda: DatasetUtilsAdapter(
            config=self.config, 
            data_dir=str(self.data_dir),
            logger=self.logger
        ))
    
    @property
    def balancer(self) -> DatasetBalancer:
        """Akses ke balancer dataset."""
        return self._get_component('balancer', lambda: DatasetBalancer(
            data_dir=str(self.data_dir), 
            logger=self.logger
        ))
    
    # ===== Metode dari utils_adapter =====
    
    def validate_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """Validasi dataset."""
        return self.utils_adapter.validate_dataset(split, **kwargs)
    
    def analyze_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """Analisis dataset."""
        return self.utils_adapter.analyze_dataset(split, **kwargs)
    
    def fix_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """Perbaiki dataset."""
        return self.utils_adapter.fix_dataset(split, **kwargs)
    
    def augment_dataset(self, **kwargs) -> Dict[str, Any]:
        """Augmentasi dataset."""
        return self.utils_adapter.augment_dataset(**kwargs)
    
    def augment_with_combinations(self, **kwargs) -> Dict[str, Any]:
        """Augmentasi dengan kombinasi parameter kustom."""
        notify(EventTopics.AUGMENTATION_EVENT, self, action="start_custom", parameters=kwargs)
        
        try:
            result = self.utils_adapter.augmentor.augment_with_combinations(**kwargs)
            notify(EventTopics.AUGMENTATION_EVENT, self, action="complete_custom", result=result)
            return result
        except Exception as e:
            notify(EventTopics.AUGMENTATION_EVENT, self, action="error_custom", error=str(e))
            self.logger.error(f"❌ Augmentasi kombinasi gagal: {str(e)}")
            raise
    
    # ===== Metode dari DatasetBalancer =====
    
    def analyze_class_distribution(self, split: str = 'train', per_layer: bool = True) -> Dict[str, Any]:
        """Analisis distribusi kelas dalam dataset."""
        notify(EventTopics.PREPROCESSING, self, action="analyze_distribution", split=split)
        result = self.balancer.analyze_class_distribution(split, per_layer)
        notify(EventTopics.PREPROCESSING, self, action="analyze_distribution_complete", result=result)
        return result
    
    def balance_by_undersampling(self, split: str = 'train', **kwargs) -> Dict[str, Any]:
        """Seimbangkan dataset dengan undersampling."""
        notify(EventTopics.PREPROCESSING, self, action="balance", split=split)
        result = self.balancer.balance_by_undersampling(split, **kwargs)
        notify(EventTopics.PREPROCESSING, self, action="balance_complete", result=result)
        return result