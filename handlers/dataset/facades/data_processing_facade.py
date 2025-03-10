# File: smartcash/handlers/dataset/facades/data_processing_facade.py
# Deskripsi: Facade untuk operasi pemrosesan dataset (validasi, augmentasi, balancing)

from typing import Dict, Optional, Any

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
    
    # ===== Metode dari DatasetBalancer =====
    
    def analyze_class_distribution(self, split: str = 'train', per_layer: bool = True) -> Dict[str, Any]:
        """Analisis distribusi kelas dalam dataset."""
        return self.balancer.analyze_class_distribution(split, per_layer)
    
    def balance_by_undersampling(self, split: str = 'train', **kwargs) -> Dict[str, Any]:
        """Seimbangkan dataset dengan undersampling."""
        return self.balancer.balance_by_undersampling(split, **kwargs)