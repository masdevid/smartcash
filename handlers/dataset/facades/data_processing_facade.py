# File: smartcash/handlers/dataset/facades/data_processing_facade.py
# Author: Alfrida Sabar
# Deskripsi: Facade khusus untuk operasi pemrosesan data seperti validasi, augmentasi, dan balancing

from typing import Dict, Optional, Any

from smartcash.handlers.dataset.facades.dataset_base_facade import DatasetBaseFacade
from smartcash.handlers.dataset.core.dataset_validator import DatasetValidator
from smartcash.handlers.dataset.core.dataset_augmentor import DatasetAugmentor
from smartcash.handlers.dataset.core.dataset_balancer import DatasetBalancer


class DataProcessingFacade(DatasetBaseFacade):
    """
    Facade yang menyediakan akses ke operasi pemrosesan dataset seperti validasi,
    augmentasi, dan penyeimbangan kelas.
    """
    
    @property
    def validator(self) -> DatasetValidator:
        """Akses ke komponen validator dengan lazy initialization."""
        return self._get_component('validator', lambda: DatasetValidator(
            config=self.config,
            data_dir=str(self.data_dir),
            logger=self.logger,
            num_workers=self.num_workers
        ))
    
    @property
    def augmentor(self) -> DatasetAugmentor:
        """Akses ke komponen augmentor dengan lazy initialization."""
        return self._get_component('augmentor', lambda: DatasetAugmentor(
            config=self.config,
            data_dir=str(self.data_dir),
            logger=self.logger,
            num_workers=self.num_workers
        ))
    
    @property
    def balancer(self) -> DatasetBalancer:
        """Akses ke komponen balancer dengan lazy initialization."""
        return self._get_component('balancer', lambda: DatasetBalancer(
            data_dir=str(self.data_dir),
            logger=self.logger
        ))
    
    # ===== Metode dari DatasetValidator =====
    
    def validate_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """
        Validasi dataset menggunakan validator terintegrasi.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            **kwargs: Parameter tambahan
            
        Returns:
            Dict berisi statistik validasi
        """
        return self.validator.validate_dataset(split, **kwargs)
    
    def analyze_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """
        Analisis dataset menggunakan validator terintegrasi.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            **kwargs: Parameter tambahan
            
        Returns:
            Dict berisi analisis dataset
        """
        return self.validator.analyze_dataset(split, **kwargs)
    
    def fix_dataset(self, split: str, **kwargs) -> Dict[str, Any]:
        """
        Perbaiki masalah dataset menggunakan validator terintegrasi.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            **kwargs: Parameter tambahan
            
        Returns:
            Dict berisi statistik perbaikan
        """
        return self.validator.fix_dataset(split, **kwargs)
    
    # ===== Metode dari DatasetAugmentor =====
    
    def augment_dataset(self, **kwargs) -> Dict[str, Any]:
        """
        Augmentasi dataset menggunakan AugmentationManager dari utils/augmentation.
        
        Args:
            **kwargs: Parameter untuk AugmentationManager.augment_dataset
            
        Returns:
            Dict berisi statistik augmentasi
        """
        return self.augmentor.augment_dataset(**kwargs)
    
    def augment_with_combinations(self, **kwargs) -> Dict[str, Any]:
        """
        Augmentasi dataset dengan kombinasi parameter kustom.
        
        Args:
            **kwargs: Parameter untuk AugmentationManager.custom_augment
            
        Returns:
            Dict berisi statistik augmentasi
        """
        return self.augmentor.augment_with_combinations(**kwargs)
    
    # ===== Metode dari DatasetBalancer =====
    
    def analyze_class_distribution(self, split: str = 'train', per_layer: bool = True) -> Dict[str, Any]:
        """
        Analisis distribusi kelas dalam split dataset.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            per_layer: Jika True, analisis per layer
            
        Returns:
            Dict berisi statistik distribusi kelas
        """
        return self.balancer.analyze_class_distribution(split, per_layer)
    
    def balance_by_undersampling(self, split: str = 'train', **kwargs) -> Dict[str, Any]:
        """
        Seimbangkan dataset dengan mengurangi jumlah sampel kelas dominan.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            **kwargs: Parameter tambahan
            
        Returns:
            Dict berisi statistik penyeimbangan
        """
        return self.balancer.balance_by_undersampling(split, **kwargs)