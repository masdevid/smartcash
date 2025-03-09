"""
File: smartcash/handlers/preprocessing/core/augmentation_component.py
Author: Alfrida Sabar
Deskripsi: Komponen augmentasi dataset yang mengintegrasikan AugmentationManager
           melalui AugmentationAdapter untuk augmentasi dataset dalam pipeline preprocessing.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.handlers.preprocessing.core.preprocessing_component import PreprocessingComponent
from smartcash.handlers.preprocessing.integration.augmentation_adapter import AugmentationAdapter


class AugmentationComponent(PreprocessingComponent):
    """
    Komponen preprocessing untuk augmentasi dataset.
    Menggunakan AugmentationAdapter untuk mengintegrasikan AugmentationManager.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        augmentation_adapter: Optional[AugmentationAdapter] = None,
        logger: Optional[SmartCashLogger] = None,
        **kwargs
    ):
        """
        Inisialisasi komponen augmentasi.
        
        Args:
            config: Konfigurasi untuk komponen
            augmentation_adapter: Instance AugmentationAdapter (opsional)
            logger: Logger kustom (opsional)
            **kwargs: Parameter tambahan
        """
        super().__init__(config, logger, **kwargs)
        
        # Buat augmentation adapter jika tidak diberikan
        output_dir = self.get_config_value('data_dir', 'data')
        self.augmentation_adapter = augmentation_adapter or AugmentationAdapter(
            config=config,
            output_dir=output_dir,
            logger=self.logger
        )
    
    def process(
        self, 
        split: str = 'train',
        augmentation_types: Optional[List[str]] = None,
        num_variations: int = 3,
        output_prefix: str = 'aug',
        resume: bool = True,
        validate_results: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Proses augmentasi dataset.
        
        Args:
            split: Split dataset yang akan diaugmentasi (train/valid/test)
            augmentation_types: Jenis augmentasi ('combined', 'lighting', 'position', dll)
            num_variations: Jumlah variasi yang akan dibuat untuk setiap gambar
            output_prefix: Prefix untuk file hasil augmentasi
            resume: Lanjutkan proses augmentasi yang terganggu
            validate_results: Validasi hasil augmentasi
            **kwargs: Parameter tambahan
            
        Returns:
            Dict[str, Any]: Hasil augmentasi
        """
        # Gunakan augmentation adapter untuk augmentasi
        augmentation_result = self.augmentation_adapter.augment(
            split=split,
            augmentation_types=augmentation_types,
            num_variations=num_variations,
            output_prefix=output_prefix,
            resume=resume,
            validate_results=validate_results,
            **kwargs
        )
        
        return augmentation_result
    
    def process_with_combinations(
        self,
        split: str = 'train',
        combinations: Optional[List[Dict[str, Any]]] = None,
        base_output_prefix: str = 'aug',
        resume: bool = True,
        validate_results: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Augmentasi dataset dengan kombinasi parameter kustom.
        
        Args:
            split: Split dataset yang akan diaugmentasi
            combinations: List kombinasi parameter augmentasi
            base_output_prefix: Prefix dasar untuk file hasil augmentasi
            resume: Lanjutkan proses augmentasi yang terganggu
            validate_results: Validasi hasil augmentasi
            **kwargs: Parameter tambahan
            
        Returns:
            Dict[str, Any]: Hasil augmentasi per kombinasi
        """
        # Buat kombinasi default jika tidak diberikan
        if combinations is None:
            # Ambil parameter augmentasi dari config
            degrees = self.get_config_value('training.degrees', 30)
            translate = self.get_config_value('training.translate', 0.1)
            scale = self.get_config_value('training.scale', 0.5)
            fliplr = self.get_config_value('training.fliplr', 0.5)
            hsv_h = self.get_config_value('training.hsv_h', 0.015)
            hsv_s = self.get_config_value('training.hsv_s', 0.7)
            hsv_v = self.get_config_value('training.hsv_v', 0.4)
            
            # Buat kombinasi parameter
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
                },
                {
                    'augmentation_types': ['position'],
                    'num_variations': 2,
                    'output_prefix': f"{base_output_prefix}_position",
                    'degrees': degrees * 1.5,  # Lebih ekstrem
                    'translate': translate * 1.5,
                    'scale': scale * 0.8,
                    'fliplr': fliplr * 1.2
                }
            ]
        
        # Gunakan augmentation adapter untuk augmentasi dengan kombinasi
        augmentation_result = self.augmentation_adapter.augment_with_combinations(
            split=split,
            combinations=combinations,
            base_output_prefix=base_output_prefix,
            resume=resume,
            validate_results=validate_results,
            **kwargs
        )
        
        return augmentation_result