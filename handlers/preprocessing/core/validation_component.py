"""
File: smartcash/handlers/preprocessing/core/validation_component.py
Author: Alfrida Sabar
Deskripsi: Komponen validasi dataset yang mengintegrasikan EnhancedDatasetValidator 
           melalui ValidatorAdapter untuk validasi dataset dalam pipeline preprocessing.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.handlers.preprocessing.core.preprocessing_component import PreprocessingComponent
from smartcash.handlers.preprocessing.integration.validator_adapter import ValidatorAdapter


class ValidationComponent(PreprocessingComponent):
    """
    Komponen preprocessing untuk validasi dataset.
    Menggunakan ValidatorAdapter untuk mengintegrasikan EnhancedDatasetValidator.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        validator_adapter: Optional[ValidatorAdapter] = None,
        logger: Optional[SmartCashLogger] = None,
        **kwargs
    ):
        """
        Inisialisasi komponen validasi.
        
        Args:
            config: Konfigurasi untuk komponen
            validator_adapter: Instance ValidatorAdapter (opsional)
            logger: Logger kustom (opsional)
            **kwargs: Parameter tambahan
        """
        super().__init__(config, logger, **kwargs)
        
        # Buat validator adapter jika tidak diberikan
        self.validator_adapter = validator_adapter or ValidatorAdapter(
            config=config,
            logger=self.logger
        )
    
    def process(
        self, 
        split: str = 'train',
        fix_issues: bool = False,
        move_invalid: bool = False,
        visualize: bool = True,
        sample_size: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Proses validasi dataset.
        
        Args:
            split: Split dataset yang akan divalidasi (train/valid/test)
            fix_issues: Otomatis memperbaiki masalah yang ditemukan
            move_invalid: Pindahkan file yang tidak valid ke direktori terpisah
            visualize: Buat visualisasi masalah
            sample_size: Jumlah sampel yang akan divalidasi (0 = semua)
            **kwargs: Parameter tambahan
            
        Returns:
            Dict[str, Any]: Hasil validasi
        """
        # Gunakan validator adapter untuk validasi
        validation_result = self.validator_adapter.validate(
            split=split,
            fix_issues=fix_issues,
            move_invalid=move_invalid,
            visualize=visualize,
            sample_size=sample_size,
            **kwargs
        )
        
        return validation_result
    
    def analyze_dataset(
        self,
        split: str = 'train',
        sample_size: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analisis dataset secara mendalam.
        
        Args:
            split: Split dataset yang akan dianalisis (train/valid/test)
            sample_size: Jumlah sampel yang akan dianalisis (0 = semua)
            **kwargs: Parameter tambahan
            
        Returns:
            Dict[str, Any]: Hasil analisis
        """
        # Gunakan validator adapter untuk analisis
        analysis_result = self.validator_adapter.analyze(
            split=split,
            sample_size=sample_size,
            **kwargs
        )
        
        return analysis_result
    
    def fix_dataset(
        self,
        split: str = 'train',
        fix_coordinates: bool = True,
        fix_labels: bool = True,
        fix_images: bool = False,
        backup: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perbaiki masalah dataset.
        
        Args:
            split: Split dataset yang akan diperbaiki (train/valid/test)
            fix_coordinates: Perbaiki koordinat tidak valid
            fix_labels: Perbaiki format label
            fix_images: Perbaiki gambar corrupted
            backup: Buat backup sebelum memperbaiki
            **kwargs: Parameter tambahan
            
        Returns:
            Dict[str, Any]: Hasil perbaikan
        """
        # Gunakan validator adapter untuk perbaikan
        fix_result = self.validator_adapter.fix_dataset(
            split=split,
            fix_coordinates=fix_coordinates,
            fix_labels=fix_labels,
            fix_images=fix_images,
            backup=backup,
            **kwargs
        )
        
        return fix_result