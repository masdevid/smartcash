"""
File: smartcash/handlers/preprocessing/pipeline/validation_pipeline.py
Author: Alfrida Sabar
Deskripsi: Pipeline khusus untuk validasi dataset yang menggunakan ValidationComponent
           untuk validasi dan analisis dataset.
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.handlers.preprocessing.core.validation_component import ValidationComponent
from smartcash.handlers.preprocessing.pipeline.preprocessing_pipeline import PreprocessingPipeline
from smartcash.handlers.preprocessing.observers.base_observer import BaseObserver
from smartcash.handlers.preprocessing.observers.progress_observer import ProgressObserver
from smartcash.handlers.preprocessing.integration.validator_adapter import ValidatorAdapter


class ValidationPipeline(PreprocessingPipeline):
    """
    Pipeline khusus untuk validasi dataset yang menggunakan ValidationComponent
    untuk validasi dan analisis dataset.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        logger: Optional[SmartCashLogger] = None,
        validator_adapter: Optional[ValidatorAdapter] = None,
        add_progress_observer: bool = True
    ):
        """
        Inisialisasi pipeline validasi.
        
        Args:
            config: Konfigurasi pipeline
            logger: Logger kustom (opsional)
            validator_adapter: Instance ValidatorAdapter (opsional)
            add_progress_observer: Tambahkan progress observer secara otomatis
        """
        super().__init__(name="ValidationPipeline", logger=logger, config=config)
        
        # Buat validator adapter jika tidak diberikan
        self.validator_adapter = validator_adapter or ValidatorAdapter(
            config=config,
            logger=self.logger
        )
        
        # Buat validation component
        self.validation_component = ValidationComponent(
            config=config,
            validator_adapter=self.validator_adapter,
            logger=self.logger
        )
        
        # Tambahkan ke pipeline
        self.add_component(self.validation_component)
        
        # Tambahkan progress observer jika diminta
        if add_progress_observer:
            self.add_observer(ProgressObserver(logger=self.logger))
    
    def validate(
        self, 
        split: str = 'train',
        fix_issues: bool = False,
        move_invalid: bool = False,
        visualize: bool = True,
        sample_size: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan pipeline validasi dataset.
        
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
        # Parameters untuk validasi
        params = {
            'split': split,
            'fix_issues': fix_issues,
            'move_invalid': move_invalid,
            'visualize': visualize,
            'sample_size': sample_size,
            **kwargs
        }
        
        # Jalankan pipeline
        results = self.run(**params)
        
        # Ekstrak hasil validasi
        validation_results = {}
        if results['status'] == 'success':
            for component in results['components']:
                if component['name'] == 'ValidationComponent' and component['status'] == 'success':
                    validation_results = component.get('result', {})
        
        return validation_results
    
    def analyze(
        self,
        split: str = 'train',
        sample_size: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan pipeline analisis dataset.
        
        Args:
            split: Split dataset yang akan dianalisis (train/valid/test)
            sample_size: Jumlah sampel yang akan dianalisis (0 = semua)
            **kwargs: Parameter tambahan
            
        Returns:
            Dict[str, Any]: Hasil analisis
        """
        # Jalankan analisis dengan metode langsung dari validation_component
        return self.validation_component.analyze_dataset(
            split=split,
            sample_size=sample_size,
            **kwargs
        )
    
    def fix(
        self,
        split: str = 'train',
        fix_coordinates: bool = True,
        fix_labels: bool = True,
        fix_images: bool = False,
        backup: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan pipeline perbaikan dataset.
        
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
        # Jalankan perbaikan dengan metode langsung dari validation_component
        return self.validation_component.fix_dataset(
            split=split,
            fix_coordinates=fix_coordinates,
            fix_labels=fix_labels,
            fix_images=fix_images,
            backup=backup,
            **kwargs
        )
    
    def validate_all_splits(
        self,
        splits: Optional[List[str]] = None,
        fix_issues: bool = False,
        move_invalid: bool = False,
        visualize: bool = True,
        sample_size: int = 0,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validasi semua split dataset.
        
        Args:
            splits: List split yang akan divalidasi (default: ['train', 'valid', 'test'])
            fix_issues: Otomatis memperbaiki masalah yang ditemukan
            move_invalid: Pindahkan file yang tidak valid ke direktori terpisah
            visualize: Buat visualisasi masalah
            sample_size: Jumlah sampel yang akan divalidasi (0 = semua)
            **kwargs: Parameter tambahan
            
        Returns:
            Dict[str, Dict[str, Any]]: Hasil validasi per split
        """
        if splits is None:
            splits = ['train', 'valid', 'test']
            
        self.logger.start(f"🔍 Memulai validasi untuk {len(splits)} splits: {', '.join(splits)}")
        
        results = {}
        for split in splits:
            self.logger.info(f"🔄 Validasi split: {split}")
            split_result = self.validate(
                split=split,
                fix_issues=fix_issues,
                move_invalid=move_invalid,
                visualize=visualize,
                sample_size=sample_size,
                **kwargs
            )
            results[split] = split_result
        
        # Hitung total statistik
        total_images = sum(r.get('validation_stats', {}).get('total_images', 0) for r in results.values())
        valid_images = sum(r.get('validation_stats', {}).get('valid_images', 0) for r in results.values())
        valid_percent = (valid_images / total_images * 100) if total_images > 0 else 0
        
        self.logger.success(
            f"✅ Validasi semua splits selesai: {valid_images}/{total_images} "
            f"gambar valid ({valid_percent:.1f}%)"
        )
        
        return results