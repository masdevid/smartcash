"""
File: smartcash/handlers/preprocessing/pipeline/augmentation_pipeline.py
Author: Alfrida Sabar
Deskripsi: Pipeline khusus untuk augmentasi dataset yang menggunakan AugmentationComponent
           untuk memperkaya dataset dengan berbagai teknik augmentasi.
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.handlers.preprocessing.core.augmentation_component import AugmentationComponent
from smartcash.handlers.preprocessing.pipeline.preprocessing_pipeline import PreprocessingPipeline
from smartcash.handlers.preprocessing.observers.base_observer import BaseObserver
from smartcash.handlers.preprocessing.observers.progress_observer import ProgressObserver
from smartcash.handlers.preprocessing.integration.augmentation_adapter import AugmentationAdapter


class AugmentationPipeline(PreprocessingPipeline):
    """
    Pipeline khusus untuk augmentasi dataset yang menggunakan AugmentationComponent
    untuk memperkaya dataset dengan berbagai teknik augmentasi.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        logger: Optional[SmartCashLogger] = None,
        augmentation_adapter: Optional[AugmentationAdapter] = None,
        add_progress_observer: bool = True
    ):
        """
        Inisialisasi pipeline augmentasi.
        
        Args:
            config: Konfigurasi pipeline
            logger: Logger kustom (opsional)
            augmentation_adapter: Instance AugmentationAdapter (opsional)
            add_progress_observer: Tambahkan progress observer secara otomatis
        """
        super().__init__(name="AugmentationPipeline", logger=logger, config=config)
        
        # Ambil output_dir dari config
        output_dir = config.get('data_dir', config.get('data', {}).get('data_dir', 'data'))
        
        # Buat augmentation adapter jika tidak diberikan
        self.augmentation_adapter = augmentation_adapter or AugmentationAdapter(
            config=config,
            output_dir=output_dir,
            logger=self.logger
        )
        
        # Buat augmentation component
        self.augmentation_component = AugmentationComponent(
            config=config,
            augmentation_adapter=self.augmentation_adapter,
            logger=self.logger
        )
        
        # Tambahkan ke pipeline
        self.add_component(self.augmentation_component)
        
        # Tambahkan progress observer jika diminta
        if add_progress_observer:
            self.add_observer(ProgressObserver(logger=self.logger))
    
    def augment(
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
        Jalankan pipeline augmentasi dataset.
        
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
        if augmentation_types is None:
            augmentation_types = ['combined', 'lighting']
            
        # Parameters untuk augmentasi
        params = {
            'split': split,
            'augmentation_types': augmentation_types,
            'num_variations': num_variations,
            'output_prefix': output_prefix,
            'resume': resume,
            'validate_results': validate_results,
            **kwargs
        }
        
        # Jalankan pipeline
        results = self.run(**params)
        
        # Ekstrak hasil augmentasi
        augmentation_results = {}
        if results['status'] == 'success':
            for component in results['components']:
                if component['name'] == 'AugmentationComponent' and component['status'] == 'success':
                    augmentation_results = component.get('result', {})
        
        return augmentation_results
    
    def augment_with_combinations(
        self,
        split: str = 'train',
        combinations: Optional[List[Dict[str, Any]]] = None,
        base_output_prefix: str = 'aug',
        resume: bool = True,
        validate_results: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan pipeline augmentasi dataset dengan kombinasi parameter kustom.
        
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
        # Parameters untuk augmentasi
        params = {
            'split': split,
            'combinations': combinations,
            'base_output_prefix': base_output_prefix,
            'resume': resume,
            'validate_results': validate_results,
            **kwargs
        }
        
        # Jalankan pipeline dengan metode khusus pada komponen
        return self.augmentation_component.process_with_combinations(**params)
    
    def augment_all_splits(
        self,
        splits: Optional[List[str]] = None,
        augmentation_types: Optional[List[str]] = None,
        num_variations: int = 3,
        output_prefix: str = 'aug',
        resume: bool = True,
        validate_results: bool = True,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Augmentasi semua split dataset.
        
        Args:
            splits: List split yang akan diaugmentasi (default: ['train'])
            augmentation_types: Jenis augmentasi ('combined', 'lighting', 'position', dll)
            num_variations: Jumlah variasi yang akan dibuat untuk setiap gambar
            output_prefix: Prefix untuk file hasil augmentasi
            resume: Lanjutkan proses augmentasi yang terganggu
            validate_results: Validasi hasil augmentasi
            **kwargs: Parameter tambahan
            
        Returns:
            Dict[str, Dict[str, Any]]: Hasil augmentasi per split
        """
        if splits is None:
            splits = ['train']  # Default hanya train, karena valid/test biasanya tidak diaugmentasi
            
        if augmentation_types is None:
            augmentation_types = ['combined', 'lighting']
            
        self.logger.start(f"🎨 Memulai augmentasi untuk {len(splits)} splits: {', '.join(splits)}")
        
        results = {}
        for split in splits:
            self.logger.info(f"🔄 Augmentasi split: {split}")
            split_result = self.augment(
                split=split,
                augmentation_types=augmentation_types,
                num_variations=num_variations,
                output_prefix=f"{output_prefix}_{split}",
                resume=resume,
                validate_results=validate_results,
                **kwargs
            )
            results[split] = split_result
        
        # Hitung total statistik
        total_augmented = sum(r.get('augmentation_stats', {}).get('augmented', 0) for r in results.values())
        
        self.logger.success(
            f"✅ Augmentasi semua splits selesai: {total_augmented} total gambar dihasilkan"
        )
        
        return results