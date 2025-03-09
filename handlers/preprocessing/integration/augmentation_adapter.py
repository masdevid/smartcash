"""
File: smartcash/handlers/preprocessing/integration/augmentation_adapter.py
Author: Alfrida Sabar
Deskripsi: Adapter untuk AugmentationManager dari utils/augmentation yang menyediakan
           antarmuka konsisten dengan komponen preprocessing lainnya.
"""

from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from smartcash.utils.augmentation import AugmentationManager
from smartcash.utils.logger import SmartCashLogger, get_logger


class AugmentationAdapter:
    """
    Adapter untuk AugmentationManager dari utils/augmentation.
    Menyediakan antarmuka yang konsisten dengan komponen preprocessing lainnya.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None,
        **kwargs
    ):
        """
        Inisialisasi adapter untuk augmentation manager.
        
        Args:
            config: Konfigurasi untuk augmentation
            output_dir: Direktori output (opsional)
            logger: Logger kustom (opsional)
            **kwargs: Parameter tambahan
        """
        self.config = config
        self.logger = logger or get_logger("AugmentationAdapter")
        
        # Ambil direktori output dari config jika tidak diberikan
        self.output_dir = output_dir or config.get('data_dir', config.get('data', {}).get('data_dir', 'data'))
        
        # Ambil jumlah workers dari config
        num_workers = self._get_config_value(
            'data.preprocessing.num_workers', 
            default=config.get('num_workers', 4)
        )
        
        # Buat instance augmentation manager
        self.augmentor = AugmentationManager(
            config=config,
            output_dir=self.output_dir,
            logger=self.logger,
            num_workers=num_workers,
            **kwargs
        )
    
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
        Augmentasi dataset menggunakan AugmentationManager.
        
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
            
        self.logger.start(
            f"🎨 Memulai augmentasi dataset untuk split '{split}' dengan "
            f"jenis augmentasi: {', '.join(augmentation_types)}"
        )
        
        # Panggil augmentor dari utils
        stats = self.augmentor.augment_dataset(
            split=split,
            augmentation_types=augmentation_types,
            num_variations=num_variations,
            output_prefix=output_prefix,
            resume=resume,
            validate_results=validate_results,
            **kwargs
        )
        
        # Format hasil untuk konsistensi dengan komponen preprocessing lainnya
        result = {
            'status': 'success',
            'augmentation_stats': stats,
            'split': split,
            'parameters': {
                'augmentation_types': augmentation_types,
                'num_variations': num_variations,
                'output_prefix': output_prefix,
                'resume': resume,
                'validate_results': validate_results
            }
        }
        
        # Log ringkasan hasil
        self.logger.success(
            f"✅ Augmentasi selesai: {stats['augmented']} gambar dihasilkan "
            f"dalam {stats['duration']:.2f} detik"
        )
        
        return result
    
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
        if combinations is None:
            combinations = [
                {
                    'augmentation_types': ['combined'],
                    'num_variations': 2,
                    'output_prefix': f"{base_output_prefix}_combined"
                },
                {
                    'augmentation_types': ['lighting'],
                    'num_variations': 2,
                    'output_prefix': f"{base_output_prefix}_lighting"
                }
            ]
        
        self.logger.start(
            f"🎨 Memulai augmentasi dengan {len(combinations)} kombinasi parameter"
        )
        
        results = []
        total_augmented = 0
        
        # Eksekusi setiap kombinasi
        for i, combo in enumerate(combinations):
            # Gabungkan dengan kwargs default
            combo_params = {**kwargs, **combo}
            
            # Log progress
            self.logger.info(
                f"🔄 Menjalankan kombinasi {i+1}/{len(combinations)}: "
                f"{combo['augmentation_types']} dengan {combo['num_variations']} variasi"
            )
            
            # Jalankan augmentasi untuk kombinasi ini
            result = self.augment(
                split=split,
                resume=resume,
                validate_results=validate_results,
                **combo_params
            )
            
            results.append(result)
            total_augmented += result['augmentation_stats']['augmented']
        
        # Format hasil untuk konsistensi dengan komponen preprocessing lainnya
        final_result = {
            'status': 'success',
            'combination_results': results,
            'total_augmented': total_augmented,
            'combinations_count': len(combinations),
            'split': split,
            'parameters': {
                'combinations': combinations,
                'base_output_prefix': base_output_prefix,
                'resume': resume,
                'validate_results': validate_results
            }
        }
        
        # Log ringkasan hasil
        self.logger.success(
            f"✅ Augmentasi kombinasi selesai: {total_augmented} total gambar dihasilkan "
            f"dari {len(combinations)} kombinasi"
        )
        
        return final_result
    
    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Helper untuk mendapatkan nilai dari hierarki konfigurasi.
        Mendukung dot notation (misalnya 'data.preprocessing.cache_dir').
        
        Args:
            key: Kunci konfigurasi, dapat menggunakan dot notation
            default: Nilai default jika kunci tidak ditemukan
            
        Returns:
            Any: Nilai dari konfigurasi
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default