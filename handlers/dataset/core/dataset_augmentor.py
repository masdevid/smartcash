# File: smartcash/handlers/dataset/core/dataset_augmentor.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk augmentasi dataset SmartCash dengan integrasi AugmentationManager

from typing import Dict, List, Optional, Any
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.augmentation import AugmentationManager


class DatasetAugmentor:
    """Komponen untuk augmentasi dataset dengan berbagai metode."""
    
    def __init__(self, config: Dict, data_dir: Optional[str] = None, logger: Optional[SmartCashLogger] = None,
                num_workers: int = 4):
        self.config = config
        self.data_dir = Path(data_dir or config.get('data_dir', 'data'))
        self.logger = logger or SmartCashLogger(__name__)
        self.num_workers = num_workers
        
        # Inisialisasi AugmentationManager dari utils
        self.augmentation_manager = AugmentationManager(
            config=config,
            output_dir=str(self.data_dir),
            logger=logger,
            num_workers=num_workers,
            checkpoint_interval=50  # Checkpoint tiap 50 gambar untuk resume
        )
        
        self.logger.info(
            f"🎨 DatasetAugmentor diinisialisasi:\n"
            f"   • Data dir: {self.data_dir}\n"
            f"   • Num workers: {num_workers}"
        )
    
    def augment_dataset(self, **kwargs) -> Dict[str, Any]:
        """Augmentasi dataset menggunakan AugmentationManager."""
        # Ekstrak parameter umum
        split = kwargs.pop('split', 'train')
        augmentation_types = kwargs.pop('augmentation_types', ['combined'])
        num_variations = kwargs.pop('num_variations', 2)
        output_prefix = kwargs.pop('output_prefix', 'aug')
        resume = kwargs.pop('resume', True)
        validate_results = kwargs.pop('validate_results', True)
        
        self.logger.info(
            f"🎨 Augmentasi dataset '{split}':\n"
            f"   • Jenis: {augmentation_types}\n"
            f"   • Variasi per gambar: {num_variations}\n"
            f"   • Prefix output: {output_prefix}\n"
            f"   • Resume: {resume}\n"
            f"   • Validasi hasil: {validate_results}"
        )
        
        # Lakukan augmentasi menggunakan manager
        stats = self.augmentation_manager.augment_dataset(
            split=split,
            augmentation_types=augmentation_types,
            num_variations=num_variations,
            output_prefix=output_prefix,
            resume=resume,
            validate_results=validate_results,
            **kwargs  # Teruskan parameter tambahan
        )
        
        # Log hasil
        self.logger.success(
            f"✅ Augmentasi selesai:\n"
            f"   • Total gambar yang dibuat: {stats.get('augmented', 0)}\n"
            f"   • Waktu eksekusi: {stats.get('duration', 0):.2f} detik\n"
            f"   • Gambar per detik: {stats.get('images_per_second', 0):.2f}"
        )
        
        return stats
    
    def augment_with_combinations(self, **kwargs) -> Dict[str, Any]:
        """Augmentasi dataset dengan kombinasi parameter kustom."""
        # Ekstrak parameter umum
        split = kwargs.pop('split', 'train')
        combinations = kwargs.pop('combinations', [{'translate': 0.2, 'scale': 0.6}])
        output_prefix = kwargs.pop('output_prefix', 'aug')
        resume = kwargs.pop('resume', True)
        validate_results = kwargs.pop('validate_results', True)
        
        self.logger.info(
            f"🎨 Augmentasi dengan kombinasi kustom untuk '{split}':\n"
            f"   • Kombinasi: {len(combinations)}\n"
            f"   • Prefix output: {output_prefix}\n"
            f"   • Resume: {resume}\n"
            f"   • Validasi hasil: {validate_results}"
        )
        
        # Lakukan augmentasi dengan kombinasi kustom
        stats = self.augmentation_manager.augment_with_combinations(
            split=split,
            combinations=combinations,
            output_prefix=output_prefix,
            resume=resume,
            validate_results=validate_results,
            **kwargs  # Teruskan parameter tambahan
        )
        
        # Log hasil
        self.logger.success(
            f"✅ Augmentasi dengan kombinasi kustom selesai:\n"
            f"   • Total gambar yang dibuat: {stats.get('augmented', 0)}\n"
            f"   • Waktu eksekusi: {stats.get('duration', 0):.2f} detik\n"
            f"   • Gambar per detik: {stats.get('images_per_second', 0):.2f}"
        )
        
        return stats
    
    def get_available_augmentation_types(self) -> List[str]:
        """Dapatkan daftar jenis augmentasi yang tersedia."""
        return [
            'position',      # Augmentasi posisi (rotasi, translasi, skala)
            'lighting',      # Augmentasi pencahayaan (brightness, contrast, hue)
            'noise',         # Augmentasi noise (gaussian, speckle)
            'combined',      # Kombinasi augmentasi posisi dan pencahayaan
            'extreme',       # Augmentasi ekstrem (rotasi besar, flip, dll)
            'edge',          # Augmentasi deteksi tepi
            'blur',          # Augmentasi blur
            'color',         # Augmentasi warna
            'geometric'      # Augmentasi geometri
        ]