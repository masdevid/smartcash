"""
File: smartcash/handlers/preprocessing/integration/validator_adapter.py
Author: Alfrida Sabar
Deskripsi: Adapter untuk EnhancedDatasetValidator dari utils/dataset yang menyediakan
           antarmuka konsisten dengan komponen preprocessing lainnya.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path

from smartcash.utils.dataset import EnhancedDatasetValidator
from smartcash.utils.logger import SmartCashLogger, get_logger


class ValidatorAdapter:
    """
    Adapter untuk EnhancedDatasetValidator dari utils/dataset.
    Menyediakan antarmuka yang konsisten dengan komponen preprocessing lainnya.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        data_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None,
        **kwargs
    ):
        """
        Inisialisasi adapter untuk validator.
        
        Args:
            config: Konfigurasi untuk validator
            data_dir: Direktori data (opsional)
            logger: Logger kustom (opsional)
            **kwargs: Parameter tambahan
        """
        self.config = config
        self.logger = logger or get_logger("ValidatorAdapter")
        
        # Ambil direktori data dari config jika tidak diberikan
        self.data_dir = data_dir or config.get('data_dir', config.get('data', {}).get('data_dir', 'data'))
        
        # Ambil jumlah workers dari config
        num_workers = self._get_config_value(
            'data.preprocessing.num_workers', 
            default=config.get('num_workers', 4)
        )
        
        # Buat instance validator
        self.validator = EnhancedDatasetValidator(
            config=config,
            data_dir=self.data_dir,
            logger=self.logger,
            num_workers=num_workers
        )
    
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
        Validasi dataset menggunakan EnhancedDatasetValidator.
        
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
        self.logger.start(f"🔍 Memulai validasi dataset untuk split '{split}'")
        
        # Panggil validator dari utils
        validation_stats = self.validator.validate_dataset(
            split=split,
            fix_issues=fix_issues,
            move_invalid=move_invalid,
            visualize=visualize,
            sample_size=sample_size,
            **kwargs
        )
        
        # Format hasil untuk konsistensi dengan komponen preprocessing lainnya
        result = {
            'status': 'success' if validation_stats['valid_images'] == validation_stats['total_images'] else 'warning',
            'validation_stats': validation_stats,
            'split': split,
            'parameters': {
                'fix_issues': fix_issues,
                'move_invalid': move_invalid,
                'visualize': visualize,
                'sample_size': sample_size
            }
        }
        
        # Log ringkasan hasil
        valid_percent = (validation_stats['valid_images'] / validation_stats['total_images'] * 100) if validation_stats['total_images'] > 0 else 0
        self.logger.success(
            f"✅ Validasi selesai: {validation_stats['valid_images']}/{validation_stats['total_images']} "
            f"gambar valid ({valid_percent:.1f}%)"
        )
        
        return result
    
    def analyze(
        self, 
        split: str = 'train',
        sample_size: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analisis dataset menggunakan EnhancedDatasetValidator.
        
        Args:
            split: Split dataset yang akan dianalisis (train/valid/test)
            sample_size: Jumlah sampel yang akan dianalisis (0 = semua)
            **kwargs: Parameter tambahan
            
        Returns:
            Dict[str, Any]: Hasil analisis
        """
        self.logger.start(f"📊 Memulai analisis dataset untuk split '{split}'")
        
        # Panggil analyzer dari utils
        analysis = self.validator.analyze_dataset(
            split=split, 
            sample_size=sample_size,
            **kwargs
        )
        
        # Format hasil untuk konsistensi dengan komponen preprocessing lainnya
        result = {
            'status': 'success',
            'analysis': analysis,
            'split': split,
            'parameters': {
                'sample_size': sample_size
            }
        }
        
        # Log ringkasan hasil
        dominant_size = analysis['image_size_distribution']['dominant_size']
        class_imbalance = analysis['class_balance']['imbalance_score']
        self.logger.success(
            f"✅ Analisis selesai: Ukuran gambar dominan: {dominant_size}, "
            f"Ketidakseimbangan kelas: {class_imbalance:.2f}/10"
        )
        
        return result
    
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
        from smartcash.utils.dataset import DatasetFixer
        
        self.logger.start(f"🔧 Memulai perbaikan dataset untuk split '{split}'")
        
        # Buat DatasetFixer
        fixer = DatasetFixer(
            config=self.config,
            data_dir=self.data_dir,
            logger=self.logger
        )
        
        # Jalankan perbaikan
        fix_stats = fixer.fix_dataset(
            split=split,
            fix_coordinates=fix_coordinates,
            fix_labels=fix_labels,
            fix_images=fix_images,
            backup=backup,
            **kwargs
        )
        
        # Format hasil untuk konsistensi dengan komponen preprocessing lainnya
        result = {
            'status': 'success',
            'fix_stats': fix_stats,
            'split': split,
            'parameters': {
                'fix_coordinates': fix_coordinates,
                'fix_labels': fix_labels,
                'fix_images': fix_images,
                'backup': backup
            }
        }
        
        # Log ringkasan hasil
        self.logger.success(
            f"✅ Perbaikan selesai: {fix_stats['fixed_labels']} label dan "
            f"{fix_stats['fixed_coordinates']} koordinat diperbaiki"
        )
        
        return result
    
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