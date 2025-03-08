# File: smartcash/handlers/dataset/core/dataset_validator.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk validasi dataset SmartCash

from typing import Dict, List, Optional, Any
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.dataset import EnhancedDatasetValidator


class DatasetValidator:
    """
    Komponen untuk validasi dataset SmartCash.
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None,
        num_workers: int = 4
    ):
        """
        Inisialisasi DatasetValidator.
        
        Args:
            config: Konfigurasi dataset
            data_dir: Direktori dataset (opsional)
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk paralelisasi
        """
        self.config = config
        self.data_dir = Path(data_dir or config.get('data_dir', 'data'))
        self.logger = logger or SmartCashLogger(__name__)
        self.num_workers = num_workers
        
        # Inisialisasi validator dari utils
        self.enhanced_validator = EnhancedDatasetValidator(
            config=config,
            data_dir=str(self.data_dir),
            logger=logger,
            num_workers=num_workers
        )
        
        self.logger.info(
            f"🔍 DatasetValidator diinisialisasi:\n"
            f"   • Data dir: {self.data_dir}\n"
            f"   • Num workers: {num_workers}"
        )
    
    def validate_dataset(
        self,
        split: str,
        fix_issues: bool = False,
        visualize: bool = True,
        sample_size: int = 0,
        move_invalid: bool = False
    ) -> Dict[str, Any]:
        """
        Validasi dataset menggunakan EnhancedDatasetValidator.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            fix_issues: Jika True, coba perbaiki masalah yang ditemukan
            visualize: Jika True, buat visualisasi masalah
            sample_size: Jumlah sampel yang akan divalidasi (0 untuk semua)
            move_invalid: Jika True, pindahkan file yang tidak valid
            
        Returns:
            Dict berisi statistik validasi
        """
        self.logger.info(
            f"🔍 Validasi dataset '{split}':\n"
            f"   • Fix issues: {fix_issues}\n"
            f"   • Visualize: {visualize}\n"
            f"   • Sample size: {sample_size if sample_size > 0 else 'all'}\n"
            f"   • Move invalid: {move_invalid}"
        )
        
        # Lakukan validasi
        result = self.enhanced_validator.validate_dataset(
            split=split,
            fix_issues=fix_issues,
            visualize=visualize,
            sample_size=sample_size,
            move_invalid=move_invalid
        )
        
        # Log hasil
        valid_images = result.get('valid_images', 0)
        total_images = result.get('total_images', 0)
        valid_labels = result.get('valid_labels', 0)
        total_labels = result.get('total_labels', 0)
        
        valid_image_percent = (valid_images / max(1, total_images)) * 100
        valid_label_percent = (valid_labels / max(1, total_labels)) * 100
        
        self.logger.info(
            f"📊 Hasil validasi '{split}':\n"
            f"   • Gambar valid: {valid_images}/{total_images} ({valid_image_percent:.1f}%)\n"
            f"   • Label valid: {valid_labels}/{total_labels} ({valid_label_percent:.1f}%)"
        )
        
        if result.get('fixed_labels', 0) > 0 or result.get('fixed_coordinates', 0) > 0:
            self.logger.success(
                f"🔧 Perbaikan yang dilakukan:\n"
                f"   • Label diperbaiki: {result.get('fixed_labels', 0)}\n"
                f"   • Koordinat diperbaiki: {result.get('fixed_coordinates', 0)}"
            )
        
        return result
    
    def analyze_dataset(
        self,
        split: str,
        sample_size: int = 0,
        detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Analisis dataset menggunakan EnhancedDatasetValidator.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            sample_size: Jumlah sampel yang akan dianalisis (0 untuk semua)
            detailed: Jika True, lakukan analisis lebih mendalam
            
        Returns:
            Dict berisi hasil analisis dataset
        """
        self.logger.info(
            f"🔍 Analisis dataset '{split}':\n"
            f"   • Sample size: {sample_size if sample_size > 0 else 'all'}\n"
            f"   • Detailed: {detailed}"
        )
        
        # Lakukan analisis
        result = self.enhanced_validator.analyze_dataset(
            split=split,
            sample_size=sample_size,
            detailed=detailed
        )
        
        # Log hasil utama
        class_balance = result.get('class_balance', {})
        layer_balance = result.get('layer_balance', {})
        image_size_distribution = result.get('image_size_distribution', {})
        
        self.logger.info(
            f"📊 Hasil analisis '{split}':\n"
            f"   • Ketidakseimbangan kelas: {class_balance.get('imbalance_score', 0):.2f}/10\n"
            f"   • Ketidakseimbangan layer: {layer_balance.get('imbalance_score', 0):.2f}/10\n"
            f"   • Ukuran gambar dominan: {image_size_distribution.get('dominant_size', 'N/A')}"
        )
        
        # Log kelas underrepresented dan overrepresented
        underrepresented = class_balance.get('underrepresented_classes', [])
        overrepresented = class_balance.get('overrepresented_classes', [])
        
        if underrepresented:
            self.logger.info(f"⚠️ Kelas kurang terwakili: {', '.join(underrepresented[:5])}")
        
        if overrepresented:
            self.logger.info(f"ℹ️ Kelas dominan: {', '.join(overrepresented[:5])}")
        
        return result
    
    def fix_dataset(
        self,
        split: str,
        fix_coordinates: bool = True,
        fix_labels: bool = True,
        fix_images: bool = False,
        backup: bool = True
    ) -> Dict[str, Any]:
        """
        Perbaiki masalah dataset menggunakan EnhancedDatasetValidator.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            fix_coordinates: Jika True, perbaiki koordinat tidak valid
            fix_labels: Jika True, perbaiki format label
            fix_images: Jika True, perbaiki gambar rusak (resize, konversi format)
            backup: Jika True, buat backup sebelum perbaikan
            
        Returns:
            Dict berisi statistik perbaikan
        """
        self.logger.info(
            f"🔧 Perbaikan dataset '{split}':\n"
            f"   • Fix coordinates: {fix_coordinates}\n"
            f"   • Fix labels: {fix_labels}\n"
            f"   • Fix images: {fix_images}\n"
            f"   • Backup: {backup}"
        )
        
        # Lakukan perbaikan
        result = self.enhanced_validator.fix_dataset(
            split=split,
            fix_coordinates=fix_coordinates,
            fix_labels=fix_labels,
            fix_images=fix_images,
            backup=backup
        )
        
        # Log hasil
        self.logger.success(
            f"✅ Perbaikan dataset '{split}' selesai:\n"
            f"   • File diproses: {result.get('processed', 0)}\n"
            f"   • Label diperbaiki: {result.get('fixed_labels', 0)}\n"
            f"   • Koordinat diperbaiki: {result.get('fixed_coordinates', 0)}\n"
            f"   • Gambar diperbaiki: {result.get('fixed_images', 0)}"
        )
        
        if backup and result.get('backup_dir'):
            self.logger.info(f"💾 Backup dibuat di: {result.get('backup_dir')}")
        
        return result
    
    def get_validator_instance(self) -> EnhancedDatasetValidator:
        """
        Dapatkan instance EnhancedDatasetValidator langsung.
        
        Returns:
            Instance EnhancedDatasetValidator
        """
        return self.enhanced_validator