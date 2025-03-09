"""
File: smartcash/handlers/preprocessing/preprocessing_manager.py
Author: Alfrida Sabar
Deskripsi: Manager utama preprocessing yang bertindak sebagai facade untuk semua
           komponen preprocessing, pipeline, dan integrasi dengan lingkungan berbeda.
"""

from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import time
import os

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.handlers.preprocessing.pipeline.preprocessing_pipeline import PreprocessingPipeline
from smartcash.handlers.preprocessing.pipeline.validation_pipeline import ValidationPipeline
from smartcash.handlers.preprocessing.pipeline.augmentation_pipeline import AugmentationPipeline
from smartcash.handlers.preprocessing.integration.validator_adapter import ValidatorAdapter
from smartcash.handlers.preprocessing.integration.augmentation_adapter import AugmentationAdapter
from smartcash.handlers.preprocessing.integration.cache_adapter import CacheAdapter
from smartcash.handlers.preprocessing.integration.colab_drive_adapter import ColabDriveAdapter
from smartcash.handlers.preprocessing.observers.progress_observer import ProgressObserver


class PreprocessingManager:
    """
    Manager utama preprocessing sebagai facade.
    Menyembunyikan kompleksitas dan meningkatkan usability.
    """
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[SmartCashLogger] = None,
        colab_mode: Optional[bool] = None,
        drive_adapter: Optional[ColabDriveAdapter] = None
    ):
        """
        Inisialisasi PreprocessingManager.
        
        Args:
            config: Konfigurasi preprocessing
            logger: Logger kustom (opsional)
            colab_mode: Mode Google Colab (opsional, auto-detect jika None)
            drive_adapter: ColabDriveAdapter kustom (opsional)
        """
        self.config = config or {}
        self.logger = logger or get_logger("PreprocessingManager")
        
        # Deteksi otomatis colab jika tidak diberikan
        if colab_mode is None:
            self.colab_mode = self._is_running_in_colab()
        else:
            self.colab_mode = colab_mode
        
        # Setup drive adapter jika di Colab
        if self.colab_mode:
            if drive_adapter is None:
                self.drive_adapter = ColabDriveAdapter(
                    logger=self.logger,
                    auto_mount=True
                )
            else:
                self.drive_adapter = drive_adapter
                if not self.drive_adapter.is_mounted:
                    self.drive_adapter.mount_drive()
        else:
            self.drive_adapter = None
        
        # Inisialisasi pipelines dan adapters (lazy)
        self._pipelines = {}
        self._adapters = {}
        self._observers = {}
        
        self.logger.start("🚀 PreprocessingManager diinisialisasi")
        
        # Log mode
        if self.colab_mode:
            self.logger.info("🔍 Berjalan di Google Colab dengan integrasi Drive")
        else:
            self.logger.info("🔍 Berjalan di lingkungan lokal")
    
    def run_full_pipeline(
        self,
        splits: Optional[List[str]] = None,
        validate_dataset: bool = True,
        fix_issues: bool = False,
        augment_data: bool = False,
        analyze_dataset: bool = True,
        report_format: str = 'json',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan pipeline preprocessing lengkap.
        
        Args:
            splits: List split dataset yang akan diproses (default: ['train', 'valid', 'test'])
            validate_dataset: Lakukan validasi dataset
            fix_issues: Perbaiki masalah dataset yang ditemukan
            augment_data: Lakukan augmentasi dataset
            analyze_dataset: Analisis dataset
            report_format: Format laporan ('json' atau 'html')
            **kwargs: Parameter tambahan untuk pipeline
            
        Returns:
            Dict[str, Any]: Hasil pipeline lengkap
        """
        if splits is None:
            splits = ['train', 'valid', 'test']
        
        start_time = time.time()
        self.logger.start(f"🚀 Memulai pipeline preprocessing lengkap untuk {len(splits)} split")
        
        results = {
            'status': 'success',
            'splits': {},
            'validation': {},
            'augmentation': {},
            'analysis': {}
        }
        
        try:
            # Validasi dataset
            if validate_dataset:
                validation_pipeline = self._get_validation_pipeline()
                validation_results = validation_pipeline.validate_all_splits(
                    splits=splits,
                    fix_issues=fix_issues,
                    **kwargs
                )
                results['validation'] = validation_results
                
                # Jika ada masalah validasi dan tidak fix_issues, berhenti
                if not fix_issues:
                    for split, split_result in validation_results.items():
                        stats = split_result.get('validation_stats', {})
                        if stats.get('valid_images', 0) != stats.get('total_images', 0):
                            self.logger.warning(
                                f"⚠️ Split {split} memiliki masalah validasi. "
                                f"Jalankan dengan fix_issues=True untuk memperbaiki."
                            )
                
            # Analisis dataset
            if analyze_dataset:
                validation_pipeline = self._get_validation_pipeline()
                analysis_results = {}
                
                for split in splits:
                    self.logger.info(f"🔄 Menganalisis split: {split}")
                    analysis = validation_pipeline.analyze(split=split, **kwargs)
                    analysis_results[split] = analysis
                
                results['analysis'] = analysis_results
            
            # Augmentasi dataset
            if augment_data:
                augmentation_pipeline = self._get_augmentation_pipeline()
                # Hanya augmentasi split train
                train_splits = [s for s in splits if s == 'train']
                
                if train_splits:
                    augmentation_results = augmentation_pipeline.augment_all_splits(
                        splits=train_splits,
                        **kwargs
                    )
                    results['augmentation'] = augmentation_results
                else:
                    self.logger.warning("⚠️ Tidak ada split 'train' untuk augmentasi")
            
            # Bangun laporan
            elapsed = time.time() - start_time
            results['elapsed'] = elapsed
            
            self.logger.success(f"✅ Pipeline preprocessing lengkap selesai dalam {elapsed:.2f} detik")
            
            return results
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"❌ Pipeline preprocessing gagal: {str(e)}")
            
            return {
                'status': 'error',
                'error': str(e),
                'elapsed': elapsed
            }
    
    def validate_dataset(
        self, 
        split: str = 'train',
        fix_issues: bool = False,
        move_invalid: bool = False,
        visualize: bool = True,
        sample_size: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan validasi dataset saja.
        
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
        validation_pipeline = self._get_validation_pipeline()
        return validation_pipeline.validate(
            split=split,
            fix_issues=fix_issues,
            move_invalid=move_invalid,
            visualize=visualize,
            sample_size=sample_size,
            **kwargs
        )
    
    def augment_dataset(
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
        Jalankan augmentasi dataset saja.
        
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
        augmentation_pipeline = self._get_augmentation_pipeline()
        return augmentation_pipeline.augment(
            split=split,
            augmentation_types=augmentation_types,
            num_variations=num_variations,
            output_prefix=output_prefix,
            resume=resume,
            validate_results=validate_results,
            **kwargs
        )
    
    def analyze_dataset(
        self,
        split: str = 'train',
        sample_size: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Jalankan analisis dataset saja.
        
        Args:
            split: Split dataset yang akan dianalisis (train/valid/test)
            sample_size: Jumlah sampel yang akan dianalisis (0 = semua)
            **kwargs: Parameter tambahan
            
        Returns:
            Dict[str, Any]: Hasil analisis
        """
        validation_pipeline = self._get_validation_pipeline()
        return validation_pipeline.analyze(
            split=split,
            sample_size=sample_size,
            **kwargs
        )
    
    def generate_report(
        self,
        results: Dict[str, Any],
        report_format: str = 'json',
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Hasilkan laporan preprocessing.
        
        Args:
            results: Hasil preprocessing
            report_format: Format laporan ('json', 'html', 'md')
            output_path: Path output laporan (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dict[str, Any]: Informasi laporan
        """
        # TODO: Implementasi generator laporan
        from datetime import datetime
        import json
        
        # Siapkan output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"reports/preprocessing_report_{timestamp}.{report_format}")
        else:
            output_path = Path(output_path)
        
        # Pastikan direktori ada
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.start(f"📊 Membuat laporan preprocessing: {output_path}")
        
        # Simpan laporan dalam format yang sesuai
        if report_format == 'json':
            # Simpan sebagai JSON
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            # Format lain belum diimplementasi
            self.logger.warning(f"⚠️ Format laporan '{report_format}' belum diimplementasi")
            
        self.logger.success(f"✅ Laporan preprocessing dibuat: {output_path}")
        
        return {
            'status': 'success',
            'report_path': str(output_path),
            'report_format': report_format
        }
    
    def setup_colab(
        self,
        project_dir: str = "/content/SmartCash",
        drive_mount_point: str = "/content/drive",
        drive_project_path: str = "MyDrive/SmartCash",
        auto_mount: bool = True,
        setup_symlinks: bool = True,
        symlink_dirs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Setup lingkungan Google Colab.
        
        Args:
            project_dir: Direktori lokal project di Colab
            drive_mount_point: Titik mount Google Drive
            drive_project_path: Jalur project di Google Drive (relatif terhadap mount_point)
            auto_mount: Mount Google Drive secara otomatis
            setup_symlinks: Setup symlink otomatis
            symlink_dirs: List direktori yang akan di-symlink (default: ['data', 'configs', 'models'])
            
        Returns:
            Dict[str, Any]: Status setup
        """
        if not self.colab_mode:
            self.logger.warning("⚠️ Bukan lingkungan Google Colab, setup tidak diperlukan")
            return {'status': 'skipped', 'colab_mode': False}
        
        # Default symlink dirs
        if symlink_dirs is None:
            symlink_dirs = ['data', 'configs', 'models']
        
        # Initialize drive adapter dengan parameter baru
        self.drive_adapter = ColabDriveAdapter(
            project_dir=project_dir,
            drive_mount_point=drive_mount_point,
            drive_project_path=drive_project_path,
            logger=self.logger,
            auto_mount=auto_mount
        )
        
        # Setup symlinks jika diminta
        symlink_status = {}
        if setup_symlinks and self.drive_adapter.is_mounted:
            symlink_status = self.drive_adapter.setup_symlinks(symlink_dirs)
        
        # Cek permissions
        permissions = self.drive_adapter.check_permissions()
        
        # Dapatkan informasi ruang disk
        disk_info = self.drive_adapter.get_available_space()
        
        result = {
            'status': 'success' if self.drive_adapter.is_mounted else 'error',
            'colab_mode': True,
            'drive_mounted': self.drive_adapter.is_mounted,
            'symlinks': symlink_status,
            'permissions': permissions,
            'disk_info': disk_info
        }
        
        if self.drive_adapter.is_mounted:
            self.logger.success("✅ Setup Colab selesai")
        else:
            self.logger.error("❌ Setup Colab gagal: Google Drive tidak ter-mount")
        
        return result
    
    def _get_validation_pipeline(self) -> ValidationPipeline:
        """
        Dapatkan ValidationPipeline dengan lazy initialization.
        
        Returns:
            ValidationPipeline: Pipeline validasi
        """
        if 'validation' not in self._pipelines:
            self._pipelines['validation'] = ValidationPipeline(
                config=self.config,
                logger=self.logger,
                validator_adapter=self._get_adapter('validator'),
                add_progress_observer=True
            )
        return self._pipelines['validation']
    
    def _get_augmentation_pipeline(self) -> AugmentationPipeline:
        """
        Dapatkan AugmentationPipeline dengan lazy initialization.
        
        Returns:
            AugmentationPipeline: Pipeline augmentasi
        """
        if 'augmentation' not in self._pipelines:
            self._pipelines['augmentation'] = AugmentationPipeline(
                config=self.config,
                logger=self.logger,
                augmentation_adapter=self._get_adapter('augmentation'),
                add_progress_observer=True
            )
        return self._pipelines['augmentation']
    
    def _get_adapter(self, adapter_type: str) -> Any:
        """
        Dapatkan adapter dengan lazy initialization.
        
        Args:
            adapter_type: Tipe adapter ('validator', 'augmentation', 'cache')
            
        Returns:
            Any: Adapter yang diminta
        """
        if adapter_type not in self._adapters:
            if adapter_type == 'validator':
                self._adapters[adapter_type] = ValidatorAdapter(
                    config=self.config,
                    logger=self.logger
                )
            elif adapter_type == 'augmentation':
                self._adapters[adapter_type] = AugmentationAdapter(
                    config=self.config,
                    logger=self.logger
                )
            elif adapter_type == 'cache':
                self._adapters[adapter_type] = CacheAdapter(
                    config=self.config,
                    logger=self.logger
                )
            
        return self._adapters.get(adapter_type)
    
    def _is_running_in_colab(self) -> bool:
        """
        Deteksi apakah kode berjalan dalam Google Colab.
        
        Returns:
            bool: True jika berjalan di Google Colab
        """
        try:
            import google.colab
            return True
        except ImportError:
            return False