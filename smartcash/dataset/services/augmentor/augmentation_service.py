"""
File: smartcash/dataset/services/augmentor/augmentation_service.py
Deskripsi: Layanan augmentasi dataset yang DRY menggunakan helper dan worker yang sudah ada (fixed logger)
"""

import os
import time
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.common.exceptions import DatasetError
from smartcash.common.config import get_config_manager
from smartcash.dataset.services.augmentor.pipeline_factory import AugmentationPipelineFactory
from smartcash.dataset.services.augmentor.bbox_augmentor import BBoxAugmentor
from smartcash.dataset.services.augmentor.balanced_class_manager import get_balanced_class_manager

# Import helper yang sudah ada
from smartcash.dataset.services.augmentor.helpers.path_helper import setup_paths, ensure_output_directories
from smartcash.dataset.services.augmentor.helpers.validation_helper import validate_input_files
from smartcash.dataset.services.augmentor.helpers.augmentation_executor import execute_augmentation_with_tracking
from smartcash.dataset.services.augmentor.helpers.ensure_output import verify_augmentation_files

class AugmentationService:
    """Layanan augmentasi dataset yang DRY menggunakan helper yang sudah ada."""
    
    def __init__(self, config: Dict[str, Any] = None, data_dir: str = 'data', 
                 logger=None, num_workers: int = 1, ui_components: Dict[str, Any] = None):
        """
        Inisialisasi AugmentationService.
        
        Args:
            config: Konfigurasi augmentasi
            data_dir: Direktori data
            logger: Logger untuk logging
            num_workers: Jumlah worker (1 untuk synchronous)
            ui_components: Komponen UI untuk notifikasi
        """
        self.config = config or {}
        self.logger = logger or get_logger(__name__)
        self.num_workers = 1  # Force synchronous untuk Colab
        self.ui_components = ui_components or {}
        self.config_manager = get_config_manager()
        
        # Setup paths dengan helper yang ada
        self.is_colab = self._detect_colab_environment()
        self.data_dir = Path(data_dir)
        self.augmented_dir = self.data_dir / 'augmented'
        
        # Initialize components menggunakan yang sudah ada
        self.pipeline_factory = AugmentationPipelineFactory(self.config, self.logger)
        self.bbox_augmentor = BBoxAugmentor(self.config, self.logger)
        self.balanced_manager = get_balanced_class_manager(self.logger)
        
        # Stop signal
        self._stop_signal = False
        
        # Log initialization
        storage_info = "Google Drive (via symlink)" if self.is_colab and self.data_dir.is_symlink() else "Local storage"
        self.logger.info(f"✅ AugmentationService diinisialisasi ({storage_info})")
    
    def _detect_colab_environment(self) -> bool:
        """Deteksi apakah berjalan di Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def augment_dataset(
        self,
        data_dir: str = None,
        split: str = 'train',
        types: List[str] = None,
        num_variations: int = 2,
        target_count: int = 500,
        output_prefix: str = 'aug_',
        balance_classes: bool = False,
        validate_results: bool = True,
        progress_callback: Callable = None,
        create_symlinks: bool = True
    ) -> Dict[str, Any]:
        """
        Augmentasi dataset menggunakan helper yang sudah ada.
        
        Args:
            data_dir: Direktori data
            split: Split dataset ('train', 'valid', 'test')
            types: Jenis augmentasi yang akan digunakan
            num_variations: Jumlah variasi per gambar
            target_count: Target jumlah gambar per kelas
            output_prefix: Prefix untuk file output
            balance_classes: Gunakan balanced class manager
            validate_results: Validasi hasil augmentasi
            progress_callback: Callback untuk progress tracking
            create_symlinks: Buat symlink hasil ke direktori preprocessed
            
        Returns:
            Dict: Statistik hasil augmentasi
        """
        start_time = time.time()
        self._stop_signal = False
        
        try:
            # Setup paths menggunakan helper yang ada
            working_data_dir = data_dir or str(self.data_dir)
            config_with_paths = {**self.config, 'data': {'dir': working_data_dir}}
            
            paths = setup_paths(config_with_paths, split)
            
            # Ensure output directories menggunakan helper yang ada
            if not ensure_output_directories(paths):
                raise DatasetError("Gagal membuat direktori output")
            
            # Validate input files menggunakan helper yang ada
            image_files, validation_result = validate_input_files(
                paths['images_input_dir'], '', self.logger
            )
            
            if not validation_result['success']:
                raise DatasetError(validation_result['error'])
            
            self.logger.info(f"📊 Dataset loaded: {len(image_files)} pasangan file")
            
            # Prepare balanced dataset jika diperlukan
            class_data = self._prepare_class_data(
                image_files, paths['labels_input_dir'], 
                balance_classes, target_count, progress_callback
            )
            
            # Execute augmentation menggunakan helper yang ada
            augmentation_result = execute_augmentation_with_tracking(
                self,  # service instance
                class_data,
                types or ['combined'],
                num_variations,
                output_prefix,
                validate_results,
                True,  # process_bboxes
                self.num_workers,
                paths,
                split,
                target_count,
                start_time,
                create_symlinks
            )
            
            # Verify output jika diperlukan
            if validate_results:
                verification = verify_augmentation_files(paths, output_prefix, self.logger)
                if verification['status'] != 'success':
                    self.logger.warning(f"⚠️ Verifikasi output: {verification['message']}")
            
            # Format hasil menggunakan yang sudah ada
            duration = time.time() - start_time
            success_rate = (augmentation_result.get('generated', 0) / 
                          len(class_data.get('selected_files', [1])) * 100) if class_data.get('selected_files') else 0
            
            status = 'success' if success_rate > 80 else 'warning'
            if self._stop_signal:
                status = 'cancelled'
            
            return {
                'status': status,
                'message': f"Augmentasi {split} selesai: {augmentation_result.get('generated', 0)} gambar dibuat",
                'generated_images': augmentation_result.get('generated', 0),
                'processed': augmentation_result.get('files_augmented', 0),
                'success_rate': success_rate,
                'duration': duration,
                'split': split,
                'output_dir': paths['output_dir'],
                'class_stats': augmentation_result.get('class_stats', {}),
                'storage_type': 'Google Drive' if (self.is_colab and self.data_dir.is_symlink()) else 'Local'
            }
            
        except Exception as e:
            error_msg = f"Error saat augmentasi dataset: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {
                'status': 'error',
                'message': error_msg,
                'generated_images': 0,
                'processed': 0,
                'duration': time.time() - start_time
            }
    
    def _prepare_class_data(
        self, 
        image_files: List[str], 
        labels_dir: str, 
        balance_classes: bool, 
        target_count: int,
        progress_callback: Callable = None
    ) -> Dict[str, Any]:
        """
        Prepare class data menggunakan balanced class manager yang sudah ada.
        
        Args:
            image_files: List file gambar
            labels_dir: Direktori label
            balance_classes: Apakah menggunakan balancing
            target_count: Target count per kelas
            progress_callback: Progress callback
            
        Returns:
            Dictionary class data
        """
        if balance_classes:
            self.logger.info("⚖️ Menggunakan balanced class manager (Layer 1 & 2)")
            
            # Gunakan balanced class manager yang sudah ada
            balanced_data = self.balanced_manager.prepare_balanced_augmentation(
                image_files, labels_dir, target_count
            )
            
            if balanced_data.get('selected_files'):
                self.logger.info(f"🎯 Balancing: {len(balanced_data['selected_files'])} file dipilih")
                return balanced_data
            else:
                self.logger.info("ℹ️ Tidak ada yang perlu dibalance, menggunakan semua file")
        
        # Fallback: gunakan semua file
        return {
            'class_counts': {},
            'files_by_class': {},
            'augmentation_needs': {},
            'selected_files': image_files,
            'balancing_enabled': False
        }
    
    def report_progress(
        self, 
        message: str = "", 
        status: str = "info", 
        step: int = 1,
        progress: int = None,
        total: int = None,
        **kwargs
    ) -> bool:
        """
        Report progress ke UI components (interface untuk helper yang ada).
        
        Args:
            message: Pesan progress
            status: Status progress
            step: Step saat ini
            progress: Progress saat ini
            total: Total progress
            **kwargs: Parameter tambahan
            
        Returns:
            True untuk continue, False untuk stop
        """
        # Cek stop signal
        if self._stop_signal or (self.ui_components and self.ui_components.get('stop_requested', False)):
            return False
        
        # Forward ke progress callback jika ada
        if 'progress_callback' in self.ui_components and callable(self.ui_components['progress_callback']):
            try:
                return self.ui_components['progress_callback'](progress or 0, total or 100, message)
            except Exception as e:
                self.logger.warning(f"⚠️ Error dalam progress callback: {str(e)}")
        
        return True
    
    def stop_processing(self) -> None:
        """Hentikan proses augmentasi."""
        self._stop_signal = True
        if self.ui_components:
            self.ui_components['stop_requested'] = True
        self.logger.info("🛑 Stop request diterima")
    
    def cleanup_augmentation_files(self, split: str = None) -> Dict[str, Any]:
        """
        Cleanup file augmentasi menggunakan utilities yang sudah ada.
        
        Args:
            split: Split tertentu (None = semua split)
            
        Returns:
            Dictionary hasil cleanup
        """
        try:
            # Gunakan symlink manager yang sudah ada untuk cleanup
            from smartcash.dataset.utils.symlink_manager import get_symlink_manager
            
            deleted_files = 0
            splits_processed = []
            
            # Cleanup files
            if split:
                split_dir = self.augmented_dir / split
                if split_dir.exists():
                    deleted_count = self._delete_split_files(split_dir)
                    deleted_files += deleted_count
                    splits_processed.append(split)
            else:
                for split_dir in self.augmented_dir.iterdir():
                    if split_dir.is_dir():
                        deleted_count = self._delete_split_files(split_dir)
                        deleted_files += deleted_count
                        splits_processed.append(split_dir.name)
            
            # Cleanup symlinks menggunakan manager yang sudah ada
            symlink_manager = get_symlink_manager(self.logger)
            cleanup_result = symlink_manager.cleanup_augmentation_symlinks(
                augmented_dir=self.augmented_dir,
                preprocessed_dir=self.data_dir / 'preprocessed',
                splits=splits_processed
            )
            
            self.logger.info(f"🧹 Symlink cleanup: {cleanup_result.get('message', 'Selesai')}")
            
            return {
                'status': 'success',
                'deleted_files': deleted_files,
                'splits_processed': splits_processed,
                'message': f'Cleanup selesai: {deleted_files} file dihapus dari {len(splits_processed)} split'
            }
            
        except Exception as e:
            error_msg = f"Error cleanup augmentation: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {'status': 'error', 'message': error_msg, 'deleted_files': 0}
    
    def _delete_split_files(self, split_dir: Path) -> int:
        """Helper untuk hapus files dalam split directory."""
        deleted_count = 0
        
        try:
            for subdir in ['images', 'labels']:
                subdir_path = split_dir / subdir
                if subdir_path.exists():
                    for file_path in subdir_path.glob('*'):
                        if file_path.is_file():
                            file_path.unlink()
                            deleted_count += 1
                    
                    if not any(subdir_path.iterdir()):
                        subdir_path.rmdir()
            
            if not any(split_dir.iterdir()):
                split_dir.rmdir()
                
        except Exception as e:
            self.logger.error(f"❌ Error hapus files di {split_dir}: {str(e)}")
        
        return deleted_count