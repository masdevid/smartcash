"""
File: smartcash/dataset/preprocessor/service.py
Deskripsi: Service utama untuk melakukan preprocessing dataset YOLOv5 dengan validasi dan normalisasi.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
import numpy as np
import time
import random
import os
from PIL import Image

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor.core.engine import PreprocessingEngine, PreprocessingValidator
from smartcash.dataset.preprocessor.utils import (
    validate_preprocessing_config, 
    get_default_preprocessing_config,
    ProgressBridge, 
    FileProcessor, 
    FileScanner, 
    PathResolver,
    CleanupManager
)


class PreprocessingService:
    """🎯 Service preprocessing dengan validasi dan visualisasi"""
    
    def __init__(self, config: Dict[str, Any] = None, progress_tracker=None):
        """Inisialisasi preprocessing service dengan konfigurasi.
        
        Args:
            config: Konfigurasi preprocessing (opsional)
            progress_tracker: Objek untuk melacak progress (opsional)
        """
        # Validasi dan dapatkan konfigurasi
        self.config = validate_preprocessing_config(config) if config else get_default_preprocessing_config()
        self.logger = get_logger(__name__)
        self.progress_bridge = ProgressBridge(progress_tracker) if progress_tracker else None
        
        # Inisialisasi komponen inti dengan konfigurasi yang sudah divalidasi
        self.validator = PreprocessingValidator(self.config, self.logger)
        self.engine = PreprocessingEngine(self.config)
        
        # Inisialisasi utilities
        self.file_processor = FileProcessor(self.config)
        self.file_scanner = FileScanner()
        self.path_resolver = PathResolver(self.config)
        self.cleanup_manager = CleanupManager(self.config)
        
        # Inisialisasi progress bridge
        self.progress_bridge = ProgressBridge(progress_tracker) if progress_tracker else None
        
        # Ambil konfigurasi yang sering digunakan
        self.preprocessing_config = self.config.get('preprocessing', {})
        self.validation_config = self.preprocessing_config.get('validation', {})
        self.output_config = self.preprocessing_config.get('output', {})
    
    def preprocess_and_visualize(self, target_split: str = "train", 
                              progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Pipeline lengkap preprocessing dengan visualisasi
        
        Args:
            target_split: Target split yang akan diproses ('train', 'valid', 'test')
            progress_callback: Fungsi callback untuk update progress (opsional)
            
        Returns:
            Dict berisi hasil preprocessing
        """
        start_time = time.time()
        self._update_progress(progress_callback, "Memulai preprocessing...", 0.0)
        
        try:
            # 1. Validasi dataset
            self._update_progress(progress_callback, "Memvalidasi dataset...", 0.1)
            validation_result = self.validator.validate_split(target_split)
            
            if not validation_result.get('is_valid', False):
                error_msg = f"Validasi gagal: {validation_result.get('message', 'Unknown error')}"
                self.logger.error(error_msg)
                return {
                    'status': 'error',
                    'message': error_msg,
                    'validation_errors': validation_result.get('errors', [])
                }
            
            # 2. Preprocessing
            self._update_progress(progress_callback, "Memproses dataset...", 0.3)
            preprocessing_result = self.engine.preprocess_split(
                target_split, 
                progress_callback=progress_callback
            )
            
            # 3. Hasil akhir
            processing_time = time.time() - start_time
            self.logger.info(f"Preprocessing selesai dalam {processing_time:.2f} detik")
            
            return {
                'status': 'success',
                'processing_time': processing_time,
                'target_split': target_split,
                'preprocessing_result': preprocessing_result,
                'validation_summary': validation_result.get('summary', {})
            }
            
        except Exception as e:
            import traceback
            error_traceback = '\n'.join(traceback.format_exception(type(e), e, e.__traceback__))
            self.logger.error(f"Error dalam preprocessing: {str(e)}\n{error_traceback}")
            return {
                'status': 'error',
                'message': str(e),
                'target_split': target_split
            }
    
    def get_sampling(self, target_split: str = "train", max_samples: int = 5) -> Dict[str, Any]:
        """Ambil sampel acak untuk evaluasi
        
        Args:
            target_split: Target split yang akan diambil sampelnya
            max_samples: Jumlah maksimal sampel yang diambil
            
        Returns:
            Dict berisi sampel data
        """
        try:
            # Dapatkan daftar file gambar
            split_dir = self.path_resolver.get_split_dir(target_split)
            image_files = self.file_scanner.scan_directory(
                split_dir / 'images', 
                extensions=['.jpg', '.jpeg', '.png']
            )
            
            # Ambil sampel acak
            selected_files = random.sample(image_files, min(max_samples, len(image_files)))
            
            # Proses setiap sampel
            samples = []
            for img_path in selected_files:
                try:
                    # Baca gambar
                    img = Image.open(img_path)
                    
                    # Dapatkan path label yang sesuai
                    label_path = self.path_resolver.get_label_path(img_path)
                    
                    samples.append({
                        'filename': img_path.name,
                        'image': np.array(img),
                        'image_path': str(img_path),
                        'label_path': str(label_path) if label_path.exists() else None
                    })
                except Exception as e:
                    self.logger.warning(f"Gagal memproses {img_path}: {str(e)}")
            
            return {
                'status': 'success',
                'samples': samples,
                'total_samples': len(samples),
                'target_split': target_split
            }
            
        except Exception as e:
            self.logger.error(f"Error dalam pengambilan sampel: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'target_split': target_split
            }
    
    def validate_dataset_only(self, target_split: str = "train") -> Dict[str, Any]:
        """Validasi dataset tanpa melakukan preprocessing
        
        Args:
            target_split: Target split yang akan divalidasi
            
        Returns:
            Dict berisi hasil validasi
        """
        try:
            result = self.validator.validate_split(target_split)
            return {
                'status': 'success' if result.get('is_valid', False) else 'failed',
                'target_split': target_split,
                'validation_result': result,
                'summary': result.get('summary', {})
            }
        except Exception as e:
            self.logger.error(f"Error dalam validasi dataset: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'target_split': target_split
            }
    
    def cleanup_preprocessed_data(self, target_split: str = None) -> Dict[str, Any]:
        """Hapus file-file hasil preprocessing
        
        Args:
            target_split: Target split yang akan dibersihkan (None untuk semua split)
            
        Returns:
            Dict berisi status cleanup
        """
        try:
            from smartcash.dataset.preprocessor.utils import create_cleanup_manager
            
            cleanup_mgr = create_cleanup_manager(self.config, self.progress_bridge)
            return cleanup_mgr.cleanup_data(target='preprocessed', target_split=target_split)
            
        except Exception as e:
            self.logger.error(f"Error dalam membersihkan data: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'target_split': target_split or 'all'
            }
    
    def get_preprocessing_status(self) -> Dict[str, Any]:
        """Dapatkan status preprocessing
        
        Returns:
            Dict berisi status preprocessing
        """
        try:
            # Implementasi sederhana, bisa dikembangkan lebih lanjut
            return {
                'status': 'success',
                'service_ready': True,
                'config': self.config,
                'message': 'Service preprocessing berjalan dengan baik'
            }
        except Exception as e:
            return {
                'status': 'error',
                'service_ready': False,
                'message': f'Error: {str(e)}'
            }
    
    def _update_progress(self, callback: Optional[Callable], message: str, progress: float):
        """Update progress dengan dukungan untuk progress bridge.
        
        Args:
            callback: Fungsi callback untuk update progress
            message: Pesan progress
            progress: Nilai progress (0.0 - 1.0)
        """
        if callback and callable(callback):
            try:
                callback(progress, message)
            except Exception as e:
                self.logger.warning(f"Error dalam progress callback: {str(e)}")
        
        # Update progress bridge jika ada
        if self.progress_bridge:
            self.progress_bridge.update(progress, message)


def create_preprocessing_service(config: Dict[str, Any] = None, progress_tracker=None):
    """Factory function untuk membuat instance PreprocessingService.
    
    Args:
        config: Konfigurasi preprocessing (opsional)
        progress_tracker: Objek untuk melacak progress (opsional)
        
    Returns:
        Instance PreprocessingService
    """
    return PreprocessingService(config=config, progress_tracker=progress_tracker)
