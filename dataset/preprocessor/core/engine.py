"""
File: smartcash/dataset/preprocessor/core/engine.py
Deskripsi: Modul inti untuk menangani logika preprocessing dataset YOLOv5.
"""
import os
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple

import cv2
import yaml

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor.utils.file_processor import FileProcessor
from smartcash.dataset.preprocessor.utils.file_scanner import FileScanner
from smartcash.dataset.preprocessor.utils.path_resolver import PathResolver
from smartcash.dataset.preprocessor.utils.cleanup_manager import CleanupManager
from smartcash.dataset.preprocessor.validators import (
    create_image_validator,
    create_label_validator,
    create_pair_validator
)


class PreprocessingValidator:
    """Validator untuk preprocessing."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config.get('preprocessing', {})
        self.logger = logger or get_logger()
        
        # Inisialisasi validator
        self.image_validator = create_image_validator(self.config.get('validation', {}))
        self.label_validator = create_label_validator(self.config.get('validation', {}))
        self.pair_validator = create_pair_validator(self.config.get('validation', {}))


class PreprocessingEngine:
    """Engine untuk melakukan preprocessing gambar dan label YOLOv5."""
    
    def __init__(self, config: Dict[str, Any]):
        """Inisialisasi preprocessing engine.
        
        Args:
            config: Konfigurasi preprocessing
        """
        self.config = config.get('preprocessing', {})
        self.logger = get_logger()
        
        # Inisialisasi komponen
        self.validator = PreprocessingValidator(config, self.logger)
        
        # Inisialisasi utils
        self.file_processor = FileProcessor(config)
        self.file_scanner = FileScanner()
        self.path_resolver = PathResolver(config)
        
        # Inisialisasi manajer
        self.cleanup_manager = CleanupManager(config)
        
        # Inisialisasi validator
        self.image_validator = create_image_validator(self.config.get('validation', {}))
        self.label_validator = create_label_validator(self.config.get('validation', {}))
        self.pair_validator = create_pair_validator(self.config.get('validation', {}))
        
        # Ekstrak konfigurasi normalisasi
        norm_config = self.config.get('normalization', {})
        self.target_size = norm_config.get('target_size', [640, 640])
        self.preserve_aspect_ratio = norm_config.get('preserve_aspect_ratio', True)
        self.denormalize = norm_config.get('denormalize', False)
        self.convert_rgb = norm_config.get('convert_rgb', True)
        self.img_size = self.target_size[0]  # Gunakan ukuran pertama dari target_size
        self.normalize = norm_config.get('normalize', True)
        
        # Konfigurasi output
        output_config = self.config.get('output', {})
        self.create_npy = output_config.get('create_npy', True)
        self.organize_by_split = output_config.get('organize_by_split', True)
    
    def preprocess_split(self, split: str, progress_callback: Callable = None) -> Dict[str, Any]:
        """Preprocess semua gambar dan label untuk split tertentu.
        
        Args:
            split: Nama split yang akan diproses (train/val/test)
            progress_callback: Fungsi callback untuk update progress
            
        Returns:
            Dictionary berisi statistik preprocessing
        """
        self._update_progress(progress_callback, f"Memulai preprocessing untuk {split}", 0.1)
        
        try:
            # Validasi direktori sumber
            src_img_dir = self.path_resolver.get_source_image_dir(split)
            src_label_dir = self.path_resolver.get_source_label_dir(split)
            
            if not src_img_dir.exists() or not src_label_dir.exists():
                raise FileNotFoundError(
                    f"Direktori sumber tidak ditemukan: {src_img_dir} atau {src_label_dir}")
            
            # Buat direktori tujuan jika belum ada
            dst_img_dir = self.path_resolver.get_preprocessed_image_dir(split)
            dst_label_dir = self.path_resolver.get_preprocessed_label_dir(split)
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_label_dir.mkdir(parents=True, exist_ok=True)
            
            # Dapatkan daftar file
            img_files = self.file_scanner.scan_directory(src_img_dir, ['.jpg', '.jpeg', '.png'])
            
            if not img_files:
                self.logger.warning(f"Tidak ada file gambar yang ditemukan di {src_img_dir}")
                return {'status': 'skipped', 'message': 'No image files found'}
            
            # Inisialisasi statistik
            stats = {
                'total': len(img_files),
                'processed': 0,
                'skipped': 0,
                'errors': 0,
                'invalid_files': []
            }
            
            # Proses setiap file gambar
            for i, img_file in enumerate(img_files):
                try:
                    # Update progress
                    progress = 0.1 + (i / len(img_files)) * 0.8  # 10-90% untuk preprocessing
                    self._update_progress(
                        progress_callback,
                        f"Memproses {img_file.name}",
                        progress
                    )
                    
                    # Proses gambar dan label
                    result = self._process_image_and_label(img_file, dst_img_dir, dst_label_dir, split)
                    
                    if result['status'] == 'success':
                        stats['processed'] += 1
                    else:
                        stats['skipped'] += 1
                        stats['invalid_files'].append({
                            'file': str(img_file),
                            'error': result.get('error', 'Unknown error')
                        })
                    
                except Exception as e:
                    stats['errors'] += 1
                    import traceback
                    error_traceback = '\n'.join(traceback.format_exception(type(e), e, e.__traceback__))
                    self.logger.error(f"Gagal memproses {img_file}: {str(e)}\n{error_traceback}")
            
            # Update progress final
            self._update_progress(progress_callback, "Preprocessing selesai", 1.0)
            
            return {
                'status': 'completed',
                'stats': stats
            }
            
        except Exception as e:
            import traceback
            error_traceback = '\n'.join(traceback.format_exception(type(e), e, e.__traceback__))
            self.logger.error(f"Error dalam preprocessing {split}: {str(e)}\n{error_traceback}")
            raise
    
    def _process_image_and_label(self, img_file: Path, dst_img_dir: Path, 
                               dst_label_dir: Path, split: str) -> Dict[str, Any]:
        """Proses satu gambar dan label yang sesuai.
        
        Args:
            img_file: Path ke file gambar sumber
            dst_img_dir: Direktori tujuan gambar yang sudah diproses
            dst_label_dir: Direktori tujuan label yang sudah diproses
            split: Nama split (untuk logging)
            
        Returns:
            Dictionary berisi status dan hasil pemrosesan
        """
        try:
            # Baca gambar menggunakan file_processor
            img = self.file_processor.read_image(str(img_file))
            if img is None:
                return {'status': 'error', 'error': 'Gagal membaca gambar'}
            
            # Konversi ke RGB jika diperlukan
            if self.convert_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize gambar
            img = cv2.resize(img, (self.img_size, self.img_size))
            
            # Normalisasi jika diperlukan
            if self.normalize:
                img = img.astype(np.float32) / 255.0
            
            # Simpan gambar yang sudah diproses menggunakan file_processor
            dst_img_path = dst_img_dir / img_file.name
            self.file_processor.write_image(str(dst_img_path), img)
            
            # Salin file label (jika ada)
            label_file = self._get_corresponding_label_file(img_file)
            if label_file and label_file.exists():
                dst_label_path = dst_label_dir / label_file.name
                shutil.copy2(label_file, dst_label_path)
            
            return {'status': 'success'}
            
        except Exception as e:
            error_msg = f"Gagal memproses {img_file}: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'status': 'error', 'error': error_msg}
    
    def _get_corresponding_label_file(self, img_file: Path) -> Optional[Path]:
        """Dapatkan path ke file label yang sesuai dengan file gambar."""
        label_file = img_file.with_suffix('.txt')
        return label_file if label_file.exists() else None
    
    @staticmethod
    def _update_progress(callback: Optional[Callable], message: str, progress: float):
        """Update progress jika callback tersedia."""
        if callback:
            callback(progress, message)


def create_preprocessing_engine(config: Dict[str, Any]) -> PreprocessingEngine:
    """Factory function untuk membuat instance PreprocessingEngine.
    
    Args:
        config: Konfigurasi preprocessing
        
    Returns:
        Instance PreprocessingEngine yang baru dibuat
    """
    return PreprocessingEngine(config)
