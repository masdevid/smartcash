"""
File: smartcash/dataset/augmentor/core/normalizer.py
Deskripsi: Normalizer dengan .npy files untuk training dan clean progress tracking
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.dataset.augmentor.utils.file_scanner import FileScanner
from smartcash.common.logger import get_logger
from smartcash.dataset.augmentor.utils.config_validator import validate_augmentation_config, get_default_augmentation_config

class NormalizationEngine:
    """🔧 Engine normalisasi dengan .npy files untuk training YOLO"""
    
    def __init__(self, config: Dict[str, Any] = None, progress_bridge=None):
        self.logger = get_logger(__name__)
        if config is None:
            self.config = get_default_augmentation_config()
        else:
            self.config = validate_augmentation_config(config)
        
        # Extract config
        self.norm_config = self.config.get('preprocessing', {}).get('normalization', {})
        self.target_size = tuple(self.norm_config.get('target_size', [640, 640]))
        self.method = self.norm_config.get('method', 'minmax')
        self.progress = progress_bridge
        
        # File scanner
        self.file_scanner = FileScanner()
        
    def normalize_augmented_files(self, aug_path: str, output_path: str, 
                                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """🎯 Normalize augmented files dengan .npy untuk training"""
        try:
            self._report_progress("overall", 0, 3, "Memulai normalisasi", progress_callback)
            
            # Phase 1: Scan files
            aug_files = self.file_scanner.scan_augmented_files(aug_path)
            if not aug_files:
                return {'status': 'success', 'message': 'Tidak ada file augmented untuk dinormalisasi', 'total_normalized': 0}
            
            self._report_progress("overall", 1, 3, "Memproses normalisasi", progress_callback)
            
            # Phase 2: Setup output
            self._setup_output_directory(output_path)
            
            # Phase 3: Execute normalization
            result = self._execute_normalization(aug_files, output_path, progress_callback)
            
            self._report_progress("overall", 3, 3, "Normalisasi selesai", progress_callback)
            
            return result
            
        except Exception as e:
            error_msg = f"Error normalisasi: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg, 'total_normalized': 0}
    
    def _execute_normalization(self, files: List[str], output_path: str, 
                             progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute normalization dengan clean progress"""
        total_files = len(files)
        processed_files = 0
        normalized_count = 0
        
        def process_file(file_path: str) -> Dict[str, Any]:
            try:
                return self._normalize_single_file(file_path, output_path)
            except Exception as e:
                return {'status': 'error', 'file': file_path, 'error': str(e)}
        
        max_workers = min(4, len(files))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_file, file_path): file_path for file_path in files}
            
            for future in as_completed(future_to_file):
                result = future.result()
                processed_files += 1
                
                if result['status'] == 'success':
                    normalized_count += 1
                else:
                    self.logger.warning(f"Error normalisasi {result['file']}: {result.get('error', 'Unknown')}")
                
                # Update current progress
                self._report_progress("current", processed_files, total_files, 
                                    f"Dinormalisasi: {processed_files}/{total_files} file", progress_callback)
        
        self.logger.success(f"Normalisasi selesai: {normalized_count}/{total_files} file")
        
        return {
            'status': 'success',
            'total_normalized': normalized_count,
            'processed_files': processed_files,
            'output_path': output_path
        }
    
    def _normalize_single_file(self, file_path: str, output_path: str) -> Dict[str, Any]:
        """Normalize single file dengan .npy untuk training"""
        try:
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                return {'status': 'error', 'file': file_path, 'error': 'Tidak dapat membaca gambar'}
            
            # Resize ke 640x640 untuk YOLO
            resized_image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Apply normalization
            normalized_image = self._apply_normalization_method(resized_image)
            
            # Generate output filename
            input_name = Path(file_path).stem
            
            # Save normalized image sebagai .npy untuk training
            npy_path = Path(output_path) / 'images' / f"{input_name}.npy"
            np.save(npy_path, normalized_image.astype(np.float32))
            
            # JUGA save sebagai .jpg untuk visualisasi/debugging
            jpg_path = Path(output_path) / 'images' / f"{input_name}.jpg"
            visual_image = self._denormalize_for_visualization(normalized_image)
            cv2.imwrite(str(jpg_path), visual_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            # Copy corresponding label
            self._copy_corresponding_label(file_path, output_path, input_name)
            
            return {'status': 'success', 'file': file_path, 'output': str(npy_path)}
                
        except Exception as e:
            return {'status': 'error', 'file': file_path, 'error': str(e)}
    
    def _apply_normalization_method(self, image: np.ndarray) -> np.ndarray:
        """Apply normalization method untuk training"""
        normalized = image.astype(np.float32)
        
        if self.method == 'minmax':
            # Min-Max [0, 1] untuk YOLO training
            normalized = normalized / 255.0
            
        elif self.method == 'standard':
            # Z-score normalization
            mean = normalized.mean()
            std = normalized.std()
            if std > 0:
                normalized = (normalized - mean) / std
                
        elif self.method == 'imagenet':
            # ImageNet normalization
            normalized = normalized / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (normalized - mean) / std
            
        elif self.method == 'none':
            # No normalization (tetap float32)
            pass
            
        else:
            # Default ke minmax
            normalized = normalized / 255.0
            
        return normalized
    
    def _denormalize_for_visualization(self, normalized_image: np.ndarray) -> np.ndarray:
        """Denormalize untuk visualisasi .jpg files"""
        if self.method == 'minmax':
            denorm = normalized_image * 255.0
        elif self.method == 'standard':
            denorm = np.clip(normalized_image * 64 + 128, 0, 255)
        elif self.method == 'imagenet':
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            denorm = (normalized_image * std + mean) * 255.0
        else:
            denorm = np.clip(normalized_image, 0, 255)
            
        return np.clip(denorm, 0, 255).astype(np.uint8)
    
    def _copy_corresponding_label(self, source_image_path: str, output_path: str, filename: str):
        """Copy label file ke preprocessed/labels"""
        try:
            # Find source label
            source_dir = Path(source_image_path).parent.parent / 'labels'
            source_label = source_dir / f"{Path(source_image_path).stem}.txt"
            
            if source_label.exists():
                # Copy ke output labels directory (BUKAN images directory)
                output_labels_dir = Path(output_path) / 'labels'
                output_labels_dir.mkdir(parents=True, exist_ok=True)
                
                output_label = output_labels_dir / f"{filename}.txt"
                
                import shutil
                shutil.copy2(source_label, output_label)
                
        except Exception as e:
            self.logger.warning(f"Error copying label untuk {filename}: {str(e)}")
    
    def _setup_output_directory(self, output_path: str):
        """Setup output directory structure untuk preprocessed"""
        output_dir = Path(output_path)
        (output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
    def _report_progress(self, level: str, current: int, total: int, message: str, callback: Optional[Callable]):
        """Report progress dengan clean format"""
        if self.progress:
            self.progress.update(level, current, total, message)
        
        if callback and callable(callback):
            try:
                callback(level, current, total, message)
            except Exception:
                pass


# Utility functions
def create_normalization_engine(config: Dict[str, Any], progress_bridge=None) -> NormalizationEngine:
    """Factory untuk create normalization engine"""
    return NormalizationEngine(config, progress_bridge)

def normalize_augmented_dataset(config: Dict[str, Any], aug_path: str, output_path: str,
                              progress_tracker=None, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """One-liner untuk normalize augmented dataset"""
    engine = create_normalization_engine(config, progress_tracker)
    return engine.normalize_augmented_files(aug_path, output_path, progress_callback)