"""
File: smartcash/dataset/augmentor/core/normalizer.py
Deskripsi: Engine normalisasi dengan resize 640x640 dan berbagai metode normalisasi
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.dataset.augmentor.utils.file_scanner import FileScanner
from smartcash.common.logger import get_logger
from smartcash.dataset.augmentor.utils.config_validator import validate_augmentation_config, get_default_augmentation_config

class NormalizationEngine:
    """🔧 Engine normalisasi dengan resize otomatis dan metode yang dapat dikonfigurasi"""
    
    def __init__(self, config: Dict[str, Any] = None, progress_bridge=None):
        self.logger = get_logger(__name__)
        # Config validation
        if config is None:
            self.logger.warning("⚠️ No config provided, menggunakan defaults")
            self.config = get_default_augmentation_config()
        else:
            self.config = validate_augmentation_config(config)
        
        # Extract normalization config dengan safe defaults
        self.norm_config = self.config.get('preprocessing', {}).get('normalization', {})
        self.target_size = tuple(self.norm_config.get('target_size', [640, 640]))
        self.method = self.norm_config.get('method', 'minmax')
        self.denormalize = self.norm_config.get('denormalize', False)
        self.progress = progress_bridge
        
        # File scanner
        self.file_scanner = FileScanner()
        
    def normalize_augmented_files(self, aug_path: str, output_path: str, 
                                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """🎯 Normalize augmented files dengan resize dan progress tracking"""
        try:
            self._report_progress("overall", 5, 100, "📊 Scanning augmented files", progress_callback)
            
            # Scan augmented files
            aug_files = self.file_scanner.scan_augmented_files(aug_path)
            if not aug_files:
                return {'status': 'success', 'message': 'No augmented files to normalize', 'total_normalized': 0}
            
            self._report_progress("overall", 15, 100, f"🔧 Processing {len(aug_files)} files", progress_callback)
            
            # Setup output directory
            self._setup_output_directory(output_path)
            
            # Execute normalization dengan threading
            result = self._execute_normalization(aug_files, output_path, progress_callback)
            
            return result
            
        except Exception as e:
            error_msg = f"🚨 Normalization error: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg, 'total_normalized': 0}
    
    def _execute_normalization(self, files: List[str], output_path: str, 
                             progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute normalization dengan threading"""
        total_files = len(files)
        processed_files = 0
        normalized_count = 0
        
        def process_file(file_path: str) -> Dict[str, Any]:
            """Process single file normalization"""
            try:
                return self._normalize_single_file(file_path, output_path)
            except Exception as e:
                return {'status': 'error', 'file': file_path, 'error': str(e)}
        
        # Execute dengan ThreadPoolExecutor
        max_workers = min(4, len(files))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_file, file_path): file_path for file_path in files}
            
            for future in as_completed(future_to_file):
                result = future.result()
                processed_files += 1
                
                if result['status'] == 'success':
                    normalized_count += 1
                else:
                    self.logger.warning(f"⚠️ Error normalizing {result['file']}: {result.get('error', 'Unknown')}")
                
                # Update progress (15-95% range)
                progress_pct = 15 + int((processed_files / total_files) * 80)
                self._report_progress("overall", progress_pct, 100, 
                                    f"🔧 Normalized {processed_files}/{total_files} files", progress_callback)
        
        self.logger.success(f"✅ Normalization complete: {normalized_count}/{total_files} files processed")
        
        return {
            'status': 'success',
            'total_normalized': normalized_count,
            'processed_files': processed_files,
            'output_path': output_path
        }
    
    def _normalize_single_file(self, file_path: str, output_path: str) -> Dict[str, Any]:
        """Normalize single file dengan resize dan method yang dikonfigurasi"""
        try:
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                return {'status': 'error', 'file': file_path, 'error': 'Cannot load image'}
            
            # Resize ke 640x640 (step pertama)
            resized_image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Apply normalization method
            normalized_image = self._apply_normalization_method(resized_image)
            
            # Generate output filename (keep original augmented name)
            input_name = Path(file_path).stem
            output_file = Path(output_path) / 'images' / f"{input_name}.jpg"
            
            # Save normalized image
            if self.denormalize:
                # Denormalize back untuk save (convert to uint8)
                save_image = self._denormalize_for_save(normalized_image)
            else:
                # Save as float32 normalized untuk training
                save_image = normalized_image
            
            # Save dengan appropriate format
            if self._save_normalized_image(save_image, output_file):
                # Copy corresponding label
                self._copy_corresponding_label(file_path, output_path, input_name)
                return {'status': 'success', 'file': file_path, 'output': str(output_file)}
            else:
                return {'status': 'error', 'file': file_path, 'error': 'Failed to save normalized image'}
                
        except Exception as e:
            return {'status': 'error', 'file': file_path, 'error': str(e)}
    
    def _apply_normalization_method(self, image: np.ndarray) -> np.ndarray:
        """Apply normalization method berdasarkan config"""
        # Convert ke float32 untuk precision
        normalized = image.astype(np.float32)
        
        if self.method == 'minmax':
            # Min-Max normalization ke [0, 1]
            normalized = normalized / 255.0
            
        elif self.method == 'standard':
            # Standardization (Z-score)
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
            # No normalization, just convert to float32
            pass
            
        else:
            # Default ke minmax
            normalized = normalized / 255.0
            
        return normalized
    
    def _denormalize_for_save(self, normalized_image: np.ndarray) -> np.ndarray:
        """Denormalize image untuk save sebagai uint8"""
        if self.method == 'minmax':
            # Convert dari [0, 1] ke [0, 255]
            denorm = normalized_image * 255.0
            
        elif self.method == 'standard':
            # Convert dari z-score ke [0, 255] (approximate)
            denorm = np.clip(normalized_image * 64 + 128, 0, 255)
            
        elif self.method == 'imagenet':
            # Reverse ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            denorm = (normalized_image * std + mean) * 255.0
            
        else:
            # Default handling
            denorm = np.clip(normalized_image, 0, 255)
            
        return np.clip(denorm, 0, 255).astype(np.uint8)
    
    def _save_normalized_image(self, image: np.ndarray, output_path: Path) -> bool:
        """Save normalized image dengan format yang sesuai"""
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.denormalize:
                # Save sebagai regular JPEG (uint8)
                save_params = [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
                return cv2.imwrite(str(output_path), image, save_params)
            else:
                # Save sebagai normalized float32 untuk training (menggunakan numpy)
                npy_path = output_path.with_suffix('.npy')
                np.save(npy_path, image)
                
                # Juga save JPEG version untuk visualization
                jpeg_image = self._denormalize_for_save(image)
                save_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
                return cv2.imwrite(str(output_path), jpeg_image, save_params)
                
        except Exception as e:
            self.logger.error(f"🚨 Error saving image {output_path}: {str(e)}")
            return False
    
    def _copy_corresponding_label(self, source_image_path: str, output_path: str, filename: str):
        """Copy corresponding label file"""
        try:
            # Find source label
            source_dir = Path(source_image_path).parent.parent / 'labels'
            source_label = source_dir / f"{Path(source_image_path).stem}.txt"
            
            if source_label.exists():
                # Copy ke output labels directory
                output_labels_dir = Path(output_path) / 'labels'
                output_labels_dir.mkdir(parents=True, exist_ok=True)
                
                output_label = output_labels_dir / f"{filename}.txt"
                
                import shutil
                shutil.copy2(source_label, output_label)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error copying label for {filename}: {str(e)}")
    
    def _setup_output_directory(self, output_path: str):
        """Setup output directory structure"""
        output_dir = Path(output_path)
        (output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
    def _report_progress(self, level: str, current: int, total: int, message: str, callback: Optional[Callable]):
        """Report progress dengan dual compatibility"""
        # Report ke progress bridge
        if self.progress:
            self.progress.update(level, current, total, message)
        
        # Report ke callback
        if callback and callable(callback):
            try:
                callback(level, current, total, message)
            except Exception:
                pass


# Utility functions
def create_normalization_engine(config: Dict[str, Any], progress_bridge=None) -> NormalizationEngine:
    """🏭 Factory untuk create normalization engine"""
    return NormalizationEngine(config, progress_bridge)

def normalize_augmented_dataset(config: Dict[str, Any], aug_path: str, output_path: str,
                              progress_tracker=None, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """🚀 One-liner untuk normalize augmented dataset"""
    engine = create_normalization_engine(config, progress_tracker)
    return engine.normalize_augmented_files(aug_path, output_path, progress_callback)