"""
File: smartcash/dataset/augmentor/core/normalizer.py
Deskripsi: Enhanced normalizer dengan auto directory creation dan improved summary
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
    """🔧 Enhanced engine dengan auto directory creation dan improved summary"""
    
    def __init__(self, config: Dict[str, Any] = None, progress_bridge=None):
        self.logger = get_logger(__name__)
        if config is None:
            self.config = get_default_augmentation_config()
        else:
            self.config = validate_augmentation_config(config)
        
        self.norm_config = self.config.get('preprocessing', {}).get('normalization', {})
        self.target_size = tuple(self.norm_config.get('target_size', [640, 640]))
        self.method = self.norm_config.get('method', 'minmax')
        self.progress = progress_bridge
        self.file_scanner = FileScanner()
        
    def normalize_augmented_files(self, aug_path: str, output_path: str, 
                                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """🎯 Enhanced normalize dengan auto directory setup dan improved summary"""
        try:
            self._report_progress("overall", 0, 4, "Setup directory dan scan files", progress_callback)
            
            # Phase 0: Auto-create directories (NEW)
            self._ensure_preprocessed_directories(output_path)
            
            # Phase 1: Scan files
            aug_files = self.file_scanner.scan_augmented_files(aug_path)
            if not aug_files:
                return {
                    'status': 'success', 
                    'message': 'Tidak ada file augmented untuk dinormalisasi', 
                    'total_normalized': 0,
                    'summary': {
                        'found_files': 0,
                        'directories_created': True,
                        'output_path': output_path
                    }
                }
            
            self._report_progress("overall", 1, 4, f"Normalizing {len(aug_files)} files", progress_callback)
            
            # Phase 2: Normalization
            result = self._execute_normalization(aug_files, output_path, progress_callback)
            
            # Phase 3: Enhanced summary (NEW)
            if result.get('status') == 'success':
                enhanced_summary = self._create_summary(result, aug_files, output_path)
                result.update(enhanced_summary)
            
            self._report_progress("overall", 4, 4, "Normalisasi selesai", progress_callback)
            
            return result
            
        except Exception as e:
            error_msg = f"Error normalisasi: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg, 'total_normalized': 0}
    
    def _ensure_preprocessed_directories(self, output_path: str):
        """🏗️ NEW: Auto-create preprocessed directories"""
        output_dir = Path(output_path)
        
        # Directories yang perlu dibuat
        directories = [
            output_dir / 'images',
            output_dir / 'labels'
        ]
        
        created_dirs = []
        for dir_path in directories:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(dir_path))
        
        if created_dirs:
            self.logger.info(f"📁 Created preprocessed directories: {', '.join([d.split('/')[-1] for d in created_dirs])}")
    
    def _create_summary(self, result: Dict[str, Any], source_files: List[str], output_path: str) -> Dict[str, Any]:
        """📊 NEW: Create detailed summary untuk UI logging"""
        total_normalized = result.get('total_normalized', 0)
        processed_files = result.get('processed_files', 0)
        
        # Calculate rates
        success_rate = (total_normalized / len(source_files)) * 100 if source_files else 0
        
        # Enhanced summary
        summary = {
            'summary': {
                'input': {
                    'augmented_files': len(source_files),
                    'processed_files': processed_files
                },
                'output': {
                    'total_normalized': total_normalized,
                    'success_rate': f"{success_rate:.1f}%",
                    'npy_files_created': total_normalized,
                    'labels_copied': total_normalized
                },
                'configuration': {
                    'method': self.method,
                    'target_size': f"{self.target_size[0]}x{self.target_size[1]}",
                    'output_format': 'float32 .npy'
                },
                'directories': {
                    'output_path': output_path,
                    'images_dir': f"{output_path}/images",
                    'labels_dir': f"{output_path}/labels"
                }
            }
        }
        
        # Log enhanced summary
        self._log_detailed_summary(summary['summary'])
        
        return summary
    
    def _log_detailed_summary(self, summary: Dict[str, Any]):
        """📋 NEW: Log detailed summary untuk UI"""
        input_info = summary['input']
        output_info = summary['output']
        config_info = summary['configuration']
        
        self.logger.success(f"✅ Normalisasi berhasil!")
        self.logger.info(f"📊 Input: {input_info['processed_files']} augmented files processed")
        self.logger.info(f"🎯 Output: {output_info['total_normalized']} .npy files ({output_info['success_rate']} success)")
        self.logger.info(f"⚙️ Config: {config_info['method']} @ {config_info['target_size']} → {config_info['output_format']}")
        self.logger.info(f"📁 Saved to: {summary['directories']['output_path']}")
    
    def _execute_normalization(self, files: List[str], output_path: str, 
                             progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute normalization dengan enhanced progress tracking"""
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
                    self.logger.warning(f"⚠️ Error normalisasi {result['file']}: {result.get('error', 'Unknown')}")
                
                # Enhanced progress reporting
                progress_pct = (processed_files / total_files) * 100
                self._report_progress("current", processed_files, total_files, 
                                    f"Normalized {normalized_count} files ({progress_pct:.1f}%)", progress_callback)
        
        return {
            'status': 'success',
            'total_normalized': normalized_count,
            'processed_files': processed_files,
            'output_path': output_path
        }
    
    def _normalize_single_file(self, file_path: str, output_path: str) -> Dict[str, Any]:
        """Normalize single file dengan detailed tracking"""
        try:
            # Load dan resize
            image = cv2.imread(file_path)
            if image is None:
                return {'status': 'error', 'file': file_path, 'error': 'Tidak dapat membaca gambar'}
            
            resized_image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            normalized_image = self._apply_normalization_method(resized_image)
            
            input_name = Path(file_path).stem
            
            # Save .npy untuk training
            npy_path = Path(output_path) / 'images' / f"{input_name}.npy"
            np.save(npy_path, normalized_image.astype(np.float32))
            
            # Copy label ke preprocessed/labels
            self._copy_corresponding_label(file_path, output_path, input_name)
            
            return {'status': 'success', 'file': file_path, 'output': str(npy_path)}
                
        except Exception as e:
            return {'status': 'error', 'file': file_path, 'error': str(e)}
    
    def _apply_normalization_method(self, image: np.ndarray) -> np.ndarray:
        """Apply normalization untuk training"""
        normalized = image.astype(np.float32)
        
        if self.method == 'minmax':
            normalized = normalized / 255.0
        elif self.method == 'standard':
            mean = normalized.mean()
            std = normalized.std()
            if std > 0:
                normalized = (normalized - mean) / std
        elif self.method == 'imagenet':
            normalized = normalized / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (normalized - mean) / std
        elif self.method == 'none':
            pass
        else:
            normalized = normalized / 255.0
            
        return normalized
    
    def _copy_corresponding_label(self, source_image_path: str, output_path: str, filename: str):
        """Copy label file ke preprocessed/labels"""
        try:
            source_dir = Path(source_image_path).parent.parent / 'labels'
            source_label = source_dir / f"{Path(source_image_path).stem}.txt"
            
            if source_label.exists():
                output_labels_dir = Path(output_path) / 'labels'
                output_labels_dir.mkdir(parents=True, exist_ok=True)
                output_label = output_labels_dir / f"{filename}.txt"
                
                import shutil
                shutil.copy2(source_label, output_label)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error copying label untuk {filename}: {str(e)}")
    
    def _report_progress(self, level: str, current: int, total: int, message: str, callback: Optional[Callable]):
        """Report progress dengan enhanced messaging"""
        if self.progress:
            self.progress.update(level, current, total, message)
        
        if callback and callable(callback):
            try:
                callback(level, current, total, message)
            except Exception:
                pass


def create_normalization_engine(config: Dict[str, Any], progress_bridge=None) -> NormalizationEngine:
    """Factory untuk create normalization engine"""
    return NormalizationEngine(config, progress_bridge)

def normalize_augmented_dataset(config: Dict[str, Any], aug_path: str, output_path: str,
                              progress_tracker=None, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """One-liner untuk normalize augmented dataset"""
    engine = create_normalization_engine(config, progress_tracker)
    return engine.normalize_augmented_files(aug_path, output_path, progress_callback)