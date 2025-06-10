"""
File: smartcash/dataset/preprocessor/service.py
Deskripsi: Enhanced preprocessing service dengan proper file format handling dan progress integration
"""
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple
import numpy as np
import time

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor.core.engine import PreprocessingEngine, PreprocessingValidator
from smartcash.dataset.preprocessor.utils import (
    validate_preprocessing_config, 
    get_default_preprocessing_config,
    ProgressBridge, 
    FileProcessor, 
    FileScanner, 
    PathResolver,
    CleanupManager,
    FilenameManager
)

class PreprocessingService:
    """🎯 Enhanced preprocessing service dengan proper file format handling"""
    
    def __init__(self, config: Dict[str, Any] = None, progress_callback: Optional[Callable] = None):
        # Setup logger dengan minimal output untuk avoid double logging
        self.logger = get_logger(__name__)
        # 🔑 KEY: Disable backend logger output saat ada UI callback
        self._ui_mode = progress_callback is not None
        
        # Config validation dan enhancement
        if config is None:
            if not self._ui_mode:
                self.logger.warning("⚠️ Konfigurasi tidak disediakan, menggunakan default")
            self.config = get_default_preprocessing_config()
        else:
            try:
                self.config = validate_preprocessing_config(config)
                # 🔑 KEY: Enhance dengan file naming dan path configuration
                self.config = self._enhance_config_for_processing(self.config)
            except ValueError as e:
                if not self._ui_mode:
                    self.logger.error(f"❌ Konfigurasi tidak valid: {str(e)}")
                raise ValueError(f"Konfigurasi tidak valid: {str(e)}") from e
        
        # Progress Bridge setup
        self.progress_callback = progress_callback
        self.progress_bridge = ProgressBridge()
        if progress_callback:
            self.progress_bridge.register_callback(progress_callback)
        
        # Initialize components dengan enhanced config
        self.validator = PreprocessingValidator(self.config, self.logger)
        self.engine = PreprocessingEngine(self.config)
        self.file_processor = FileProcessor(self.config)
        self.file_scanner = FileScanner()
        self.path_resolver = PathResolver(self.config)
        self.cleanup_manager = CleanupManager(self.config)
        self.filename_manager = FilenameManager(self.config.get('file_naming', {}))
        
        # Register progress callback ke engine
        if hasattr(self.engine, 'register_progress_callback'):
            self.engine.register_progress_callback(self._engine_progress_callback)
        
        # Extract config sections
        self.preprocessing_config = self.config.get('preprocessing', {})
        self.validation_config = self.preprocessing_config.get('validation', {})
        self.data_config = self.config.get('data', {})
    
    def _enhance_config_for_processing(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """🔧 Enhance config dengan file naming dan path configuration"""
        enhanced = config.copy()
        
        # Setup file naming patterns jika belum ada
        if 'file_naming' not in enhanced:
            enhanced['file_naming'] = {
                'raw_pattern': 'rp_{nominal}_{uuid}_{sequence}',
                'preprocessed_pattern': 'pre_rp_{nominal}_{uuid}_{sequence}_{variance}',
                'augmented_pattern': 'aug_rp_{nominal}_{uuid}_{sequence}_{variance}',
                'preserve_uuid': True
            }
        
        # Setup data paths dengan auto-detection
        data_section = enhanced.setdefault('data', {})
        data_section.setdefault('dir', 'data')
        
        # Setup local paths untuk splits
        target_splits = enhanced.get('preprocessing', {}).get('target_splits', ['train', 'valid'])
        local_paths = data_section.setdefault('local', {})
        
        base_dir = data_section['dir']
        for split in target_splits:
            if split not in local_paths:
                local_paths[split] = f"{base_dir}/{split}"
        
        # Setup output directory
        preprocessing_section = enhanced.setdefault('preprocessing', {})
        preprocessing_section.setdefault('output_dir', f"{base_dir}/preprocessed")
        
        return enhanced
    
    def preprocess_dataset(self) -> Dict[str, Any]:
        """🚀 Main preprocessing method dengan enhanced file format handling"""
        start_time = time.time()
        
        try:
            self._update_progress("overall", 0, 100, "🚀 Memulai preprocessing dataset")
            
            # Phase 1: Pre-validation dan file format check (0-15%)
            self._update_progress("overall", 5, 100, "🔍 Pre-validation dan format check")
            
            format_check_result = self._check_and_prepare_file_formats()
            if not format_check_result.get('success', False):
                error_msg = format_check_result.get('message', 'Format check failed')
                self._update_progress("overall", 0, 100, f"❌ {error_msg}")
                return {
                    'success': False,
                    'message': error_msg,
                    'stats': format_check_result.get('stats', {})
                }
            
            self._update_progress("overall", 15, 100, "✅ Format check selesai")
            
            # Phase 2: Validation (15-25%)
            self._update_progress("overall", 18, 100, "🔍 Memvalidasi struktur dataset")
            
            validation_result = self._comprehensive_validation()
            if not validation_result.get('success', False):
                error_msg = validation_result.get('message', 'Validation failed')
                self._update_progress("overall", 0, 100, f"❌ Validasi gagal: {error_msg}")
                return {
                    'success': False,
                    'message': error_msg,
                    'validation_errors': validation_result.get('errors', []),
                    'stats': validation_result.get('stats', {})
                }
            
            self._update_progress("overall", 25, 100, "✅ Validasi selesai")
            
            # Phase 3: Preprocessing (25-90%)
            self._update_progress("overall", 30, 100, "🔄 Memulai pipeline preprocessing")
            
            preprocessing_result = self.engine.preprocess_dataset(
                progress_callback=self._engine_progress_callback
            )
            
            if not preprocessing_result.get('success', False):
                error_msg = preprocessing_result.get('message', 'Preprocessing failed')
                self._update_progress("overall", 0, 100, f"❌ Preprocessing gagal: {error_msg}")
                return {
                    'success': False,
                    'message': error_msg,
                    'stats': preprocessing_result.get('stats', {})
                }
            
            # Phase 4: Finalization (90-100%)
            self._update_progress("overall", 95, 100, "🏁 Menyelesaikan preprocessing")
            processing_time = time.time() - start_time
            
            final_result = {
                'success': True,
                'message': "✅ Preprocessing berhasil",
                'processing_time': processing_time,
                'stats': self._compile_stats(preprocessing_result, validation_result, processing_time),
                'configuration': self._get_config_summary(),
                'file_format_info': format_check_result.get('stats', {})
            }
            
            self._update_progress("overall", 100, 100, "✅ Preprocessing selesai")
            return final_result
            
        except Exception as e:
            error_msg = f"❌ Error preprocessing: {str(e)}"
            if not self._ui_mode:
                self.logger.error(error_msg)
            self._update_progress("overall", 0, 100, error_msg)
            return {
                'success': False,
                'message': error_msg,
                'processing_time': time.time() - start_time,
                'stats': {}
            }
    
    def _check_and_prepare_file_formats(self) -> Dict[str, Any]:
        """🔧 Check dan prepare file formats untuk processing"""
        try:
            target_splits = self.preprocessing_config.get('target_splits', ['train', 'valid'])
            if isinstance(target_splits, str):
                target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
            
            format_stats = {
                'total_files': 0,
                'raw_format_files': 0,
                'needs_renaming': 0,
                'renamed_files': 0,
                'errors': []
            }
            
            for split in target_splits:
                # Get source paths
                local_paths = self.data_config.get('local', {})
                if split in local_paths:
                    base_path = Path(local_paths[split])
                else:
                    base_dir = self.data_config.get('dir', 'data')
                    base_path = Path(base_dir) / split
                
                img_dir = base_path / 'images'
                
                if not img_dir.exists():
                    continue
                
                # Scan image files
                image_files = self.file_scanner.scan_directory(img_dir, {'.jpg', '.jpeg', '.png'})
                format_stats['total_files'] += len(image_files)
                
                # Check format dan rename jika perlu
                for img_file in image_files:
                    if self._is_raw_format(img_file.name):
                        format_stats['raw_format_files'] += 1
                    else:
                        format_stats['needs_renaming'] += 1
                        # Perform renaming jika diperlukan
                        try:
                            renamed = self._rename_to_raw_format(img_file, split)
                            if renamed:
                                format_stats['renamed_files'] += 1
                        except Exception as e:
                            format_stats['errors'].append(f"Error renaming {img_file.name}: {str(e)}")
            
            success = len(format_stats['errors']) == 0
            message = f"Format check: {format_stats['raw_format_files'] + format_stats['renamed_files']}/{format_stats['total_files']} files ready"
            
            return {
                'success': success,
                'message': message,
                'stats': format_stats
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"❌ Error format check: {str(e)}",
                'stats': {'errors': [str(e)]}
            }
    
    def _is_raw_format(self, filename: str) -> bool:
        """Check jika filename sudah dalam format raw yang benar"""
        return self.filename_manager.parse_filename(filename) is not None
    
    def _rename_to_raw_format(self, img_file: Path, split: str) -> bool:
        """Rename file ke format raw jika diperlukan"""
        try:
            # Extract nominal dari filename atau path
            nominal = self._extract_nominal_from_source(img_file)
            
            # Generate new filename dalam format raw
            import uuid
            new_uuid = str(uuid.uuid4())
            sequence = 1  # Default sequence
            
            new_name = f"rp_{nominal}_{new_uuid}_{sequence:03d}"
            new_img_file = img_file.parent / f"{new_name}{img_file.suffix}"
            
            # Rename image file
            img_file.rename(new_img_file)
            
            # Rename corresponding label file jika ada
            label_dir = img_file.parent.parent / 'labels'
            old_label = label_dir / f"{img_file.stem}.txt"
            if old_label.exists():
                new_label = label_dir / f"{new_name}.txt"
                old_label.rename(new_label)
            
            return True
            
        except Exception as e:
            if not self._ui_mode:
                self.logger.error(f"Error renaming {img_file}: {str(e)}")
            return False
    
    def _extract_nominal_from_source(self, img_file: Path) -> str:
        """Extract nominal dari source filename atau directory structure"""
        # Try dari filename
        filename_lower = img_file.name.lower()
        nominal_patterns = [
            (r'1000|rp1000', '001000'),
            (r'2000|rp2000', '002000'), 
            (r'5000|rp5000', '005000'),
            (r'10000|rp10000', '010000'),
            (r'20000|rp20000', '020000'),
            (r'50000|rp50000', '050000'),
            (r'100000|rp100000', '100000')
        ]
        
        import re
        for pattern, nominal in nominal_patterns:
            if re.search(pattern, filename_lower):
                return nominal
        
        # Try dari directory path
        path_str = str(img_file.parent).lower()
        for pattern, nominal in nominal_patterns:
            if re.search(pattern, path_str):
                return nominal
        
        # Default fallback
        return '000000'
    
    def check_preprocessed_exists(self, target_split: str = None) -> Tuple[bool, int]:
        """📊 Check preprocessed data existence tanpa logging"""
        try:
            output_dir = Path(self.preprocessing_config.get('output_dir', 'data/preprocessed'))
            
            if not output_dir.exists():
                return False, 0
            
            file_count = self._count_preprocessed_files(output_dir, target_split)
            return file_count > 0, file_count
            
        except Exception:
            return False, 0
    
    def _count_preprocessed_files(self, output_dir: Path, target_split: str = None) -> int:
        """📊 Count files in preprocessed directory"""
        total_files = 0
        
        splits = [target_split] if target_split else self.preprocessing_config.get('target_splits', ['train', 'valid'])
        if isinstance(splits, str):
            splits = [splits] if splits != 'all' else ['train', 'valid', 'test']
        
        for split in splits:
            split_dir = output_dir / split
            if split_dir.exists():
                # Count .npy files in images directory
                images_dir = split_dir / 'images'
                if images_dir.exists():
                    npy_files = list(images_dir.glob('pre_*.npy'))
                    total_files += len(npy_files)
                
                # Count .txt files in labels directory
                labels_dir = split_dir / 'labels'
                if labels_dir.exists():
                    txt_files = list(labels_dir.glob('pre_*.txt'))
                    total_files += len(txt_files)
        
        return total_files
    
    def cleanup_preprocessed_data(self, target_split: str = None) -> Dict[str, Any]:
        """🧹 Enhanced cleanup method tanpa excessive logging"""
        try:
            output_dir = Path(self.preprocessing_config.get('output_dir', 'data/preprocessed'))
            
            if not output_dir.exists():
                return {
                    'success': True,
                    'message': "ℹ️ Tidak ada data untuk dibersihkan",
                    'stats': {'files_removed': 0}
                }
            
            # Count files sebelum cleanup
            files_count = self._count_preprocessed_files(output_dir, target_split)
            
            if files_count == 0:
                return {
                    'success': True,
                    'message': "ℹ️ Tidak ada file untuk dibersihkan",
                    'stats': {'files_removed': 0}
                }
            
            # Update progress untuk cleanup start
            self._update_progress("current", 25, 100, f"Menghapus {files_count} files...")
            
            # Perform cleanup
            cleanup_stats = self._perform_cleanup(output_dir, target_split)
            
            # Update progress untuk completion
            self._update_progress("current", 100, 100, "Cleanup selesai")
            
            return {
                'success': True,
                'message': f"🧹 Cleanup berhasil: {cleanup_stats.get('files_removed', 0)} file dihapus",
                'stats': cleanup_stats,
                'files_count': files_count
            }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"❌ Error cleanup: {str(e)}"
            }
    
    def _perform_cleanup(self, output_dir: Path, target_split: str = None) -> Dict[str, int]:
        """🧹 Internal cleanup logic dengan progress updates"""
        stats = {'files_removed': 0, 'dirs_removed': 0, 'errors': 0}
        
        splits = [target_split] if target_split else self.preprocessing_config.get('target_splits', ['train', 'valid'])
        if isinstance(splits, str):
            splits = [splits] if splits != 'all' else ['train', 'valid', 'test']
        
        total_splits = len(splits)
        for i, split in enumerate(splits):
            split_dir = output_dir / split
            if split_dir.exists():
                try:
                    # Update progress per split
                    progress = 25 + int((i / total_splits) * 50)  # 25-75%
                    self._update_progress("current", progress, 100, f"Menghapus {split} split...")
                    
                    import shutil
                    files_in_split = sum(1 for _ in split_dir.rglob('*.*'))
                    shutil.rmtree(split_dir)
                    stats['files_removed'] += files_in_split
                    stats['dirs_removed'] += 1
                except Exception as e:
                    if not self._ui_mode:
                        self.logger.error(f"Error removing {split_dir}: {str(e)}")
                    stats['errors'] += 1
        
        return stats
    
    def validate_dataset_only(self, target_split: str = None) -> Dict[str, Any]:
        """🔍 Backend validation method tanpa excessive logging"""
        try:
            if target_split:
                result = self.validator.validate_split(target_split)
                return {
                    'success': result.get('is_valid', False),
                    'message': result.get('message', 'Validation completed'),
                    'target_split': target_split,
                    'summary': result.get('summary', {})
                }
            else:
                # Validate all target splits
                target_splits = self.preprocessing_config.get('target_splits', ['train', 'valid'])
                if isinstance(target_splits, str):
                    target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
                
                all_valid = True
                total_images = 0
                
                for split in target_splits:
                    split_result = self.validator.validate_split(split)
                    if not split_result.get('is_valid'):
                        all_valid = False
                    total_images += split_result.get('summary', {}).get('valid_images', 0)
                
                return {
                    'success': all_valid,
                    'message': f"Validation {'berhasil' if all_valid else 'partial'} untuk {len(target_splits)} splits",
                    'summary': {'total_valid_images': total_images, 'splits_validated': len(target_splits)}
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"❌ Error validasi: {str(e)}",
                'summary': {}
            }
    
    def _comprehensive_validation(self) -> Dict[str, Any]:
        """🔍 Internal validation logic tanpa excessive logging"""
        try:
            target_splits = self.preprocessing_config.get('target_splits', ['train', 'valid'])
            if isinstance(target_splits, str):
                target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
            
            validation_results = {}
            total_valid_images = 0
            all_errors = []
            
            for split in target_splits:
                try:
                    result = self.validator.validate_split(split)
                    validation_results[split] = result
                    
                    if result.get('is_valid', False):
                        total_valid_images += result.get('summary', {}).get('valid_images', 0)
                    else:
                        all_errors.extend(result.get('summary', {}).get('validation_errors', []))
                except Exception as e:
                    error_msg = f"Error validasi {split}: {str(e)}"
                    all_errors.append(error_msg)
                    validation_results[split] = {'is_valid': False, 'error': error_msg}
            
            all_valid = all(result.get('is_valid', False) for result in validation_results.values())
            
            return {
                'success': all_valid,
                'message': f"Validation {'berhasil' if all_valid else 'partial'} untuk {len(target_splits)} splits",
                'stats': {
                    'splits_validated': len(target_splits),
                    'total_valid_images': total_valid_images,
                    'validation_results': validation_results
                },
                'errors': all_errors[:10]
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"❌ Validation error: {str(e)}",
                'stats': {},
                'errors': [str(e)]
            }
    
    def _compile_stats(self, preprocessing_result: Dict, validation_result: Dict, processing_time: float) -> Dict[str, Any]:
        """📊 Compile processing statistics"""
        preprocessing_stats = preprocessing_result.get('stats', {})
        validation_stats = validation_result.get('stats', {})
        
        return {
            'processing_time': round(processing_time, 2),
            'input': {
                'splits_processed': len(preprocessing_stats.get('target_splits', [])),
                'total_input_images': validation_stats.get('total_valid_images', 0)
            },
            'output': {
                'total_processed': preprocessing_stats.get('total_processed', 0),
                'total_normalized': preprocessing_stats.get('total_normalized', 0),
                'total_errors': preprocessing_stats.get('total_errors', 0),
                'success_rate': f"{preprocessing_stats.get('success_rate', 0):.1f}%"
            },
            'performance': {
                'processing_time_seconds': round(processing_time, 2),
                'avg_time_per_image': round(processing_time / max(preprocessing_stats.get('total_processed', 1), 1), 3),
                'images_per_second': round(preprocessing_stats.get('total_processed', 0) / max(processing_time, 0.1), 2)
            }
        }
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """⚙️ Get configuration summary"""
        norm_config = self.preprocessing_config.get('normalization', {})
        return {
            'target_splits': self.preprocessing_config.get('target_splits', ['train', 'valid']),
            'normalization': {
                'enabled': norm_config.get('enabled', True),
                'method': norm_config.get('method', 'minmax'),
                'target_size': norm_config.get('target_size', [640, 640])
            },
            'output_dir': self.preprocessing_config.get('output_dir', 'data/preprocessed'),
            'file_formats': {
                'input': 'rp_{nominal}_{uuid}_{sequence}.ext',
                'output': 'pre_rp_{nominal}_{uuid}_{sequence}_{variance}.ext'
            }
        }
    
    def _engine_progress_callback(self, level: str, current: int, total: int, message: str):
        """🔧 Engine progress callback yang meneruskan ke UI callback TANPA logging backend"""
        # Map engine progress ke overall progress (25-90%)
        if level in ['overall', 'step']:
            engine_progress = (current / total) if total > 0 else 0
            overall_progress = int(25 + (engine_progress * 65))  # 25% to 90%
            self._update_progress("overall", overall_progress, 100, message)
        
        # Forward current operation progress
        self._update_progress("current", current, total, message)
    
    def _update_progress(self, level: str, current: int, total: int, message: str):
        """📊 Update progress via Progress Bridge TANPA backend logging"""
        try:
            if self.progress_bridge:
                self.progress_bridge.update(level, current, total, message)
        except Exception as e:
            if not self._ui_mode:
                self.logger.debug(f"Progress callback error: {str(e)}")

def create_preprocessing_service(config: Dict[str, Any] = None, 
                               progress_callback: Optional[Callable] = None) -> PreprocessingService:
    """🏭 Factory untuk enhanced preprocessing service"""
    return PreprocessingService(config, progress_callback)