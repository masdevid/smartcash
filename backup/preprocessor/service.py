"""
File: smartcash/dataset/preprocessor/service.py
Deskripsi: Simplified preprocessing service menggunakan consolidated utils
"""
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple
import time

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor.core.engine import PreprocessingEngine, PreprocessingValidator
from smartcash.dataset.preprocessor.utils import (
    validate_preprocessing_config, get_default_preprocessing_config,
    create_progress_bridge, create_path_manager, create_file_operations,
    create_metadata_manager, PathManager, FileOperations
)

class PreprocessingService:
    """🎯 Simplified preprocessing service menggunakan consolidated utils"""
    
    def __init__(self, config: Dict[str, Any] = None, progress_callback: Optional[Callable] = None):
        self.logger = get_logger(__name__)
        self._ui_mode = progress_callback is not None
        
        # Enhanced config validation
        if config is None:
            if not self._ui_mode:
                self.logger.warning("⚠️ No config provided, using defaults")
            self.config = get_default_preprocessing_config()
        else:
            try:
                self.config = validate_preprocessing_config(config)
                self.config = self._enhance_config_for_processing(self.config)
            except ValueError as e:
                if not self._ui_mode:
                    self.logger.error(f"❌ Invalid config: {str(e)}")
                raise ValueError(f"Invalid config: {str(e)}") from e
        
        # Progress management
        self.progress_bridge = create_progress_bridge()
        if progress_callback:
            self.progress_bridge.register_callback(progress_callback)
        
        # Initialize components
        self.validator = PreprocessingValidator(self.config, self.logger)
        self.engine = PreprocessingEngine(self.config)
        self.path_manager = create_path_manager(self.config)
        self.file_ops = create_file_operations(self.config)
        
        # Register progress callback ke engine
        if hasattr(self.engine, 'register_progress_callback'):
            self.engine.register_progress_callback(self._engine_progress_callback)
        
        # Extract config sections
        self.preprocessing_config = self.config.get('preprocessing', {})
        self.data_config = self.config.get('data', {})
    
    def _enhance_config_for_processing(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """🔧 Enhance config dengan file naming dan path configuration"""
        enhanced = config.copy()
        
        # Ensure required sections
        enhanced.setdefault('file_naming', {
            'preprocessed_pattern': 'pre_rp_{nominal}_{uuid}_{sequence}_{variance}',
            'preserve_uuid': True
        })
        
        # Ensure data paths
        data_section = enhanced.setdefault('data', {})
        data_section.setdefault('dir', 'data')
        
        # Ensure preprocessing output
        preprocessing_section = enhanced.setdefault('preprocessing', {})
        preprocessing_section.setdefault('output_dir', f"{data_section['dir']}/preprocessed")
        
        return enhanced
    
    def preprocess_dataset(self) -> Dict[str, Any]:
        """🚀 Main preprocessing method"""
        start_time = time.time()
        
        try:
            self._update_progress("overall", 0, 100, "🚀 Starting preprocessing")
            
            # Validation phase
            self._update_progress("overall", 5, 100, "🔍 Validating dataset structure")
            validation_result = self._comprehensive_validation()
            
            if not validation_result.get('success', False):
                error_msg = validation_result.get('message', 'Validation failed')
                self._update_progress("overall", 0, 100, f"❌ {error_msg}")
                return {
                    'success': False,
                    'message': error_msg,
                    'stats': validation_result.get('stats', {})
                }
            
            self._update_progress("overall", 20, 100, "✅ Validation complete")
            
            # Processing phase
            self._update_progress("overall", 25, 100, "🔄 Starting preprocessing pipeline")
            
            processing_result = self.engine.preprocess_dataset(
                progress_callback=self._engine_progress_callback
            )
            
            if not processing_result.get('success', False):
                error_msg = processing_result.get('message', 'Processing failed')
                self._update_progress("overall", 0, 100, f"❌ {error_msg}")
                return {
                    'success': False,
                    'message': error_msg,
                    'stats': processing_result.get('stats', {})
                }
            
            # Finalization
            processing_time = time.time() - start_time
            final_result = {
                'success': True,
                'message': "✅ Preprocessing berhasil",
                'processing_time': processing_time,
                'stats': self._compile_stats(processing_result, validation_result, processing_time),
                'configuration': self._get_config_summary()
            }
            
            self._update_progress("overall", 100, 100, "✅ Preprocessing complete")
            return final_result
            
        except Exception as e:
            error_msg = f"❌ Preprocessing error: {str(e)}"
            if not self._ui_mode:
                self.logger.error(error_msg)
            self._update_progress("overall", 0, 100, error_msg)
            return {
                'success': False,
                'message': error_msg,
                'processing_time': time.time() - start_time,
                'stats': {}
            }
    
    def check_preprocessed_exists(self, target_split: str = None) -> Tuple[bool, int]:
        """📊 Check preprocessed data existence"""
        try:
            output_dir = Path(self.preprocessing_config.get('output_dir', 'data/preprocessed'))
            if not output_dir.exists():
                return False, 0
            
            file_count = self._count_preprocessed_files(output_dir, target_split)
            return file_count > 0, file_count
            
        except Exception:
            return False, 0
    
    def _count_preprocessed_files(self, output_dir: Path, target_split: str = None) -> int:
        """📊 Count preprocessed files"""
        total_files = 0
        splits = [target_split] if target_split else self.preprocessing_config.get('target_splits', ['train', 'valid'])
        
        if isinstance(splits, str):
            splits = [splits] if splits != 'all' else ['train', 'valid', 'test']
        
        for split in splits:
            split_dir = output_dir / split / 'images'
            if split_dir.exists():
                npy_files = list(split_dir.glob('pre_*.npy'))
                total_files += len(npy_files)
        
        return total_files
    
    def cleanup_preprocessed_data(self, target_split: str = None) -> Dict[str, Any]:
        """🧹 Cleanup preprocessed data"""
        try:
            self._update_progress("current", 10, 100, "🧹 Starting cleanup")
            
            # Use PathManager untuk cleanup
            splits = [target_split] if target_split else None
            stats = self.path_manager.cleanup_output_dirs(splits, confirm=True)
            
            self._update_progress("current", 100, 100, "✅ Cleanup complete")
            
            return {
                'success': True,
                'message': f"🧹 Cleanup berhasil: {stats.get('files_removed', 0)} files removed",
                'stats': stats
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"❌ Cleanup error: {str(e)}",
                'stats': {'files_removed': 0}
            }
    
    def validate_dataset_only(self, target_split: str = None) -> Dict[str, Any]:
        """🔍 Validation-only method"""
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
                'message': f"❌ Validation error: {str(e)}",
                'summary': {}
            }
    
    def get_preprocessing_status(self) -> Dict[str, Any]:
        """📊 Get comprehensive preprocessing status"""
        try:
            # Check source structure
            source_validation = self.path_manager.validate_source_structure()
            
            # Check preprocessed data
            preprocessed_exists, file_count = self.check_preprocessed_exists()
            
            # System info
            disk_info = self.path_manager.check_disk_space()
            
            return {
                'success': True,
                'service_ready': True,
                'message': "✅ Service ready",
                'source_validation': source_validation,
                'preprocessed_status': {
                    'exists': preprocessed_exists,
                    'file_count': file_count
                },
                'system_info': {
                    'disk_space': disk_info,
                    'output_directory': str(self.path_manager.output_root)
                },
                'configuration': self._get_config_summary()
            }
            
        except Exception as e:
            return {
                'success': False,
                'service_ready': False,
                'message': f"❌ Service error: {str(e)}"
            }
    
    def get_sampling(self, target_split: str = "train", max_samples: int = 5) -> Dict[str, Any]:
        """🎲 Get sampling untuk preview"""
        try:
            img_dir, label_dir = self.path_manager.get_source_paths(target_split)
            image_files = self.file_ops.scan_images(img_dir)
            
            if not image_files:
                return {
                    'success': False,
                    'message': f"❌ No images found in {target_split}",
                    'samples': [],
                    'total_samples': 0
                }
            
            # Sample files
            sample_count = min(max_samples, len(image_files))
            import random
            sampled_files = random.sample(image_files, sample_count)
            
            # Get basic info
            samples = []
            for img_file in sampled_files:
                file_info = self.file_ops.get_file_info(img_file)
                samples.append({
                    'filename': img_file.name,
                    'path': str(img_file),
                    'size_mb': file_info.get('size_mb', 0),
                    'dimensions': f"{file_info.get('width', 0)}x{file_info.get('height', 0)}"
                })
            
            return {
                'success': True,
                'message': f"✅ Retrieved {sample_count} samples from {target_split}",
                'samples': samples,
                'total_samples': len(image_files)
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"❌ Sampling error: {str(e)}",
                'samples': [],
                'total_samples': 0
            }
    
    def _comprehensive_validation(self) -> Dict[str, Any]:
        """🔍 Comprehensive validation"""
        try:
            # Structure validation
            structure_result = self.path_manager.validate_source_structure()
            
            if not structure_result['is_valid']:
                return {
                    'success': False,
                    'message': f"❌ Invalid source structure: {', '.join(structure_result['missing_dirs'])}",
                    'stats': structure_result
                }
            
            return {
                'success': True,
                'message': f"✅ Structure validation passed: {structure_result['total_images']} images found",
                'stats': structure_result
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"❌ Validation error: {str(e)}",
                'stats': {}
            }
    
    def _compile_stats(self, processing_result: Dict, validation_result: Dict, processing_time: float) -> Dict[str, Any]:
        """📊 Compile comprehensive statistics"""
        processing_stats = processing_result.get('stats', {})
        validation_stats = validation_result.get('stats', {})
        
        return {
            'processing_time': round(processing_time, 2),
            'input': {
                'total_images': validation_stats.get('total_images', 0),
                'splits_processed': len(self.preprocessing_config.get('target_splits', []))
            },
            'output': {
                'total_processed': processing_stats.get('output', {}).get('total_processed', 0),
                'total_errors': processing_stats.get('output', {}).get('total_errors', 0),
                'success_rate': processing_stats.get('output', {}).get('success_rate', '0%')
            },
            'performance': {
                'avg_time_per_image': round(processing_time / max(processing_stats.get('input', {}).get('total_images', 1), 1), 3),
                'images_per_second': round(processing_stats.get('input', {}).get('total_images', 0) / max(processing_time, 0.1), 2)
            }
        }
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """⚙️ Get configuration summary"""
        norm_config = self.preprocessing_config.get('normalization', {})
        return {
            'target_splits': self.preprocessing_config.get('target_splits', ['train', 'valid']),
            'normalization': {
                'enabled': norm_config.get('enabled', True),
                'target_size': norm_config.get('target_size', [640, 640]),
                'preserve_aspect_ratio': norm_config.get('preserve_aspect_ratio', True)
            },
            'output_dir': self.preprocessing_config.get('output_dir', 'data/preprocessed'),
            'output_format': 'npy + txt'
        }
    
    def _engine_progress_callback(self, level: str, current: int, total: int, message: str):
        """🔧 Engine progress callback"""
        # Map engine progress ke overall progress
        if level in ['overall', 'step']:
            engine_progress = (current / total) if total > 0 else 0
            overall_progress = int(25 + (engine_progress * 65))  # 25% to 90%
            self._update_progress("overall", overall_progress, message)
        
        self._update_progress("current", current, message)
    
    def _update_progress(self, level: str, current: int, total: int = 100, message: str = ""):
        """📊 Update progress via bridge"""
        try:
            if self.progress_bridge:
                self.progress_bridge.update(level, current, total, message)
        except Exception as e:
            if not self._ui_mode:
                self.logger.debug(f"Progress callback error: {str(e)}")

def create_preprocessing_service(config: Dict[str, Any] = None, 
                               progress_callback: Optional[Callable] = None) -> PreprocessingService:
    """🏭 Factory untuk preprocessing service"""
    return PreprocessingService(config, progress_callback)