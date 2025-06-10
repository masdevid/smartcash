"""
File: smartcash/dataset/preprocessor/service.py
Deskripsi: Enhanced preprocessing service dengan dual progress tracker integration dan API consistency
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
    """🎯 Enhanced preprocessing service dengan dual progress tracker integration"""
    
    def __init__(self, config: Dict[str, Any] = None, progress_tracker=None):
        self.logger = get_logger(__name__)
        
        # Enhanced config validation dan merging
        if config is None:
            self.logger.warning("⚠️ Konfigurasi tidak disediakan, menggunakan konfigurasi default")
            self.config = get_default_preprocessing_config()
        else:
            try:
                self.config = validate_preprocessing_config(config)
            except ValueError as e:
                self.logger.error(f"❌ Konfigurasi tidak valid: {str(e)}")
                raise ValueError(f"Konfigurasi tidak valid: {str(e)}") from e
        
        # Enhanced progress bridge untuk dual tracker compatibility
        self.progress_bridge = ProgressBridge(progress_tracker) if progress_tracker else None
        
        # Initialize enhanced components
        self.validator = PreprocessingValidator(self.config, self.logger)
        self.engine = PreprocessingEngine(self.config)
        
        # Register progress callback jika ada
        if hasattr(self.engine, 'register_progress_callback') and self.progress_bridge:
            self.engine.register_progress_callback(self._dual_progress_callback)
        
        # Initialize utility components
        self.file_processor = FileProcessor(self.config)
        self.file_scanner = FileScanner()
        self.path_resolver = PathResolver(self.config)
        self.cleanup_manager = CleanupManager(self.config)
        
        # Extract commonly used config
        self.preprocessing_config = self.config.get('preprocessing', {})
        self.validation_config = self.preprocessing_config.get('validation', {})
        self.output_config = self.preprocessing_config.get('output', {})
    
    def preprocess_dataset(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """🚀 Enhanced preprocessing dengan dual progress integration"""
        start_time = time.time()
        
        try:
            # Register external progress callback
            if progress_callback:
                self._external_progress_callback = progress_callback
            
            self._update_progress("Memulai preprocessing dataset", 0.0)
            
            # Phase 1: Enhanced validation (0-20%)
            self._update_progress("🔍 Validating dataset structure", 0.1)
            print("\n=== Memulai Validasi ===")
            print(f"Config yang digunakan: {self.config}")
            
            validation_result = self._comprehensive_validation()
            print(f"Hasil validasi: {validation_result}")
            
            if not validation_result.get('success', False):
                error_msg = validation_result.get('message', 'Validation failed')
                print(f"❌ Validasi gagal: {error_msg}")
                return {
                    'success': False,
                    'message': error_msg,
                    'validation_errors': validation_result.get('errors', []),
                    'stats': validation_result.get('stats', {})
                }
                
            print("✅ Validasi berhasil")
            
            self._update_progress("✅ Validation completed", 0.2)
            
            # Phase 2: Enhanced preprocessing (20-90%)
            self._update_progress("🔄 Starting preprocessing pipeline", 0.3)
            
            # Pass the progress callback to the engine
            preprocessing_result = self.engine.preprocess_dataset(
                progress_callback=lambda level, current, total, msg: self._update_progress(msg, 0.3 + (0.6 * (current / max(1, total))))
            )
            
            if not preprocessing_result.get('success', False):
                return {
                    'success': False,
                    'message': preprocessing_result.get('message', 'Preprocessing failed'),
                    'stats': preprocessing_result.get('stats', {})
                }
            
            # Phase 3: Finalization (90-100%)
            self._update_progress("🏁 Finalizing preprocessing", 0.9)
            processing_time = time.time() - start_time
            
            # Enhanced final result
            final_result = {
                'success': True,
                'message': "✅ Preprocessing completed successfully",
                'processing_time': processing_time,
                'stats': self._compile_enhanced_stats(preprocessing_result, validation_result, processing_time),
                'configuration': self._get_config_summary()
            }
            
            self._update_progress("✅ Preprocessing selesai", 1.0)
            self._log_enhanced_summary(final_result)
            
            return final_result
            
        except Exception as e:
            import traceback
            error_traceback = '\n'.join(traceback.format_exception(type(e), e, e.__traceback__))
            error_msg = f"❌ Error dalam preprocessing: {str(e)}"
            self.logger.error(f"{error_msg}\n{error_traceback}")
            
            return {
                'success': False,
                'message': error_msg,
                'processing_time': time.time() - start_time,
                'stats': {}
            }
    
    def _comprehensive_validation(self) -> Dict[str, Any]:
        """🔍 Enhanced comprehensive validation"""
        print("\n=== Memulai Validasi Komprehensif ===")
        print(f"  - Preprocessing config: {self.preprocessing_config}")
        
        try:
            # Get target splits from config
            target_splits = self.preprocessing_config.get('target_splits', ['train', 'valid'])
            if isinstance(target_splits, str):
                target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
            
            print(f"  - Target splits: {target_splits}")
            
            # Validate each split
            validation_results = {}
            total_valid_images = 0
            all_errors = []
            
            for split in target_splits:
                print(f"\n  🔍 Memvalidasi split: {split}")
                try:
                    result = self.validator.validate_split(split)
                    validation_results[split] = result
                    print(f"  - Hasil validasi {split}: {result.get('is_valid', False)}")
                    
                    if result.get('is_valid', False):
                        valid_imgs = result.get('summary', {}).get('valid_images', 0)
                        total_valid_images += valid_imgs
                        print(f"  - Jumlah gambar valid: {valid_imgs}")
                    else:
                        errors = result.get('summary', {}).get('validation_errors', [])
                        all_errors.extend(errors)
                        print(f"  - Ditemukan {len(errors)} error")
                        for i, error in enumerate(errors[:3], 1):
                            print(f"    {i}. {error}")
                        if len(errors) > 3:
                            print(f"    ... dan {len(errors) - 3} error lainnya")
                except Exception as e:
                    error_msg = f"Error saat memvalidasi split {split}: {str(e)}"
                    print(f"  ❌ {error_msg}")
                    all_errors.append(error_msg)
                    validation_results[split] = {'is_valid': False, 'error': error_msg}
            
            # Overall validation success
            all_valid = all(result.get('is_valid', False) for result in validation_results.values())
            
            result = {
                'success': all_valid,
                'message': f"✅ Validation passed untuk {len(target_splits)} splits" if all_valid else f"⚠️ Validation issues di beberapa splits",
                'stats': {
                    'splits_validated': len(target_splits),
                    'total_valid_images': total_valid_images,
                    'validation_results': validation_results
                },
                'errors': all_errors[:10]  # Limit error list
            }
            
            print(f"\n=== Hasil Validasi ===")
            print(f"  - Status: {'✅ Berhasil' if result['success'] else '❌ Gagal'}")
            print(f"  - Pesan: {result['message']}")
            print(f"  - Total gambar valid: {total_valid_images}")
            if result['errors']:
                print(f"  - Error ditemukan: {len(result['errors'])}")
                for i, error in enumerate(result['errors'][:3], 1):
                    print(f"    {i}. {error}")
                if len(result['errors']) > 3:
                    print(f"    ... dan {len(result['errors']) - 3} error lainnya")
                    
            return result
            
        except Exception as e:
            return {
                'success': False,
                'message': f"❌ Validation error: {str(e)}",
                'stats': {},
                'errors': [str(e)]
            }
    
    def _compile_enhanced_stats(self, preprocessing_result: Dict, validation_result: Dict, processing_time: float) -> Dict[str, Any]:
        """📊 Compile enhanced statistics untuk UI logging"""
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
                'success_rate': f"{preprocessing_stats.get('success_rate', 0):.1f}%",
                'normalization_rate': f"{preprocessing_stats.get('normalization_rate', 0):.1f}%"
            },
            'performance': {
                'processing_time_seconds': round(processing_time, 2),
                'avg_time_per_image': round(processing_time / max(preprocessing_stats.get('total_processed', 1), 1), 3),
                'images_per_second': round(preprocessing_stats.get('total_processed', 0) / max(processing_time, 0.1), 2)
            },
            'configuration': preprocessing_stats.get('configuration', {}),
            'splits_detail': preprocessing_stats.get('splits', {}),
            'validation_summary': validation_stats
        }
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """⚙️ Get configuration summary untuk logging"""
        norm_config = self.preprocessing_config.get('normalization', {})
        return {
            'target_splits': self.preprocessing_config.get('target_splits', ['train', 'valid']),
            'normalization': {
                'enabled': norm_config.get('enabled', True),
                'method': norm_config.get('method', 'minmax'),
                'target_size': norm_config.get('target_size', [640, 640]),
                'preserve_aspect_ratio': norm_config.get('preserve_aspect_ratio', True)
            },
            'validation': {
                'enabled': self.validation_config.get('enabled', True),
                'move_invalid': self.validation_config.get('move_invalid', True)
            },
            'output': {
                'output_dir': self.preprocessing_config.get('output_dir', 'data/preprocessed'),
                'create_npy': self.output_config.get('create_npy', True)
            }
        }
    
    def _log_enhanced_summary(self, result: Dict[str, Any]):
        """📋 Log enhanced summary untuk UI"""
        stats = result.get('stats', {})
        input_stats = stats.get('input', {})
        output_stats = stats.get('output', {})
        perf_stats = stats.get('performance', {})
        
        self.logger.success(f"✅ Preprocessing berhasil!")
        self.logger.info(f"📊 Input: {input_stats.get('splits_processed', 0)} splits, {input_stats.get('total_input_images', 0)} gambar")
        self.logger.info(f"🎯 Output: {output_stats.get('total_processed', 0)} processed ({output_stats.get('success_rate', '0%')})")
        self.logger.info(f"🎨 Normalized: {output_stats.get('total_normalized', 0)} files ({output_stats.get('normalization_rate', '0%')})")
        self.logger.info(f"⚡ Performance: {perf_stats.get('images_per_second', 0)} img/sec, {perf_stats.get('processing_time_seconds', 0)}s total")
        self.logger.info(f"📁 Output saved to: {self.preprocessing_config.get('output_dir', 'data/preprocessed')}")
    
    def get_sampling(self, target_split: str = "train", max_samples: int = 5) -> Dict[str, Any]:
        """🎲 Enhanced sampling dengan better error handling"""
        try:
            # Get split directory
            split_dir = self.path_resolver.get_split_dir(target_split)
            image_files = self.file_scanner.scan_directory(
                split_dir / 'images', 
                extensions=['.jpg', '.jpeg', '.png']
            )
            
            if not image_files:
                return {
                    'success': False,
                    'message': f"❌ Tidak ada gambar ditemukan di {target_split}",
                    'samples': [],
                    'total_samples': 0
                }
            
            # Get random samples
            selected_files = random.sample(image_files, min(max_samples, len(image_files)))
            
            # Process each sample
            samples = []
            for img_path in selected_files:
                try:
                    # Read image
                    img = Image.open(img_path)
                    
                    # Get corresponding label path
                    label_path = self.path_resolver.get_label_path(img_path)
                    
                    samples.append({
                        'filename': img_path.name,
                        'image': np.array(img),
                        'image_path': str(img_path),
                        'label_path': str(label_path) if label_path.exists() else None,
                        'image_size': img.size
                    })
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal memproses sample {img_path}: {str(e)}")
            
            return {
                'success': True,
                'message': f"✅ Berhasil mengambil {len(samples)} sample dari {target_split}",
                'samples': samples,
                'total_samples': len(samples),
                'target_split': target_split
            }
            
        except Exception as e:
            error_msg = f"❌ Error sampling: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'samples': [],
                'total_samples': 0
            }
    
    def validate_dataset_only(self, target_split: str = "train") -> Dict[str, Any]:
        """🔍 Enhanced validation only dengan detailed reporting"""
        try:
            result = self.validator.validate_split(target_split)
            
            return {
                'success': result.get('is_valid', False),
                'message': result.get('message', 'Validation completed'),
                'target_split': target_split,
                'validation_result': result,
                'summary': result.get('summary', {})
            }
        except Exception as e:
            error_msg = f"❌ Error validasi dataset: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'target_split': target_split,
                'summary': {}
            }
    
    def cleanup_preprocessed_data(self, target_split: str = None) -> Dict[str, Any]:
        """🧹 Enhanced cleanup dengan detailed stats"""
        try:
            from smartcash.dataset.preprocessor.utils import create_preprocessing_cleanup_manager
            
            cleanup_mgr = create_preprocessing_cleanup_manager(self.config, self.progress_bridge)
            result = cleanup_mgr.cleanup_data(target='preprocessed', target_split=target_split)
            
            # Enhanced result formatting
            if result.get('status') == 'success':
                stats = result.get('stats', {})
                return {
                    'success': True,
                    'message': f"🧹 Cleanup berhasil: {stats.get('files_removed', 0)} file dihapus",
                    'stats': stats,
                    'target_split': target_split or 'all'
                }
            else:
                return {
                    'success': False,
                    'message': result.get('message', 'Cleanup failed'),
                    'target_split': target_split or 'all'
                }
                
        except Exception as e:
            error_msg = f"❌ Error cleanup: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'target_split': target_split or 'all'
            }
    
    def get_preprocessing_status(self) -> Dict[str, Any]:
        """📊 Enhanced status dengan detailed system info"""
        try:
            # Check directory structure
            data_dir = Path(self.config.get('data', {}).get('dir', 'data'))
            output_dir = Path(self.preprocessing_config.get('output_dir', 'data/preprocessed'))
            
            # Check splits
            target_splits = self.preprocessing_config.get('target_splits', ['train', 'valid'])
            if isinstance(target_splits, str):
                target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
            
            split_status = {}
            for split in target_splits:
                split_dir = data_dir / split
                images_dir = split_dir / 'images'
                labels_dir = split_dir / 'labels'
                
                split_status[split] = {
                    'exists': split_dir.exists(),
                    'has_images': images_dir.exists() and any(images_dir.glob('*.jpg')),
                    'has_labels': labels_dir.exists() and any(labels_dir.glob('*.txt')),
                    'image_count': len(list(images_dir.glob('*.*'))) if images_dir.exists() else 0
                }
            
            return {
                'success': True,
                'service_ready': True,
                'message': '✅ Preprocessing service ready',
                'system_info': {
                    'data_directory': str(data_dir),
                    'output_directory': str(output_dir),
                    'target_splits': target_splits,
                    'split_status': split_status
                },
                'configuration': self._get_config_summary()
            }
        except Exception as e:
            return {
                'success': False,
                'service_ready': False,
                'message': f'❌ Service error: {str(e)}'
            }
    
    def _dual_progress_callback(self, level: str, current: int, total: int, message: str):
        """📈 Dual progress callback untuk UI integration"""
        try:
            # Update progress bridge
            if self.progress_bridge:
                progress_value = (current / total) * 100 if total > 0 else 0
                self.progress_bridge.update(progress_value, message)
            
            # Call external callback jika ada
            if hasattr(self, '_external_progress_callback') and callable(self._external_progress_callback):
                self._external_progress_callback(level, current, total, message)
        
        except Exception as e:
            self.logger.debug(f"Progress callback error: {str(e)}")
    
    def _update_progress(self, message: str, progress: float):
        """📊 Update progress dengan dual tracker compatibility"""
        total_steps = 100
        current_step = int(progress * total_steps)
        
        if self.progress_bridge:
            self.progress_bridge.update("overall", current_step, total_steps, message)
        
        # Call dual progress callback
        self._dual_progress_callback("overall", current_step, total_steps, message)

def create_preprocessing_service(config: Dict[str, Any] = None, 
                             progress_tracker=None) -> PreprocessingService:
    """🏭 Factory untuk create enhanced preprocessing service"""
    return PreprocessingService(config, progress_tracker)