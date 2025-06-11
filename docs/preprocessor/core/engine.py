"""
File: smartcash/dataset/preprocessor/core/engine.py
Deskripsi: Simplified preprocessing engine menggunakan consolidated utils
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor.utils import (
    ValidationCore, FileOperations, PathManager, MetadataManager, 
    YOLONormalizer, ProgressBridge, create_validation_core,
    create_file_operations, create_path_manager, create_metadata_manager,
    create_yolo_normalizer, create_compatible_bridge
)

class PreprocessingValidator:
    """🔍 Simplified validator menggunakan ValidationCore"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config.get('preprocessing', {})
        self.logger = logger or get_logger()
        self.validation_core = create_validation_core(self.config.get('validation', {}))
    
    def validate_split(self, split: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """🎯 Validate split menggunakan consolidated utils"""
        try:
            # Path resolution
            path_manager = create_path_manager(self.config)
            img_dir, label_dir = path_manager.get_source_paths(split)
            
            if not img_dir.exists() or not label_dir.exists():
                return {
                    'is_valid': False,
                    'message': f"❌ Direktori tidak ditemukan: {img_dir} atau {label_dir}",
                    'summary': {'total_images': 0, 'valid_images': 0}
                }
            
            # File scanning
            file_ops = create_file_operations()
            image_files = file_ops.scan_images(img_dir)
            
            if not image_files:
                return {
                    'is_valid': False,
                    'message': f"❌ Tidak ada gambar di {img_dir}",
                    'summary': {'total_images': 0, 'valid_images': 0}
                }
            
            # Batch validation dengan progress
            results = self.validation_core.batch_validate_pairs(image_files, progress_callback)
            
            # Compile stats
            valid_count = sum(1 for r in results.values() if r.is_valid)
            total_count = len(results)
            
            return {
                'is_valid': valid_count == total_count,
                'message': f"✅ {valid_count}/{total_count} files valid" if valid_count == total_count else f"⚠️ {valid_count}/{total_count} files valid",
                'summary': {
                    'total_images': total_count,
                    'valid_images': valid_count,
                    'validation_results': results
                }
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'message': f"❌ Validation error: {str(e)}",
                'summary': {'total_images': 0, 'valid_images': 0}
            }

class PreprocessingEngine:
    """🚀 Simplified preprocessing engine menggunakan consolidated utils"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger()
        
        # Initialize consolidated components
        self.validator = PreprocessingValidator(config, self.logger)
        self.file_ops = create_file_operations(config)
        self.path_manager = create_path_manager(config)
        self.metadata_manager = create_metadata_manager(config.get('file_naming', {}))
        self.normalizer = create_yolo_normalizer(config.get('preprocessing', {}).get('normalization', {}))
        
        # Configuration
        self.preprocessing_config = config.get('preprocessing', {})
        self.target_splits = self._resolve_target_splits()
        self.progress_callback = None
    
    def _resolve_target_splits(self) -> List[str]:
        """🎯 Resolve target splits dari config"""
        splits = self.preprocessing_config.get('target_splits', ['train', 'valid'])
        if isinstance(splits, str):
            return [splits] if splits != 'all' else ['train', 'valid', 'test']
        return splits
    
    def register_progress_callback(self, callback: Callable[[str, int, int, str], None]):
        """📊 Register progress callback"""
        self.progress_callback = callback
    
    def preprocess_dataset(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """🎯 Main preprocessing dengan consolidated utils"""
        current_callback = progress_callback or self.progress_callback
        start_time = time.time()
        
        try:
            self._update_progress("overall", 0, 100, "🚀 Starting preprocessing", current_callback)
            
            # Phase 1: Validation (0-20%)
            validation_results = self._validate_all_splits(current_callback)
            if not validation_results['valid']:
                return {
                    'success': False,
                    'message': validation_results['message'],
                    'stats': validation_results.get('stats', {})
                }
            
            self._update_progress("overall", 20, 100, "✅ Validation complete", current_callback)
            
            # Phase 2: Setup output structure (20-30%)
            self._setup_output_structure()
            self._update_progress("overall", 30, 100, "📁 Output structure ready", current_callback)
            
            # Phase 3: Process splits (30-90%)
            processing_results = self._process_all_splits(current_callback)
            
            # Phase 4: Finalization (90-100%)
            self._update_progress("overall", 90, 100, "🏁 Finalizing", current_callback)
            
            final_stats = self._compile_final_stats(processing_results, validation_results, time.time() - start_time)
            self._update_progress("overall", 100, 100, "✅ Complete", current_callback)
            
            return {
                'success': True,
                'message': f"✅ Preprocessing complete for {len(self.target_splits)} splits",
                'stats': final_stats
            }
            
        except Exception as e:
            error_msg = f"❌ Preprocessing error: {str(e)}"
            self.logger.error(error_msg)
            self._update_progress("overall", 0, 100, error_msg, current_callback)
            return {'success': False, 'message': error_msg, 'stats': {}}
    
    def _validate_all_splits(self, callback: Optional[Callable]) -> Dict[str, Any]:
        """🔍 Validate all splits dengan progress"""
        all_valid = True
        total_images = 0
        results = {}
        
        for i, split in enumerate(self.target_splits):
            progress = int((i / len(self.target_splits)) * 20)
            self._update_progress("overall", progress, 100, f"Validating {split}", callback)
            
            def validation_progress(level, current, total, message):
                if callback:
                    callback("current", current, total, f"{split}: {message}")
            
            result = self.validator.validate_split(split, validation_progress)
            results[split] = result
            
            if not result['is_valid']:
                all_valid = False
            else:
                total_images += result['summary'].get('valid_images', 0)
        
        return {
            'valid': all_valid,
            'message': f"Validation {'passed' if all_valid else 'partial'} for {len(self.target_splits)} splits",
            'stats': {'total_images': total_images, 'results': results}
        }
    
    def _setup_output_structure(self):
        """📁 Setup output directories"""
        self.path_manager.create_output_structure(self.target_splits)
        self.path_manager.create_auxiliary_dirs()
    
    def _process_all_splits(self, callback: Optional[Callable]) -> Dict[str, Any]:
        """🔄 Process all splits dengan progress"""
        all_results = {}
        split_progress_step = 60 / len(self.target_splits)
        
        for i, split in enumerate(self.target_splits):
            split_start = 30 + (i * split_progress_step)
            split_end = 30 + ((i + 1) * split_progress_step)
            
            self._update_progress("overall", int(split_start), 100, f"Processing {split}", callback)
            
            def split_progress(current, total, message):
                overall_progress = split_start + ((current / total) * split_progress_step)
                if callback:
                    callback("overall", int(overall_progress), 100, f"{split}: {message}")
                    callback("current", current, total, message)
            
            result = self._process_single_split(split, split_progress)
            all_results[split] = result
        
        return all_results
    
    def _process_single_split(self, split: str, progress_callback: Callable) -> Dict[str, Any]:
        """🔄 Process single split dengan optimized pipeline"""
        try:
            # Get paths
            img_dir, label_dir = self.path_manager.get_source_paths(split)
            out_img_dir, out_label_dir = self.path_manager.get_output_paths(split)
            
            # Scan files
            image_files = self.file_ops.scan_images(img_dir)
            if not image_files:
                return {'status': 'skipped', 'message': f'No images in {split}', 'stats': {}}
            
            # Process files
            stats = {'total': len(image_files), 'processed': 0, 'errors': 0}
            
            for i, img_file in enumerate(image_files):
                try:
                    progress_callback(i + 1, len(image_files), f"Processing {img_file.name}")
                    
                    # Load dan preprocess image
                    image = self.file_ops.read_image(img_file)
                    if image is None:
                        stats['errors'] += 1
                        continue
                    
                    # YOLO preprocessing
                    normalized_image, metadata = self.normalizer.preprocess_for_yolo(image)
                    
                    # Generate output filename
                    preprocessed_name = self.metadata_manager.generate_preprocessed_filename(img_file.name)
                    
                    # Save normalized image sebagai .npy
                    output_img_path = out_img_dir / f"{preprocessed_name}.npy"
                    self.file_ops.save_normalized_array(output_img_path, normalized_image, metadata)
                    
                    # Copy label dengan matching filename
                    label_file = label_dir / f"{img_file.stem}.txt"
                    if label_file.exists():
                        output_label_path = out_label_dir / f"{preprocessed_name}.txt"
                        labels = self.file_ops.read_yolo_label(label_file)
                        self.file_ops.write_yolo_label(output_label_path, labels)
                    
                    stats['processed'] += 1
                    
                except Exception as e:
                    self.logger.error(f"❌ Error processing {img_file.name}: {str(e)}")
                    stats['errors'] += 1
            
            return {
                'status': 'success',
                'message': f"✅ {split}: {stats['processed']}/{stats['total']} processed",
                'stats': stats
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"❌ Error processing {split}: {str(e)}",
                'stats': {'total': 0, 'processed': 0, 'errors': 1}
            }
    
    def _compile_final_stats(self, processing_results: Dict, validation_results: Dict, processing_time: float) -> Dict[str, Any]:
        """📊 Compile final statistics"""
        total_processed = sum(r.get('stats', {}).get('processed', 0) for r in processing_results.values())
        total_errors = sum(r.get('stats', {}).get('errors', 0) for r in processing_results.values())
        total_input = validation_results.get('stats', {}).get('total_images', 0)
        
        return {
            'processing_time_seconds': round(processing_time, 2),
            'input': {
                'total_images': total_input,
                'splits': len(self.target_splits)
            },
            'output': {
                'total_processed': total_processed,
                'total_errors': total_errors,
                'success_rate': f"{(total_processed / max(total_input, 1)) * 100:.1f}%"
            },
            'configuration': {
                'target_splits': self.target_splits,
                'normalization': self.normalizer.config,
                'output_format': 'npy + txt'
            },
            'splits_detail': processing_results
        }
    
    def _update_progress(self, level: str, current: int, total: int, message: str, callback: Optional[Callable]):
        """📈 Update progress dengan safe calling"""
        if callback:
            try:
                callback(level, current, total, message)
            except Exception as e:
                self.logger.debug(f"⚠️ Progress callback error: {str(e)}")