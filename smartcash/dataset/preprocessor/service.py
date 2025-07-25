"""
File: smartcash/dataset/preprocessor/service.py
Deskripsi: Main preprocessing service dengan simplified validation dan YOLO focus
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Optional

from smartcash.common.logger import get_logger
from .config.validator import validate_preprocessing_config
from .config.defaults import get_default_config
from .utils.progress_bridge import create_preprocessing_bridge
from .validation.directory_validator import DirectoryValidator
from .validation.filename_validator import FilenameValidator
from .core.normalizer import YOLONormalizer
from .core.file_processor import FileProcessor
from .core.stats_collector import StatsCollector

class PreprocessingService:
    """🚀 Main preprocessing service dengan minimal validation dan YOLO normalization"""
    
    def __init__(self, config: Dict[str, Any] = None, progress_callback: Optional[Callable] = None):
        self.logger = get_logger(__name__)
        
        # Validate dan setup config
        if config is None:
            self.config = get_default_config()
        else:
            self.config = validate_preprocessing_config(config)
        
        # Extract config sections
        self.preprocessing_config = self.config['preprocessing']
        self.data_config = self.config['data']
        self.performance_config = self.config.get('performance', {})
        
        # Initialize components
        self.normalizer = YOLONormalizer(self.preprocessing_config['normalization'])
        self.file_processor = FileProcessor(self.performance_config)
        self.stats_collector = StatsCollector()
        
        # Progress management
        self.progress_bridge = create_preprocessing_bridge()
        if progress_callback:
            self.progress_bridge.register_callback(progress_callback)
        
        # Validation components (minimal setup)
        self.dir_validator = DirectoryValidator({'auto_fix': True})
        self.filename_validator = FilenameValidator()
    
    def preprocess_dataset(self) -> Dict[str, Any]:
        """🎯 Main preprocessing method dengan phases"""
        start_time = time.time()
        
        try:
            # Phase 1: Validation (20%)
            self.progress_bridge.start_phase('validation', "🔍 Starting validation")
            validation_result = self._validate_inputs()
            
            if not validation_result['success']:
                return validation_result
            
            self.progress_bridge.complete_phase('validation', "✅ Validation complete")
            
            # Phase 2: Processing (70%)
            self.progress_bridge.start_phase('processing', "🔄 Starting preprocessing")
            processing_result = self._process_all_splits()
            
            if not processing_result['success']:
                return processing_result
            
            self.progress_bridge.complete_phase('processing', "✅ Processing complete")
            
            # Phase 3: Finalization (10%)
            self.progress_bridge.start_phase('finalization', "🏁 Finalizing")
            final_stats = self._compile_final_stats(processing_result, time.time() - start_time)
            
            self.progress_bridge.complete_phase('finalization', "✅ Complete")
            
            return {
                'success': True,
                'message': f"✅ Preprocessing completed for {len(self._get_target_splits())} splits",
                'stats': final_stats
            }
            
        except Exception as e:
            error_msg = f"❌ Preprocessing error: {str(e)}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'message': error_msg,
                'processing_time': time.time() - start_time,
                'stats': {}
            }
    
    def get_preprocessing_status(self) -> Dict[str, Any]:
        """📊 Get comprehensive status"""
        try:
            # Check directory structure
            structure_valid = self.dir_validator.validate_structure(
                self.data_config['dir'], 
                self._get_target_splits()
            )
            
            # Check output directory
            output_dir = Path(self.data_config['preprocessed_dir'])
            output_exists = output_dir.exists()
            
            # Get basic file counts
            file_stats = {}
            data_root = Path(self.data_config['dir'])
            
            for split in self._get_target_splits():
                split_path = data_root / split
                if split_path.exists():
                    # Check raw images in source directory
                    raw_count = len(self.file_processor.scan_files(split_path / 'images', 'rp_'))
                    # Check preprocessed files in output directory  
                    preprocessed_count = len(self.file_processor.scan_files(output_dir / split / 'images', 'pre_', {'.npy'}))
                    
                    file_stats[split] = {
                        'raw_images': raw_count,
                        'preprocessed_files': preprocessed_count,
                        'ready_for_processing': raw_count > 0
                    }
            
            return {
                'success': True,
                'service_ready': structure_valid['is_valid'],
                'message': "✅ Service ready" if structure_valid['is_valid'] else "⚠️ Structure issues detected",
                'structure_validation': structure_valid,
                'output_directory': {
                    'exists': output_exists,
                    'path': str(output_dir)
                },
                'file_statistics': file_stats,
                'configuration': {
                    'target_splits': self._get_target_splits(),
                    'normalization': self.preprocessing_config['normalization'],
                    'validation_enabled': self.preprocessing_config['validation']['enabled']
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'service_ready': False,
                'message': f"❌ Status check error: {str(e)}"
            }
    
    def _validate_inputs(self) -> Dict[str, Any]:
        """🔍 Minimal validation - structure dan filenames only"""
        try:
            # Directory structure validation (with auto-fix)
            self.progress_bridge.update_phase_progress(25, 100, "Checking directory structure")
            structure_result = self.dir_validator.validate_structure(
                self.data_config['dir'], 
                self._get_target_splits()
            )
            
            if not structure_result['is_valid']:
                return {
                    'success': False,
                    'message': f"❌ Directory structure invalid: {', '.join(structure_result['missing_dirs'])}"
                }
            
            # Filename validation (with auto-rename if enabled)
            if self.preprocessing_config['validation']['filename_pattern']:
                self.progress_bridge.update_phase_progress(75, 100, "Validating filenames")
                
                for split in self._get_target_splits():
                    split_path = Path(self.data_config['dir']) / split / 'images'
                    if split_path.exists():
                        image_files = self.file_processor.scan_files(split_path)
                        
                        if self.preprocessing_config['validation']['auto_fix']:
                            rename_result = self.filename_validator.rename_invalid_files(image_files)
                            if not rename_result['success']:
                                return {
                                    'success': False,
                                    'message': f"❌ Filename validation failed for {split}"
                                }
            
            self.progress_bridge.update_phase_progress(100, 100, "Validation complete")
            
            return {
                'success': True,
                'message': "✅ Input validation passed"
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"❌ Validation error: {str(e)}"
            }
    
    def _process_all_splits(self) -> Dict[str, Any]:
        """🔄 Process all target splits dengan progress format baru"""
        splits = self._get_target_splits()
        split_results = {}
        total_processed = 0
        
        # Setup output directories
        output_dir = Path(self.data_config['preprocessed_dir'])
        self.dir_validator.create_preprocessing_structure(output_dir, splits)
        
        # Setup split processing
        self.progress_bridge.setup_split_processing(splits)
        
        for split in splits:
            # Start split processing
            self.progress_bridge.start_split(split)
            
            split_result = self._process_single_split(split)
            split_results[split] = split_result
            total_processed += split_result.get('processed', 0)
            
            # Complete split
            self.progress_bridge.complete_split(split)
        
        return {
            'success': True,
            'total_processed': total_processed,
            'by_split': split_results
        }
    
    def _process_single_split(self, split: str) -> Dict[str, Any]:
        """🎯 Process single split dengan split-specific progress"""
        try:
            # Get paths
            input_dir = Path(self.data_config['dir']) / split
            output_dir = Path(self.data_config['preprocessed_dir']) / split
            
            # Scan input files
            image_files = self.file_processor.scan_files(input_dir / 'images', 'rp_')
            
            if not image_files:
                return {'processed': 0, 'errors': 0, 'message': f'No images in {split}'}
            
            # Process files dengan split progress
            processed = 0
            errors = 0
            
            for i, img_file in enumerate(image_files):
                try:
                    # Update split progress only at milestones (reduce verbosity)
                    batch_size = max(1, len(image_files) // 10)  # Update every 10%
                    if i % batch_size == 0 or i == len(image_files) - 1:
                        self.progress_bridge.update_split_progress(
                            i + 1, len(image_files), f"Processing batch {i//batch_size + 1}/10..."
                        )
                    
                    # Load image
                    image = self.file_processor.read_image(img_file)
                    if image is None:
                        errors += 1
                        continue
                    
                    # Normalize untuk YOLO
                    normalized, metadata = self.normalizer.normalize(image)
                    
                    # Generate output filename
                    output_name = f"pre_{img_file.stem}.npy"
                    output_path = output_dir / 'images' / output_name
                    
                    # Save normalized array
                    success = self.file_processor.save_normalized_array(output_path, normalized, metadata)
                    
                    if success:
                        # Copy corresponding label
                        label_file = input_dir / 'labels' / f"{img_file.stem}.txt"
                        if label_file.exists():
                            label_output = output_dir / 'labels' / f"pre_{img_file.stem}.txt"
                            self.file_processor.copy_file(label_file, label_output)
                        
                        processed += 1
                    else:
                        errors += 1
                
                except Exception as e:
                    self.logger.error(f"❌ Error processing {img_file.name}: {str(e)}")
                    errors += 1
            
            return {
                'processed': processed,
                'errors': errors,
                'total': len(image_files),
                'message': f"✅ {split}: {processed}/{len(image_files)} processed"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Split processing error {split}: {str(e)}")
            return {
                'processed': 0,
                'errors': 1,
                'message': f"❌ {split} processing failed: {str(e)}"
            }
    
    def _compile_final_stats(self, processing_result: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
        """📊 Compile comprehensive final statistics"""
        total_processed = processing_result['total_processed']
        total_errors = sum(split_data.get('errors', 0) for split_data in processing_result['by_split'].values())
        total_input = sum(split_data.get('total', 0) for split_data in processing_result['by_split'].values())
        
        return {
            'processing_time_seconds': round(processing_time, 2),
            'processing_time_minutes': round(processing_time / 60, 2),
            'input': {
                'total_images': total_input,
                'splits_processed': len(self._get_target_splits())
            },
            'output': {
                'total_processed': total_processed,
                'total_errors': total_errors,
                'success_rate': f"{(total_processed / max(total_input, 1)) * 100:.1f}%"
            },
            'performance': {
                'avg_time_per_image_ms': round((processing_time / max(total_input, 1)) * 1000, 1),
                'images_per_second': round(total_input / max(processing_time, 0.1), 2)
            },
            'configuration': {
                'target_splits': self._get_target_splits(),
                'normalization_config': self.preprocessing_config['normalization'],
                'output_format': 'npy + txt',
                'output_directory': self.data_config['preprocessed_dir']
            },
            'by_split': processing_result['by_split']
        }
    
    def _get_target_splits(self) -> list:
        """🎯 Get target splits dari config"""
        splits = self.preprocessing_config['target_splits']
        if isinstance(splits, str):
            return [splits] if splits != 'all' else ['train', 'valid', 'test']
        return splits