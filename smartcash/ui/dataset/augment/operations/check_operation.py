"""
File: smartcash/ui/dataset/augment/operations/check_operation.py
Description: Dataset check operation with preserved business logic

This operation handles dataset validation and statistics gathering
with all original business logic preserved.
"""

from typing import Dict, Any, Optional
import logging
import os
import time
from smartcash.ui.core.decorators import handle_ui_errors
from ..constants import BANKNOTE_CLASSES, FILE_PROCESSING_CONFIG


class CheckOperation:
    """
    Dataset check operation with preserved business logic.
    
    Features:
    - 📊 Dataset structure validation
    - 📈 Image count statistics
    - 🔍 Class distribution analysis
    - ✅ File format validation
    - 📝 Detailed reporting
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """
        Initialize check operation.
        
        Args:
            ui_components: UI components for progress updates
        """
        self.ui_components = ui_components or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Operation state
        self._progress = 0.0
        self._is_cancelled = False
        
        # Check results
        self._dataset_stats = {}
        self._class_distribution = {}
        self._validation_errors = []
        
        self.logger.debug("🔍 CheckOperation initialized")
    
    @handle_ui_errors(error_component_title="Dataset Check Operation Error")
    def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute dataset check operation.
        
        Args:
            config: Check configuration
            
        Returns:
            Dictionary containing check results
        """
        start_time = time.time()
        self.logger.info("🔍 Starting dataset check")
        
        try:
            # Reset state
            self._reset_state()
            
            # Get data directory
            data_dir = config.get('data', {}).get('dir', 'data')
            
            # Execute check phases
            structure_result = self._check_dataset_structure(data_dir)
            if not structure_result['success']:
                return structure_result
            
            statistics_result = self._gather_statistics(data_dir)
            if not statistics_result['success']:
                return statistics_result
            
            validation_result = self._validate_files(data_dir)
            
            # Compile results
            processing_time = time.time() - start_time
            
            final_result = {
                'success': True,
                'processing_time': processing_time,
                'dataset_stats': self._dataset_stats,
                'class_distribution': self._class_distribution,
                'validation_errors': self._validation_errors,
                'structure_check': structure_result,
                'statistics': statistics_result,
                'file_validation': validation_result,
                'total_images': sum(self._class_distribution.values()),
                'classes_found': len(self._class_distribution),
                'is_balanced': self._analyze_balance()
            }
            
            # Update UI with results
            self._update_check_results(final_result)
            
            self.logger.info(f"✅ Dataset check completed in {processing_time:.2f}s")
            return final_result
            
        except Exception as e:
            self.logger.error(f"❌ Dataset check failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def cancel(self) -> None:
        """Cancel the check operation."""
        self._is_cancelled = True
        self.logger.info("🛑 Check operation cancelled")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current operation status."""
        return {
            'progress': self._progress,
            'is_cancelled': self._is_cancelled,
            'dataset_stats': self._dataset_stats,
            'validation_errors': len(self._validation_errors)
        }
    
    def _reset_state(self) -> None:
        """Reset operation state."""
        self._progress = 0.0
        self._is_cancelled = False
        self._dataset_stats = {}
        self._class_distribution = {}
        self._validation_errors = []
    
    def _check_dataset_structure(self, data_dir: str) -> Dict[str, Any]:
        """Check dataset directory structure."""
        self._update_progress(0.0, "Checking dataset structure...")
        
        try:
            # Check if data directory exists
            if not os.path.exists(data_dir):
                return {
                    'success': False,
                    'error': f"Data directory does not exist: {data_dir}"
                }
            
            if not os.path.isdir(data_dir):
                return {
                    'success': False,
                    'error': f"Data path is not a directory: {data_dir}"
                }
            
            # Check for required subdirectories (train, valid, test)
            expected_splits = ['train', 'valid', 'test']
            found_splits = []
            missing_splits = []
            
            for split in expected_splits:
                split_path = os.path.join(data_dir, split)
                if os.path.exists(split_path) and os.path.isdir(split_path):
                    found_splits.append(split)
                else:
                    missing_splits.append(split)
            
            # Check class directories within splits
            class_structure = {}
            for split in found_splits:
                split_path = os.path.join(data_dir, split)
                classes_in_split = []
                
                for item in os.listdir(split_path):
                    item_path = os.path.join(split_path, item)
                    if os.path.isdir(item_path):
                        classes_in_split.append(item)
                
                class_structure[split] = classes_in_split
            
            self._update_progress(0.3, "Dataset structure validated")
            
            return {
                'success': True,
                'data_directory': data_dir,
                'found_splits': found_splits,
                'missing_splits': missing_splits,
                'class_structure': class_structure,
                'structure_valid': len(found_splits) > 0
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Structure check error: {str(e)}"}
    
    def _gather_statistics(self, data_dir: str) -> Dict[str, Any]:
        """Gather dataset statistics."""
        self._update_progress(0.3, "Gathering dataset statistics...")
        
        try:
            # Initialize statistics
            total_images = 0
            split_stats = {}
            class_distribution = {}
            
            # Supported image formats
            supported_formats = FILE_PROCESSING_CONFIG.get('supported_formats', ['.jpg', '.jpeg', '.png', '.bmp'])
            
            # Check each split
            splits = ['train', 'valid', 'test']
            
            for i, split in enumerate(splits):
                if self._is_cancelled:
                    return {'success': False, 'error': 'Operation cancelled'}
                
                split_path = os.path.join(data_dir, split)
                if not os.path.exists(split_path):
                    continue
                
                split_images = 0
                split_classes = {}
                
                # Check each class directory
                for class_name in os.listdir(split_path):
                    class_path = os.path.join(split_path, class_name)
                    if not os.path.isdir(class_path):
                        continue
                    
                    # Count images in class
                    class_images = 0
                    for file_name in os.listdir(class_path):
                        file_ext = os.path.splitext(file_name)[1].lower()
                        if file_ext in supported_formats:
                            class_images += 1
                    
                    if class_images > 0:
                        split_classes[class_name] = class_images
                        split_images += class_images
                        
                        # Update global class distribution
                        if class_name not in class_distribution:
                            class_distribution[class_name] = 0
                        class_distribution[class_name] += class_images
                
                split_stats[split] = {
                    'total_images': split_images,
                    'classes': split_classes,
                    'class_count': len(split_classes)
                }
                
                total_images += split_images
                
                # Update progress
                progress = 0.3 + ((i + 1) / len(splits)) * 0.4
                self._update_progress(progress, f"Analyzed {split} split: {split_images} images")
            
            # Store results
            self._dataset_stats = {
                'total_images': total_images,
                'split_stats': split_stats,
                'total_classes': len(class_distribution),
                'supported_formats': supported_formats
            }
            self._class_distribution = class_distribution
            
            self._update_progress(0.7, "Statistics gathering completed")
            
            return {
                'success': True,
                'dataset_stats': self._dataset_stats,
                'class_distribution': class_distribution
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Statistics error: {str(e)}"}
    
    def _validate_files(self, data_dir: str) -> Dict[str, Any]:
        """Validate dataset files."""
        self._update_progress(0.7, "Validating files...")
        
        try:
            validation_results = {
                'valid_files': 0,
                'invalid_files': 0,
                'empty_directories': [],
                'unsupported_formats': [],
                'corrupted_files': []
            }
            
            supported_formats = FILE_PROCESSING_CONFIG.get('supported_formats', ['.jpg', '.jpeg', '.png', '.bmp'])
            
            # Check each split
            splits = ['train', 'valid', 'test']
            
            for split in splits:
                if self._is_cancelled:
                    return {'success': False, 'error': 'Operation cancelled'}
                
                split_path = os.path.join(data_dir, split)
                if not os.path.exists(split_path):
                    continue
                
                # Check each class directory
                for class_name in os.listdir(split_path):
                    class_path = os.path.join(split_path, class_name)
                    if not os.path.isdir(class_path):
                        continue
                    
                    class_files = os.listdir(class_path)
                    if not class_files:
                        validation_results['empty_directories'].append(f"{split}/{class_name}")
                        continue
                    
                    # Check each file
                    for file_name in class_files:
                        file_path = os.path.join(class_path, file_name)
                        file_ext = os.path.splitext(file_name)[1].lower()
                        
                        if file_ext not in supported_formats:
                            validation_results['unsupported_formats'].append(f"{split}/{class_name}/{file_name}")
                            validation_results['invalid_files'] += 1
                        else:
                            # Basic file validation
                            try:
                                if os.path.getsize(file_path) > 0:
                                    validation_results['valid_files'] += 1
                                else:
                                    validation_results['corrupted_files'].append(f"{split}/{class_name}/{file_name}")
                                    validation_results['invalid_files'] += 1
                            except Exception:
                                validation_results['corrupted_files'].append(f"{split}/{class_name}/{file_name}")
                                validation_results['invalid_files'] += 1
            
            # Store validation errors
            self._validation_errors = []
            if validation_results['empty_directories']:
                self._validation_errors.append(f"Empty directories: {len(validation_results['empty_directories'])}")
            if validation_results['unsupported_formats']:
                self._validation_errors.append(f"Unsupported formats: {len(validation_results['unsupported_formats'])}")
            if validation_results['corrupted_files']:
                self._validation_errors.append(f"Corrupted files: {len(validation_results['corrupted_files'])}")
            
            self._update_progress(1.0, "File validation completed")
            
            return {
                'success': True,
                'validation_results': validation_results,
                'validation_errors': self._validation_errors
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Validation error: {str(e)}"}
    
    def _analyze_balance(self) -> bool:
        """Analyze if dataset is balanced."""
        if not self._class_distribution:
            return False
        
        # Check if classes are reasonably balanced (within 50% of average)
        values = list(self._class_distribution.values())
        if not values:
            return False
        
        avg_count = sum(values) / len(values)
        threshold = avg_count * 0.5
        
        for count in values:
            if abs(count - avg_count) > threshold:
                return False
        
        return True
    
    def _update_progress(self, progress: float, message: str) -> None:
        """Update progress in UI."""
        self._progress = progress
        
        if self.ui_components and 'update_methods' in self.ui_components:
            update_methods = self.ui_components['update_methods']
            
            if 'progress' in update_methods:
                update_methods['progress'](progress, "Dataset Check")
            
            if 'activity' in update_methods:
                update_methods['activity'](message)
    
    def _update_check_results(self, results: Dict[str, Any]) -> None:
        """Update check results in UI."""
        if self.ui_components and 'update_methods' in self.ui_components:
            update_methods = self.ui_components['update_methods']
            
            # Update dataset statistics
            if 'dataset_stats' in update_methods:
                total_images = results.get('total_images', 0)
                classes_found = results.get('classes_found', 0)
                
                update_methods['dataset_stats'](
                    total_images,
                    total_images,  # Same as original for check operation
                    classes_found
                )
            
            # Update operation metrics
            if 'operation_metrics' in update_methods:
                processing_time = f"{results.get('processing_time', 0):.2f}s"
                total_images = results.get('total_images', 0)
                success_rate = 100.0 if results.get('success', False) else 0.0
                
                update_methods['operation_metrics'](
                    processing_time,
                    total_images,
                    success_rate
                )