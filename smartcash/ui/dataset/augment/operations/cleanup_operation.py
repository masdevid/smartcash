"""
File: smartcash/ui/dataset/augment/operations/cleanup_operation.py
Description: Cleanup operation with preserved business logic

This operation handles augmented file cleanup with all original
business logic and safety measures preserved.
"""

from typing import Dict, Any, Optional
import logging
import os
import time
from smartcash.ui.core.decorators import handle_ui_errors
from ..constants import CleanupTarget, DEFAULT_CLEANUP_PARAMS, WARNING_MESSAGES


class CleanupOperation:
    """
    Cleanup operation with preserved business logic.
    
    Features:
    - 🗑️ Targeted file cleanup (augmented, samples, both)
    - 🛡️ Safety measures and confirmations
    - 📊 Cleanup statistics tracking
    - 🔍 Pattern-based file identification
    - ✅ Comprehensive error handling
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """
        Initialize cleanup operation.
        
        Args:
            ui_components: UI components for progress updates
        """
        self.ui_components = ui_components or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Operation state
        self._progress = 0.0
        self._is_cancelled = False
        
        # Cleanup results
        self._files_deleted = 0
        self._directories_removed = 0
        self._space_freed = 0
        self._cleanup_errors = []
        
        self.logger.debug("🗑️ CleanupOperation initialized")
    
    @handle_ui_errors(error_component_title="Cleanup Operation Error")
    def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute cleanup operation with preserved business logic.
        
        Args:
            config: Cleanup configuration
            
        Returns:
            Dictionary containing cleanup results
        """
        start_time = time.time()
        self.logger.info("🗑️ Starting cleanup operation")
        
        try:
            # Reset state
            self._reset_state()
            
            # Get cleanup configuration
            cleanup_config = config.get('cleanup', DEFAULT_CLEANUP_PARAMS)
            data_dir = config.get('data', {}).get('dir', 'data')
            target = cleanup_config.get('default_target', 'both')
            
            # Show warning
            self._update_progress(0.0, f"⚠️ {WARNING_MESSAGES['cleanup_warning']}")
            
            # Validate target
            if not self._validate_cleanup_target(target):
                return {
                    'success': False,
                    'error': f"Invalid cleanup target: {target}"
                }
            
            # Execute cleanup based on target
            if target == CleanupTarget.AUGMENTED.value:
                result = self._cleanup_augmented_files(data_dir, cleanup_config)
            elif target == CleanupTarget.SAMPLES.value:
                result = self._cleanup_sample_files(data_dir, cleanup_config)
            elif target == CleanupTarget.BOTH.value:
                result = self._cleanup_both_types(data_dir, cleanup_config)
            else:
                return {
                    'success': False,
                    'error': f"Unknown cleanup target: {target}"
                }
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Compile final results
            final_result = {
                'success': result['success'],
                'processing_time': processing_time,
                'cleanup_target': target,
                'files_deleted': self._files_deleted,
                'directories_removed': self._directories_removed,
                'space_freed_mb': self._space_freed / (1024 * 1024),
                'cleanup_errors': self._cleanup_errors,
                'operation_details': result
            }
            
            # Update UI with results
            self._update_cleanup_results(final_result)
            
            self.logger.info(f"✅ Cleanup completed: {self._files_deleted} files deleted")
            return final_result
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def cancel(self) -> None:
        """Cancel the cleanup operation."""
        self._is_cancelled = True
        self.logger.info("🛑 Cleanup operation cancelled")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current operation status."""
        return {
            'progress': self._progress,
            'is_cancelled': self._is_cancelled,
            'files_deleted': self._files_deleted,
            'directories_removed': self._directories_removed,
            'cleanup_errors': len(self._cleanup_errors)
        }
    
    def _reset_state(self) -> None:
        """Reset operation state."""
        self._progress = 0.0
        self._is_cancelled = False
        self._files_deleted = 0
        self._directories_removed = 0
        self._space_freed = 0
        self._cleanup_errors = []
    
    def _validate_cleanup_target(self, target: str) -> bool:
        """Validate cleanup target."""
        valid_targets = [t.value for t in CleanupTarget]
        return target in valid_targets
    
    def _cleanup_augmented_files(self, data_dir: str, cleanup_config: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up augmented files only."""
        self._update_progress(0.1, "Scanning for augmented files...")
        
        try:
            # Get augmented file patterns
            targets_config = cleanup_config.get('targets', {})
            augmented_config = targets_config.get('augmented', {})
            patterns = augmented_config.get('patterns', ['aug_*'])
            
            cleaned_files = self._cleanup_files_by_patterns(data_dir, patterns, "augmented")
            
            self._update_progress(1.0, f"Cleaned {cleaned_files} augmented files")
            
            return {
                'success': True,
                'cleaned_files': cleaned_files,
                'cleanup_type': 'augmented'
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Augmented cleanup error: {str(e)}"}
    
    def _cleanup_sample_files(self, data_dir: str, cleanup_config: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up sample files only."""
        self._update_progress(0.1, "Scanning for sample files...")
        
        try:
            # Get sample file patterns
            targets_config = cleanup_config.get('targets', {})
            sample_config = targets_config.get('samples', {})
            patterns = sample_config.get('patterns', ['sample_aug_*'])
            preserve_originals = sample_config.get('preserve_originals', True)
            
            cleaned_files = self._cleanup_files_by_patterns(data_dir, patterns, "samples")
            
            self._update_progress(1.0, f"Cleaned {cleaned_files} sample files")
            
            return {
                'success': True,
                'cleaned_files': cleaned_files,
                'cleanup_type': 'samples',
                'preserved_originals': preserve_originals
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Sample cleanup error: {str(e)}"}
    
    def _cleanup_both_types(self, data_dir: str, cleanup_config: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up both augmented and sample files."""
        self._update_progress(0.1, "Scanning for all cleanup files...")
        
        try:
            # Get configuration for both types
            targets_config = cleanup_config.get('targets', {})
            sequential = targets_config.get('both', {}).get('sequential', True)
            
            total_cleaned = 0
            
            if sequential:
                # Clean augmented files first
                self._update_progress(0.2, "Cleaning augmented files...")
                augmented_result = self._cleanup_augmented_files(data_dir, cleanup_config)
                if not augmented_result['success']:
                    return augmented_result
                total_cleaned += augmented_result.get('cleaned_files', 0)
                
                # Then clean sample files
                self._update_progress(0.6, "Cleaning sample files...")
                sample_result = self._cleanup_sample_files(data_dir, cleanup_config)
                if not sample_result['success']:
                    return sample_result
                total_cleaned += sample_result.get('cleaned_files', 0)
            else:
                # Clean both types simultaneously
                augmented_patterns = targets_config.get('augmented', {}).get('patterns', ['aug_*'])
                sample_patterns = targets_config.get('samples', {}).get('patterns', ['sample_aug_*'])
                all_patterns = augmented_patterns + sample_patterns
                
                total_cleaned = self._cleanup_files_by_patterns(data_dir, all_patterns, "both")
            
            self._update_progress(1.0, f"Cleaned {total_cleaned} files total")
            
            return {
                'success': True,
                'cleaned_files': total_cleaned,
                'cleanup_type': 'both',
                'sequential': sequential
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Combined cleanup error: {str(e)}"}
    
    def _cleanup_files_by_patterns(self, data_dir: str, patterns: list, cleanup_type: str) -> int:
        """Clean up files matching specified patterns."""
        cleaned_count = 0
        
        try:
            # Check each split directory
            splits = ['train', 'valid', 'test']
            total_splits = len(splits)
            
            for i, split in enumerate(splits):
                if self._is_cancelled:
                    break
                
                split_path = os.path.join(data_dir, split)
                if not os.path.exists(split_path):
                    continue
                
                # Check each class directory
                for class_name in os.listdir(split_path):
                    class_path = os.path.join(split_path, class_name)
                    if not os.path.isdir(class_path):
                        continue
                    
                    # Check each file
                    for file_name in os.listdir(class_path):
                        if self._is_cancelled:
                            break
                        
                        # Check if file matches any pattern
                        if self._matches_patterns(file_name, patterns):
                            file_path = os.path.join(class_path, file_name)
                            
                            try:
                                # Get file size before deletion
                                file_size = os.path.getsize(file_path)
                                
                                # Delete file
                                os.remove(file_path)
                                
                                # Update statistics
                                cleaned_count += 1
                                self._files_deleted += 1
                                self._space_freed += file_size
                                
                                self.logger.debug(f"🗑️ Deleted {file_path}")
                                
                            except Exception as e:
                                error_msg = f"Failed to delete {file_path}: {str(e)}"
                                self._cleanup_errors.append(error_msg)
                                self.logger.warning(error_msg)
                
                # Update progress
                progress = 0.2 + ((i + 1) / total_splits) * 0.7
                self._update_progress(progress, f"Processed {split} split - {cleanup_type}")
                
                # Clean empty directories if configured
                if DEFAULT_CLEANUP_PARAMS.get('cleanup_empty_dirs', True):
                    self._cleanup_empty_directories(split_path)
            
            return cleaned_count
            
        except Exception as e:
            error_msg = f"Pattern cleanup error: {str(e)}"
            self._cleanup_errors.append(error_msg)
            self.logger.error(error_msg)
            return cleaned_count
    
    def _matches_patterns(self, filename: str, patterns: list) -> bool:
        """Check if filename matches any of the patterns."""
        import fnmatch
        
        for pattern in patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False
    
    def _cleanup_empty_directories(self, base_path: str) -> None:
        """Clean up empty directories."""
        try:
            for root, dirs, files in os.walk(base_path, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        # Check if directory is empty
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                            self._directories_removed += 1
                            self.logger.debug(f"🗑️ Removed empty directory {dir_path}")
                    except Exception as e:
                        error_msg = f"Failed to remove directory {dir_path}: {str(e)}"
                        self._cleanup_errors.append(error_msg)
                        self.logger.warning(error_msg)
        except Exception as e:
            error_msg = f"Directory cleanup error: {str(e)}"
            self._cleanup_errors.append(error_msg)
            self.logger.error(error_msg)
    
    def _update_progress(self, progress: float, message: str) -> None:
        """Update progress in UI."""
        self._progress = progress
        
        if self.ui_components and 'update_methods' in self.ui_components:
            update_methods = self.ui_components['update_methods']
            
            if 'progress' in update_methods:
                update_methods['progress'](progress, "Cleanup")
            
            if 'activity' in update_methods:
                update_methods['activity'](message)
    
    def _update_cleanup_results(self, results: Dict[str, Any]) -> None:
        """Update cleanup results in UI."""
        if self.ui_components and 'update_methods' in self.ui_components:
            update_methods = self.ui_components['update_methods']
            
            # Update operation metrics
            if 'operation_metrics' in update_methods:
                processing_time = f"{results.get('processing_time', 0):.2f}s"
                files_deleted = results.get('files_deleted', 0)
                success_rate = 100.0 if results.get('success', False) else 0.0
                
                update_methods['operation_metrics'](
                    processing_time,
                    files_deleted,
                    success_rate
                )