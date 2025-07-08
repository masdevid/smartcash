"""
File: smartcash/ui/dataset/preprocess/operations/cleanup_operation.py
Description: Cleanup operation handler using real backend API
"""

from typing import Dict, Any, Optional, Callable
import asyncio
from smartcash.ui.dataset.preprocess.operations.base_preprocess_operation import BasePreprocessOperation
from smartcash.ui.dataset.preprocess.constants import (
    PreprocessingOperation, CleanupTarget, SUCCESS_MESSAGES, ERROR_MESSAGES
)


class CleanupOperation(BasePreprocessOperation):
    """
    Cleanup operation handler using real backend API.
    
    Features:
    - 🗑️ File cleanup using backend cleanup API
    - 📊 Progress tracking with backend callbacks
    - 🎯 Targeted cleanup (preprocessed, augmented, samples, both)
    - 🔄 Real backend integration without simulation
    """
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any],
                 progress_callback: Optional[Callable] = None, 
                 log_callback: Optional[Callable] = None):
        """
        Initialize cleanup operation.
        
        Args:
            ui_components: UI components dictionary
            config: Configuration dictionary
            progress_callback: Progress update callback
            log_callback: Log message callback
        """
        super().__init__(
            ui_components=ui_components,
            config=config,
            progress_callback=progress_callback,
            log_callback=log_callback
        )
        
        # Additional config references for convenience
        self.data_config = config.get('data', {})
        self.preprocessing_config = config.get('preprocessing', {})
        
        # Operation metadata
        self.operation_type = PreprocessingOperation.CLEANUP
        self.cleanup_results = {}
        self.cleanup_target = self.preprocessing_config.get('cleanup_target', CleanupTarget.PREPROCESSED.value)
        self.data_dir = self.data_config.get('dir', 'data')
        self.target_splits = self.preprocessing_config.get('target_splits', ['train', 'valid'])
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations for this handler."""
        return {
            'cleanup': self.execute
        }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute cleanup operation using real backend API.
        
        Args:
            **kwargs: Additional execution parameters
            
        Returns:
            Cleanup results dictionary
        """
        try:
            self.log_info(f"🗑️ Starting cleanup operation for target: {self.cleanup_target}")
            
            # Phase 1: Preview cleanup (20%)
            await self._preview_cleanup()
            
            # Phase 2: Execute cleanup (70%)
            await self._execute_cleanup()
            
            # Phase 3: Verify cleanup (10%)
            await self._verify_cleanup()
            
            # Mark operation as completed
            self.mark_completed()
            self.log_success(SUCCESS_MESSAGES['cleanup_complete'])
            
            return {
                'success': True,
                'operation': self.operation_type.value,
                'message': SUCCESS_MESSAGES['cleanup_complete'],
                'files_removed': self.cleanup_results.get('files_removed', 0),
                'cleanup_target': self.cleanup_target,
                'affected_splits': self.target_splits,
                'results': self.cleanup_results
            }
            
        except Exception as e:
            error_msg = f"Cleanup operation failed: {str(e)}"
            self.log_error(error_msg)
            self.mark_failed(error_msg)
            
            return {
                'success': False,
                'operation': self.operation_type.value,
                'message': error_msg,
                'error': str(e),
                'files_removed': 0
            }
    
    async def _preview_cleanup(self) -> None:
        """Preview cleanup operation using backend API."""
        self.log_info("👀 Previewing cleanup operation")
        
        try:
            from smartcash.dataset.preprocessor.api.cleanup_api import get_cleanup_preview
            
            self.update_progress(10, "Getting cleanup preview")
            
            # Get preview from backend
            preview = get_cleanup_preview(
                data_dir=self.data_dir,
                target=self.cleanup_target,
                splits=self.target_splits
            )
            
            if preview.get('success', False):
                total_files = preview.get('total_files', 0)
                total_size_mb = preview.get('total_size_mb', 0)
                
                self.log_info(f"📊 Cleanup Preview:")
                self.log_info(f"  • Target: {self.cleanup_target}")
                self.log_info(f"  • Files to remove: {total_files}")
                self.log_info(f"  • Total size: {total_size_mb:.1f} MB")
                self.log_info(f"  • Affected splits: {', '.join(self.target_splits)}")
                
                # Store preview results
                self.cleanup_results['preview'] = preview
                
                if total_files == 0:
                    self.log_info("ℹ️ No files found matching cleanup criteria")
                else:
                    self.log_success(f"✅ Preview completed: {total_files} files will be removed")
            else:
                error_msg = preview.get('message', 'Preview failed')
                self.log_warning(f"⚠️ Preview warning: {error_msg}")
                # Continue with cleanup even if preview fails
                
        except ImportError as e:
            error_msg = f"Backend cleanup module not available: {str(e)}"
            self.log_error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            self.log_warning(f"Preview failed, continuing with cleanup: {str(e)}")
        
        self.update_progress(20, "Preview completed")
    
    async def _execute_cleanup(self) -> None:
        """Execute cleanup using real backend API."""
        self.log_info("🚀 Executing cleanup operation")
        
        try:
            from smartcash.dataset.preprocessor.api.cleanup_api import cleanup_preprocessing_files
            
            # Create progress callback for backend
            def backend_progress_callback(level: str, current: int, total: int, message: str):
                if level == 'overall':
                    # Map to cleanup phase progress (20% base + 70% * progress)
                    progress = 20 + (70 * current / total) if total > 0 else 20
                    self.update_progress(progress, f"Cleanup: {message}")
                    self.log_info(f"📊 Overall progress: {current}/{total} - {message}")
                elif level == 'current':
                    self.log_info(f"  🗑️ Removing: {message}")
            
            self.update_progress(30, "Starting backend cleanup")
            
            # Execute backend cleanup
            result = cleanup_preprocessing_files(
                data_dir=self.data_dir,
                target=self.cleanup_target,
                splits=self.target_splits,
                confirm=True,  # Always confirm in operation context
                progress_callback=backend_progress_callback,
                ui_components=self.ui_components
            )
            
            if result.get('success', False):
                files_removed = result.get('files_removed', 0)
                size_freed_mb = result.get('size_freed_mb', 0)
                
                # Store cleanup results
                self.cleanup_results.update(result)
                
                # Log cleanup results
                self.log_success("✅ Cleanup executed successfully")
                self.log_info(f"📊 Cleanup Results:")
                self.log_info(f"  • Files removed: {files_removed}")
                if size_freed_mb > 0:
                    self.log_info(f"  • Space freed: {size_freed_mb:.1f} MB")
                
                # Log per-split results if available
                if 'by_split' in result:
                    for split, split_result in result['by_split'].items():
                        split_files = split_result.get('files_removed', 0)
                        if split_files > 0:
                            self.log_info(f"  • {split}: {split_files} files removed")
                
            else:
                error_msg = result.get('message', 'Backend cleanup failed')
                self.log_error(f"❌ Backend cleanup error: {error_msg}")
                raise RuntimeError(error_msg)
                
        except ImportError as e:
            error_msg = f"Backend cleanup module not available: {str(e)}"
            self.log_error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            self.log_error(f"Cleanup execution failed: {str(e)}")
            raise
        
        self.update_progress(90, "Cleanup execution completed")
    
    async def _verify_cleanup(self) -> None:
        """Verify cleanup results."""
        self.log_info("🔍 Verifying cleanup results")
        
        try:
            # Optional: Use backend to verify cleanup
            from smartcash.dataset.preprocessor.api.cleanup_api import get_cleanup_preview
            
            # Check if files are actually removed
            post_cleanup_preview = get_cleanup_preview(
                data_dir=self.data_dir,
                target=self.cleanup_target,
                splits=self.target_splits
            )
            
            if post_cleanup_preview.get('success', False):
                remaining_files = post_cleanup_preview.get('total_files', 0)
                
                if remaining_files == 0:
                    self.log_success("✅ Verification passed: All target files removed")
                else:
                    self.log_warning(f"⚠️ Verification: {remaining_files} files still remain")
                
                self.cleanup_results['verification'] = {
                    'remaining_files': remaining_files,
                    'cleanup_verified': remaining_files == 0
                }
            else:
                self.log_info("ℹ️ Verification skipped (preview not available)")
                
        except Exception as e:
            self.log_warning(f"Verification failed but cleanup completed: {str(e)}")
        
        # Simulate brief verification delay
        await asyncio.sleep(0.2)
        self.update_progress(100, "Cleanup verification completed")
    
    def get_operation_status(self) -> Dict[str, Any]:
        """
        Get current operation status.
        
        Returns:
            Status dictionary
        """
        status = super().get_operation_status()
        status.update({
            'operation_type': self.operation_type.value,
            'cleanup_target': self.cleanup_target,
            'data_directory': self.data_dir,
            'target_splits': self.target_splits,
            'files_removed': self.cleanup_results.get('files_removed', 0),
            'cleanup_summary': {
                'target': self.cleanup_target,
                'splits': self.target_splits,
                'preview_files': self.cleanup_results.get('preview', {}).get('total_files', 0),
                'actual_removed': self.cleanup_results.get('files_removed', 0)
            }
        })
        return status
    
    def cancel(self) -> bool:
        """
        Cancel the cleanup operation.
        
        Returns:
            True if cancellation was successful
        """
        try:
            self.log_warning("⚠️ Cancelling cleanup operation")
            self.mark_cancelled("Operation cancelled by user")
            return True
        except Exception as e:
            self.log_error(f"Failed to cancel operation: {e}")
            return False