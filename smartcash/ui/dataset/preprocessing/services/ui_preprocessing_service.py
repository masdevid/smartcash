"""
File: smartcash/ui/dataset/preprocessing/services/ui_preprocessing_service.py
Deskripsi: UI wrapper service untuk preprocessing dengan integrated progress tracking
"""

from typing import Dict, Any, Optional
from smartcash.dataset.preprocessor.utils.preprocessing_factory import PreprocessingFactory
from smartcash.ui.dataset.preprocessing.utils.progress_bridge import create_preprocessing_progress_bridge

class UIPreprocessingService:
    """UI wrapper service untuk preprocessing operations dengan seamless integration."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
        self.progress_bridge = create_preprocessing_progress_bridge(ui_components)
        
        # Cache untuk service instances
        self._preprocessing_manager = None
        self._dataset_checker = None
        self._cleanup_executor = None
    
    def get_preprocessing_manager(self, config: Dict[str, Any]):
        """Get preprocessing manager dengan caching dan progress integration."""
        if not self._preprocessing_manager:
            self._preprocessing_manager = PreprocessingFactory.create_preprocessing_manager(
                config, self.logger, self.progress_bridge.notify_progress
            )
        return self._preprocessing_manager
    
    def get_dataset_checker(self, config: Dict[str, Any]):
        """Get dataset checker dengan caching."""
        if not self._dataset_checker:
            self._dataset_checker = PreprocessingFactory.create_dataset_checker(config, self.logger)
        return self._dataset_checker
    
    def get_cleanup_executor(self, config: Dict[str, Any]):
        """Get cleanup executor dengan caching dan progress integration."""
        if not self._cleanup_executor:
            self._cleanup_executor = PreprocessingFactory.create_cleanup_executor(
                config, self.logger, self._create_cleanup_progress_callback()
            )
        return self._cleanup_executor
    
    def coordinate_preprocessing(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Coordinate preprocessing dengan UI progress integration."""
        try:
            self._show_operation_progress('preprocessing')
            
            manager = self.get_preprocessing_manager(config)
            result = manager.coordinate_preprocessing(**kwargs)
            
            if result['success']:
                self._handle_operation_success('preprocessing', result)
            else:
                self._handle_operation_error('preprocessing', result['message'])
            
            return result
            
        except Exception as e:
            self._handle_operation_error('preprocessing', str(e))
            return {'success': False, 'message': str(e)}
    
    def check_dataset(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check dataset dengan UI integration."""
        try:
            checker = self.get_dataset_checker(config)
            
            source_result = checker.check_source_dataset(detailed=True)
            preprocessed_result = checker.check_preprocessed_dataset(detailed=True)
            
            return {
                'success': True,
                'source': source_result,
                'preprocessed': preprocessed_result
            }
            
        except Exception as e:
            self._handle_operation_error('check', str(e))
            return {'success': False, 'message': str(e)}
    
    def cleanup_dataset(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Cleanup dataset dengan UI progress integration."""
        try:
            self._show_operation_progress('cleanup')
            
            executor = self.get_cleanup_executor(config)
            result = executor.cleanup_preprocessed_data(safe_mode=True)
            
            if result['success']:
                self._handle_operation_success('cleanup', result)
            else:
                self._handle_operation_error('cleanup', result['message'])
            
            return result
            
        except Exception as e:
            self._handle_operation_error('cleanup', str(e))
            return {'success': False, 'message': str(e)}
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status semua services."""
        return {
            'preprocessing_manager_ready': self._preprocessing_manager is not None,
            'dataset_checker_ready': self._dataset_checker is not None,
            'cleanup_executor_ready': self._cleanup_executor is not None,
            'progress_bridge_ready': self.progress_bridge is not None,
            'ui_components_available': len(self.ui_components) > 0
        }
    
    def cleanup_service_cache(self) -> None:
        """Cleanup service cache untuk fresh start."""
        if self._preprocessing_manager:
            self._preprocessing_manager.cleanup_preprocessing_state()
        
        self._preprocessing_manager = None
        self._dataset_checker = None
        self._cleanup_executor = None
        
        self.logger and self.logger.debug("🧹 UI service cache cleaned up")
    
    def _show_operation_progress(self, operation: str) -> None:
        """Show progress untuk operation."""
        show_fn = self.ui_components.get('show_for_operation')
        if show_fn:
            show_fn(operation)
    
    def _handle_operation_success(self, operation: str, result: Dict[str, Any]) -> None:
        """Handle successful operation completion."""
        completion_fn = self.ui_components.get('handle_service_completion')
        if completion_fn:
            message = self._format_success_message(operation, result)
            completion_fn(operation, message)
    
    def _handle_operation_error(self, operation: str, error_message: str) -> None:
        """Handle operation error."""
        error_fn = self.ui_components.get('handle_service_error')
        if error_fn:
            error_fn(operation, error_message)
    
    def _format_success_message(self, operation: str, result: Dict[str, Any]) -> str:
        """Format success message berdasarkan operation dan result."""
        if operation == 'preprocessing':
            total = result.get('total_images', 0)
            time_taken = result.get('processing_time', 0)
            return f"{total:,} gambar diproses dalam {time_taken:.1f}s"
        elif operation == 'cleanup':
            files = result.get('stats', {}).get('files_removed', 0)
            return f"{files:,} file dihapus"
        else:
            return "operasi berhasil"
    
    def _create_cleanup_progress_callback(self):
        """Create progress callback khusus untuk cleanup operations."""
        def cleanup_callback(**kwargs):
            progress = kwargs.get('progress', 0)
            message = kwargs.get('message', 'Cleaning up...')
            
            # Map ke overall progress untuk cleanup
            update_fn = self.ui_components.get('update_progress')
            if update_fn:
                update_fn('overall', progress, message)
        
        return cleanup_callback

def create_ui_preprocessing_service(ui_components: Dict[str, Any]) -> UIPreprocessingService:
    """Factory untuk membuat UI preprocessing service."""
    return UIPreprocessingService(ui_components)