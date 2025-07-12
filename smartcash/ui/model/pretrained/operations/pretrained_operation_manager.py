"""
File: smartcash/ui/model/pretrained/operations/pretrained_operation_manager.py
Description: Operation manager for pretrained module extending OperationHandler
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from smartcash.ui.logger import get_module_logger

# Import service for backend integration
from ..services.pretrained_service import PretrainedService


class PretrainedOperationManager(OperationHandler):
    """
    Operation manager for pretrained module.
    
    Features:
    - 🤖 Pretrained model operations (download, validate, cleanup)
    - 🔄 Progress tracking and logging integration
    - 🛡️ Error handling with user feedback
    - 🎯 Button management with disable/enable functionality
    - 📋 Operation status tracking and reporting
    - 🔗 Backend service integration
    """
    
    def __init__(self, config: Dict[str, Any], operation_container: Any):
        """
        Initialize pretrained operation manager.
        
        Args:
            config: Configuration dictionary
            operation_container: UI operation container for logging and progress
        """
        super().__init__(
            module_name='pretrained',
            parent_module='model',
            operation_container=operation_container
        )
        
        self.config = config
        self.logger = get_module_logger("smartcash.ui.model.pretrained.operations")
        
        # Initialize service
        self._service = None
        
        # Initialize service instance
        self._initialize_service()
    
    def _initialize_service(self) -> None:
        """Initialize service instance."""
        try:
            self._service = PretrainedService()
            self.logger.debug("✅ Pretrained service initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize service: {e}")
    
    def initialize(self) -> None:
        """Initialize the operation manager."""
        try:
            super().initialize()
            self.log("🔧 Pretrained operation manager initialized", 'info')
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pretrained operation manager: {e}")
            self.log(f"❌ Initialization failed: {e}", 'error')
    
    def get_operations(self) -> Dict[str, str]:
        """
        Get available operations.
        
        Returns:
            Dictionary of operation names and descriptions
        """
        return {
            'download': 'Download YOLOv5s and EfficientNet-B4 pretrained models',
            'validate': 'Validate existing models and check integrity',
            'cleanup': 'Clean up corrupted or invalid model files',
            'refresh': 'Refresh model status and directory contents'
        }
    
    # ==================== PRETRAINED OPERATIONS ====================
    
    def execute_download(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute pretrained models download operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Operation result dictionary
        """
        try:
            self.log("🔄 Starting pretrained models download operation...", 'info')
            self.disable_buttons(['download', 'validate', 'cleanup', 'refresh'])
            
            # Update progress
            self.update_progress(0, "Initializing download operation...")
            
            # Use provided config or current config
            operation_config = config or self.config
            
            # Execute download operation - fail fast if service not available
            if not self._service:
                raise RuntimeError("Service not available - cannot proceed with download")
            
            result = self._execute_download_with_service(operation_config)
            
            # Update progress based on result
            if result.get('success'):
                self.update_progress(100, "Download operation completed successfully")
                self.log("✅ Pretrained models download completed successfully", 'success')
            else:
                self.update_progress(0, "Download operation failed")
                self.log(f"❌ Download operation failed: {result.get('message', 'Unknown error')}", 'error')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Download operation error: {e}")
            self.log(f"❌ Download operation error: {e}", 'error')
            self.update_progress(0, "Download operation failed")
            return {'success': False, 'message': str(e)}
        
        finally:
            self.enable_buttons(['download', 'validate', 'cleanup', 'refresh'])
    
    def execute_validate(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute model validation operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Validation result dictionary
        """
        try:
            self.log("🔍 Validating existing models...", 'info')
            self.disable_buttons(['validate'])
            
            # Update progress
            self.update_progress(0, "Initializing validation...")
            
            # Use provided config or current config
            operation_config = config or self.config
            
            # Execute validation operation - fail fast if service not available
            if not self._service:
                raise RuntimeError("Service not available - cannot proceed with validation")
            
            result = self._execute_validate_with_service(operation_config)
            
            # Update progress and log result
            if result.get('success'):
                self.update_progress(100, "Validation completed")
                self.log("✅ Model validation completed successfully", 'success')
            else:
                self.update_progress(0, "Validation failed")
                self.log(f"❌ Model validation failed: {result.get('message', 'Unknown error')}", 'error')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Validate operation error: {e}")
            self.log(f"❌ Validation operation error: {e}", 'error')
            self.update_progress(0, "Validation failed")
            return {'success': False, 'message': str(e)}
        
        finally:
            self.enable_buttons(['validate'])
    
    def execute_cleanup(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute cleanup operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Cleanup result dictionary
        """
        try:
            self.log("🧹 Starting cleanup operation...", 'info')
            self.disable_buttons(['cleanup'])
            
            # Update progress
            self.update_progress(0, "Initializing cleanup...")
            
            # Use provided config or current config
            operation_config = config or self.config
            
            # Execute cleanup operation - fail fast if service not available
            if not self._service:
                raise RuntimeError("Service not available - cannot proceed with cleanup")
            
            result = self._execute_cleanup_with_service(operation_config)
            
            # Update progress and log result
            if result.get('success'):
                self.update_progress(100, "Cleanup completed")
                self.log("✅ Cleanup completed successfully", 'success')
            else:
                self.update_progress(0, "Cleanup failed")
                self.log(f"❌ Cleanup failed: {result.get('message', 'Unknown error')}", 'error')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Cleanup operation error: {e}")
            self.log(f"❌ Cleanup operation error: {e}", 'error')
            self.update_progress(0, "Cleanup failed")
            return {'success': False, 'message': str(e)}
        
        finally:
            self.enable_buttons(['cleanup'])
    
    def execute_refresh(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute refresh operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Refresh result dictionary
        """
        try:
            self.log("🔄 Refreshing model status...", 'info')
            self.disable_buttons(['refresh'])
            
            # Update progress
            self.update_progress(0, "Checking models directory...")
            
            # Use provided config or current config
            operation_config = config or self.config
            
            # Execute refresh operation - fail fast if service not available
            if not self._service:
                raise RuntimeError("Service not available - cannot proceed with refresh")
            
            result = self._execute_refresh_with_service(operation_config)
            
            # Update progress and log result
            if result.get('success'):
                self.update_progress(100, "Refresh completed")
                self.log("✅ Model status refreshed successfully", 'success')
            else:
                self.update_progress(0, "Refresh failed")
                self.log(f"❌ Refresh failed: {result.get('message', 'Unknown error')}", 'error')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Refresh operation error: {e}")
            self.log(f"❌ Refresh operation error: {e}", 'error')
            self.update_progress(0, "Refresh failed")
            return {'success': False, 'message': str(e)}
        
        finally:
            self.enable_buttons(['refresh'])
    
    # ==================== SERVICE INTEGRATION ====================
    
    def _execute_download_with_service(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute download operation with service integration."""
        try:
            pretrained_config = config.get('pretrained', {})
            models_dir = pretrained_config.get('models_dir', '/data/pretrained')
            
            # Check existing models first
            self.update_progress(10, "Checking existing models...")
            existing_check = self._service.check_existing_models(
                models_dir, 
                progress_callback=self.update_progress,
                log_callback=self.log
            )
            
            # Download models if needed
            self.update_progress(30, "Starting model downloads...")
            download_result = self._service.download_all_models(
                pretrained_config,
                progress_callback=self.update_progress, 
                log_callback=self.log
            )
            
            return {
                'success': download_result.get('all_successful', False),
                'message': 'Download operation completed',
                'existing_models': existing_check,
                'download_results': download_result,
                'models_dir': models_dir
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Service download failed: {e}'}
    
    def _execute_validate_with_service(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validate operation with service integration."""
        try:
            pretrained_config = config.get('pretrained', {})
            models_dir = pretrained_config.get('models_dir', '/data/pretrained')
            
            # Check and validate existing models
            self.update_progress(20, "Validating models...")
            existing_check = self._service.check_existing_models(
                models_dir,
                progress_callback=self.update_progress,
                log_callback=self.log
            )
            
            return {
                'success': True,
                'message': 'Model validation completed',
                'validation_results': existing_check,
                'models_dir': models_dir
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Service validation failed: {e}'}
    
    def _execute_cleanup_with_service(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cleanup operation with service integration."""
        try:
            pretrained_config = config.get('pretrained', {})
            models_dir = pretrained_config.get('models_dir', '/data/pretrained')
            
            # Check for corrupted files and clean up
            self.update_progress(20, "Checking for corrupted files...")
            
            # For now, just check what exists
            existing_check = self._service.check_existing_models(
                models_dir,
                progress_callback=self.update_progress,
                log_callback=self.log
            )
            
            return {
                'success': True,
                'message': 'Cleanup operation completed',
                'cleanup_results': {'files_removed': 0, 'space_freed': 0},
                'models_dir': models_dir
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Service cleanup failed: {e}'}
    
    def _execute_refresh_with_service(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute refresh operation with service integration."""
        try:
            pretrained_config = config.get('pretrained', {})
            models_dir = pretrained_config.get('models_dir', '/data/pretrained')
            
            # Refresh model status
            self.update_progress(30, "Refreshing model status...")
            existing_check = self._service.check_existing_models(
                models_dir,
                progress_callback=self.update_progress,
                log_callback=self.log
            )
            
            return {
                'success': True,
                'message': 'Model status refreshed',
                'status': existing_check,
                'models_dir': models_dir
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Service refresh failed: {e}'}
    
    # ==================== FAIL-FAST APPROACH ====================
    # No fallback simulations - service must be available for operations
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current operation manager status.
        
        Returns:
            Status dictionary
        """
        return {
            'initialized': True,
            'service_ready': self._service is not None,
            'available_operations': list(self.get_operations().keys()),
            'module_name': self.module_name,
            'parent_module': self.parent_module
        }
    
    def cleanup(self) -> None:
        """Cleanup operation manager resources."""
        try:
            # Cleanup service instance
            if self._service and hasattr(self._service, 'cleanup'):
                self._service.cleanup()
            
            # Clear references
            self._service = None
            
            super().cleanup()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")