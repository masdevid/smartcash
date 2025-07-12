"""
File: smartcash/ui/dataset/augment/operations/augment_operation_manager.py
Description: Augmentation operation manager extending OperationHandler
"""

from typing import Dict, Any, Optional, Callable
import asyncio
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from smartcash.ui.dataset.augment.constants import (
    AugmentationOperation, SUCCESS_MESSAGES, ERROR_MESSAGES
)

# Import backend augmentation services
from smartcash.ui.dataset.augment.services.augment_service import AugmentService


class AugmentOperationManager(OperationHandler):
    """
    Augmentation operation manager extending OperationHandler.
    
    Features:
    - 🎯 Extends OperationHandler for consistent architecture
    - 🎨 Backend service integration for all augmentation operations
    - 🔧 Operation container logging and progress tracking
    - 🔄 Button management with disable/enable functionality
    - 🚨 Enhanced error handling and user feedback
    """
    
    def __init__(self, config: Dict[str, Any], operation_container: Any):
        """
        Initialize augmentation operation manager.
        
        Args:
            config: Configuration dictionary
            operation_container: Operation container for logging and progress
        """
        super().__init__(
            module_name='augment',
            parent_module='dataset',
            operation_container=operation_container
        )
        
        # Store configuration
        self.config = config
        
        # Initialize backend service
        self._backend_service = AugmentService()
        
        # Track operation state
        self._current_operation = None
        self._is_processing = False
        
        # UI components reference (will be set by UIModule)
        self._ui_components = {}
    
    def initialize(self) -> None:
        """Initialize the operation manager."""
        try:
            self.log("🎨 Initializing augmentation operation manager", 'info')
            
            # Initialize backend service
            if self._backend_service:
                self._backend_service.initialize()
            
            self.log("✅ Augmentation operation manager initialized", 'info')
            
        except Exception as e:
            error_msg = f"Failed to initialize operation manager: {str(e)}"
            self.log(error_msg, 'error')
            raise RuntimeError(error_msg)
    
    def get_operations(self) -> Dict[str, Callable]:
        """
        Get available operations for this manager.
        
        Returns:
            Dictionary of operation name to callable mappings
        """
        return {
            'augment': self.execute_augment,
            'check': self.execute_check,
            'cleanup': self.execute_cleanup,
            'preview': self.execute_preview
        }
    
    def execute_augment(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute augmentation operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Operation result dictionary
        """
        if self._is_processing:
            return {'success': False, 'message': 'Another operation is already in progress'}
        
        try:
            self._is_processing = True
            self._current_operation = AugmentationOperation.AUGMENT
            
            # Use provided config or instance config
            operation_config = config or self.config
            
            self.log("🚀 Starting augmentation operation", 'info')
            
            # Disable buttons during operation
            button_states = self.disable_all_buttons("⏳ Augmenting...")
            
            try:
                # Execute via backend service
                result = self._backend_service.augment_dataset(operation_config)
                
                # Log result
                if result.get('success', False):
                    self.log(SUCCESS_MESSAGES['augmentation_complete'], 'info')
                    
                    # Log augmentation statistics
                    stats = result.get('stats', {})
                    if 'total_augmented' in stats:
                        self.log(f"🎨 Generated {stats['total_augmented']} augmented images", 'info')
                    if 'augmented_splits' in result:
                        splits = ', '.join(result['augmented_splits'])
                        self.log(f"📁 Augmented splits: {splits}", 'info')
                    if 'processing_time' in stats:
                        self.log(f"⏱️ Processing time: {stats['processing_time']:.1f}s", 'info')
                    
                    # Enable buttons with success state
                    self.enable_all_buttons(button_states, success=True)
                else:
                    error_msg = result.get('message', 'Augmentation failed')
                    self.log(f"❌ {error_msg}", 'error')
                    
                    # Enable buttons with error state
                    self.enable_all_buttons(button_states, success=False)
                
                return result
                
            except Exception as e:
                error_msg = f"Operation execution failed: {str(e)}"
                self.log(error_msg, 'error')
                
                # Enable buttons with error state
                self.enable_all_buttons(button_states, success=False)
                
                return {'success': False, 'message': error_msg}
            
        finally:
            self._is_processing = False
            self._current_operation = None
    
    def execute_check(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute dataset check operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Check result dictionary
        """
        if self._is_processing:
            return {'success': False, 'message': 'Another operation is already in progress'}
        
        try:
            self._is_processing = True
            self._current_operation = AugmentationOperation.CHECK
            
            # Use provided config or instance config
            operation_config = config or self.config
            
            self.log("🔍 Starting dataset check operation", 'info')
            
            # Disable buttons during operation
            button_states = self.disable_all_buttons("⏳ Checking...")
            
            try:
                # Execute via backend service
                result = self._backend_service.get_augmentation_status(operation_config)
                
                # Log result
                if result.get('success', False):
                    self.log(SUCCESS_MESSAGES['check_complete'], 'info')
                    
                    # Log check statistics
                    if 'service_ready' in result:
                        status = "ready" if result['service_ready'] else "not ready"
                        self.log(f"📊 Service status: {status}", 'info')
                    if 'files_found' in result:
                        self.log(f"📁 Files found: {result['files_found']}", 'info')
                    if 'augmented_count' in result:
                        self.log(f"🎨 Augmented files: {result['augmented_count']}", 'info')
                    if 'splits_available' in result:
                        splits = ', '.join(result['splits_available'])
                        self.log(f"📂 Available splits: {splits}", 'info')
                    
                    # Enable buttons with success state
                    self.enable_all_buttons(button_states, success=True)
                else:
                    error_msg = result.get('message', 'Check failed')
                    self.log(f"❌ {error_msg}", 'error')
                    
                    # Enable buttons with error state
                    self.enable_all_buttons(button_states, success=False)
                
                return result
                
            except Exception as e:
                error_msg = f"Check operation failed: {str(e)}"
                self.log(error_msg, 'error')
                
                # Enable buttons with error state
                self.enable_all_buttons(button_states, success=False)
                
                return {'success': False, 'message': error_msg}
            
        finally:
            self._is_processing = False
            self._current_operation = None
    
    def execute_cleanup(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute cleanup operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Cleanup result dictionary
        """
        if self._is_processing:
            return {'success': False, 'message': 'Another operation is already in progress'}
        
        try:
            self._is_processing = True
            self._current_operation = AugmentationOperation.CLEANUP
            
            # Use provided config or instance config
            operation_config = config or self.config
            
            self.log("🗑️ Starting cleanup operation", 'info')
            
            # Disable buttons during operation
            button_states = self.disable_all_buttons("⏳ Cleaning...")
            
            try:
                # Execute via backend service
                result = self._backend_service.cleanup_augmentation_files(operation_config)
                
                # Log result
                if result.get('success', False):
                    self.log(SUCCESS_MESSAGES['cleanup_complete'], 'info')
                    
                    # Log cleanup statistics
                    if 'files_removed' in result:
                        self.log(f"🗑️ Files removed: {result['files_removed']}", 'info')
                    if 'space_freed' in result:
                        self.log(f"💾 Space freed: {result['space_freed']}", 'info')
                    if 'cleanup_target' in result:
                        self.log(f"🎯 Target: {result['cleanup_target']}", 'info')
                    
                    # Enable buttons with success state
                    self.enable_all_buttons(button_states, success=True)
                else:
                    error_msg = result.get('message', 'Cleanup failed')
                    self.log(f"❌ {error_msg}", 'error')
                    
                    # Enable buttons with error state
                    self.enable_all_buttons(button_states, success=False)
                
                return result
                
            except Exception as e:
                error_msg = f"Cleanup operation failed: {str(e)}"
                self.log(error_msg, 'error')
                
                # Enable buttons with error state
                self.enable_all_buttons(button_states, success=False)
                
                return {'success': False, 'message': error_msg}
            
        finally:
            self._is_processing = False
            self._current_operation = None
    
    def execute_preview(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute preview operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Preview result dictionary
        """
        if self._is_processing:
            return {'success': False, 'message': 'Another operation is already in progress'}
        
        try:
            self._is_processing = True
            self._current_operation = AugmentationOperation.PREVIEW
            
            # Use provided config or instance config
            operation_config = config or self.config
            
            self.log("👁️ Starting preview operation", 'info')
            
            # Disable buttons during operation
            button_states = self.disable_all_buttons("⏳ Generating preview...")
            
            try:
                # Execute via backend service
                result = self._backend_service.generate_preview(operation_config)
                
                # Log result
                if result.get('success', False):
                    self.log(SUCCESS_MESSAGES['preview_ready'], 'info')
                    
                    # Log preview statistics
                    if 'preview_count' in result:
                        self.log(f"👁️ Generated {result['preview_count']} preview images", 'info')
                    if 'preview_path' in result:
                        self.log(f"📁 Preview saved to: {result['preview_path']}", 'info')
                    
                    # Enable buttons with success state
                    self.enable_all_buttons(button_states, success=True)
                else:
                    error_msg = result.get('message', 'Preview failed')
                    self.log(f"❌ {error_msg}", 'error')
                    
                    # Enable buttons with error state
                    self.enable_all_buttons(button_states, success=False)
                
                return result
                
            except Exception as e:
                error_msg = f"Preview operation failed: {str(e)}"
                self.log(error_msg, 'error')
                
                # Enable buttons with error state
                self.enable_all_buttons(button_states, success=False)
                
                return {'success': False, 'message': error_msg}
            
        finally:
            self._is_processing = False
            self._current_operation = None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current operation manager status.
        
        Returns:
            Status dictionary
        """
        return {
            'is_processing': self._is_processing,
            'current_operation': self._current_operation.value if self._current_operation else None,
            'backend_service_ready': self._backend_service is not None,
            'operations_available': list(self.get_operations().keys()),
            'module_name': self.module_name
        }
    
    def cancel_current_operation(self) -> bool:
        """
        Cancel the currently running operation.
        
        Returns:
            True if cancellation successful, False otherwise
        """
        if not self._is_processing:
            return False
        
        try:
            self.log("⚠️ Cancelling current operation", 'warning')
            
            # Reset processing state
            self._is_processing = False
            self._current_operation = None
            
            # Re-enable buttons
            self.enable_all_buttons({}, success=False, success_message="⚠️ Cancelled")
            
            return True
            
        except Exception as e:
            self.log(f"Failed to cancel operation: {str(e)}", 'error')
            return False
    
    def cleanup(self) -> None:
        """Cleanup operation manager resources."""
        try:
            self.log("Cleaning up augmentation operation manager", 'info')
            
            # Cancel any running operation
            if self._is_processing:
                self.cancel_current_operation()
            
            # Cleanup backend service
            if self._backend_service and hasattr(self._backend_service, 'cleanup'):
                self._backend_service.cleanup()
            
            # Clear references
            self._backend_service = None
            self._ui_components.clear()
            
            super().cleanup()
            
        except Exception as e:
            self.log(f"Error during cleanup: {str(e)}", 'error')