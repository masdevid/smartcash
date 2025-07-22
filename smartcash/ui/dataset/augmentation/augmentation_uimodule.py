"""
File: smartcash/ui/dataset/augmentation/augmentation_uimodule.py
Description: Augmentation Module implementation using BaseUIModule pattern.
"""

from typing import Dict, Any, Optional

# BaseUIModule imports
from smartcash.ui.core.base_ui_module import BaseUIModule

# Augmentation module imports
from smartcash.ui.core.decorators import suppress_ui_init_logs
from .components.augmentation_ui import create_augment_ui
from .configs.augmentation_config_handler import AugmentationConfigHandler
from .configs.augmentation_defaults import get_default_augmentation_config


class AugmentationUIModule(BaseUIModule):
    """
    Augmentation Module implementation using BaseUIModule.
    
    Features:
    - 🖼️ Image augmentation operations
    - 🎛️ Configurable augmentation pipelines
    - 🔄 Batch processing support
    - 📊 Preview functionality
    - ✅ Full compliance with BaseUIModule pattern
    - 🇮🇩 Bahasa Indonesia interface
    """
    
    def __init__(self):
        """Initialize the Augmentation UIModule."""
        super().__init__(
            module_name='augmentation',
            parent_module='dataset',
            enable_environment=True
        )
        self.log_debug("AugmentationUIModule diinisialisasi.")
        
        self._required_components = [
            'main_container',
            'header_container',
            'form_container',
            'action_container',
            'summary_container',
            'operation_container'
        ]
        
        # Initialize resources dictionary for cleanup tracking
        self._resources = {}
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this module."""
        return get_default_augmentation_config()
    
    def create_config_handler(self, config: Dict[str, Any]) -> AugmentationConfigHandler:
        """Create config handler instance for this module."""
        return AugmentationConfigHandler(default_config=config)
    
    @suppress_ui_init_logs(duration=3.0)
    def initialize(self) -> bool:
        """Initialize the Augmentation module."""
        # Initialize base and create UI components
        if not super().initialize():
            return False
        
        if not hasattr(self, '_ui_components') or not self._ui_components:
            self._ui_components = self.create_ui_components(self.get_current_config())
        
        # Set UI components in config handler
        if hasattr(self, '_config_handler') and self._config_handler:
            if hasattr(self._config_handler, 'set_ui_components'):
                self._config_handler.set_ui_components(self._ui_components)
        
        # Run post-init tasks
        if hasattr(self, '_post_init_tasks'):
            self._post_init_tasks()
        
        return True
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create and configure all UI components for this module."""
        return create_augment_ui(config=config)
    
    def create_ui(self) -> Dict[str, Any]:
        """Create UI components for testing compatibility."""
        self._ui_components = self.create_ui_components(self.get_current_config())
        return self._ui_components
    
    def _get_module_button_handlers(self) -> Dict[str, Any]:
        """Get Augmentation module-specific button handlers."""
        # Call parent method to get base handlers (save, reset)
        handlers = super()._get_module_button_handlers()
        
        # Add Augmentation-specific handlers
        augmentation_handlers = {
            'augment': self._operation_augment,
            'status': self._operation_check,  # Renamed from 'check' to 'status'
            'cleanup': self._operation_cleanup,
            'generate': self._operation_generate_preview  # Live preview button
        }
        
        handlers.update(augmentation_handlers)
        return handlers
    
    def _operation_augment(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle augment operation with confirmation dialog."""
        def validate_augment():
            return {'valid': True}  # Add validation logic here
        
        def execute_augment():
            self.log("🎨 Starting augmentation...", 'info')
            return self._execute_augment_operation()
        
        # Show confirmation dialog before augmenting
        confirmation_result = self._show_confirmation_dialog(
            title="Confirm Augmentation",
            message="Are you sure you want to proceed with data augmentation? This will add new data to the existing dataset.",
            confirm_text="Yes, Augment",
            cancel_text="Cancel"
        )
        
        if not confirmation_result:
            return {'success': False, 'message': 'Augmentation cancelled by user'}
        
        return self._execute_operation_with_wrapper(
            operation_name="Data Augmentation",
            operation_func=execute_augment,
            button=button,
            validation_func=validate_augment,
            success_message="Data augmentation completed successfully",
            error_message="Data augmentation error"
        )
    
    def _operation_check(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle check operation using common wrapper."""
        def validate_system():
            return {'valid': True}  # Check operation is always available
        
        def execute_check():
            self.log("🔍 Checking data status...", 'info')
            return self._execute_check_operation()
        
        return self._execute_operation_with_wrapper(
            operation_name="Data Check",
            operation_func=execute_check,
            button=button,
            validation_func=validate_system,
            success_message="Data check completed successfully",
            error_message="Data check error"
        )
    
    def _operation_cleanup(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle cleanup operation with confirmation dialog."""
        def validate_cleanup():
            return {'valid': True}  # Add validation logic here
        
        def execute_cleanup():
            self.log("🧹 Cleaning augmentation data...", 'info')
            return self._execute_cleanup_operation()
        
        # Show confirmation dialog before cleanup
        confirmation_result = self._show_confirmation_dialog(
            title="Confirm Cleanup",
            message="⚠️ WARNING: This will delete all existing augmentation files. This action cannot be undone. Continue?",
            confirm_text="Yes, Delete",
            cancel_text="Cancel"
        )
        
        if not confirmation_result:
            return {'success': False, 'message': 'Cleanup cancelled by user'}
        
        return self._execute_operation_with_wrapper(
            operation_name="Data Cleanup",
            operation_func=execute_cleanup,
            button=button,
            validation_func=validate_cleanup,
            success_message="Data cleanup completed successfully",
            error_message="Data cleanup error"
        )
    
    def _operation_preview(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle preview operation."""
        def validate_preview():
            return {'valid': True}  # Add validation logic here
        
        def execute_preview():
            self.log("👁️ Menampilkan pratinjau augmentasi...", 'info')
            return self._execute_preview_operation()
        
        return self._execute_operation_with_wrapper(
            operation_name="Pratinjau Augmentasi",
            operation_func=execute_preview,
            button=button,
            validation_func=validate_preview,
            success_message="Pratinjau berhasil ditampilkan",
            error_message="Kesalahan pratinjau"
        )
    
    def _operation_generate_preview(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle generate preview operation for live preview widget."""
        def validate_generate():
            return {'valid': True}  # Add validation logic here
        
        def execute_generate():
            self.log("🎬 Generating live preview...", 'info')
            result = self._execute_preview_operation()
            
            # The preview operation now handles loading the preview automatically
            # No need to explicitly call _load_existing_preview
            
            return result
        
        return self._execute_operation_with_wrapper(
            operation_name="Generate Live Preview",
            operation_func=execute_generate,
            button=button,
            validation_func=validate_generate,
            success_message="Live preview generated successfully",
            error_message="Live preview generation error"
        )
    
    # ==================== POST-INITIALIZATION METHODS ====================
    
    def _post_init_tasks(self) -> None:
        """Run post-initialization tasks like loading existing preview."""
        try:
            # Load existing preview image if available
            self._load_existing_preview()
            
            # Update preview status
            self._update_preview_status()
            
            self.log_debug("Post-initialization tasks completed successfully")
            
        except Exception as e:
            self.log_warning(f"Post-initialization tasks failed: {e}")
    
    def _load_existing_preview(self) -> None:
        """Load existing preview image using the preview operation."""
        try:
            from .operations.augment_preview_operation import AugmentPreviewOperation
            
            # Create preview operation instance
            preview_operation = AugmentPreviewOperation(
                ui_module=self,
                config=self.get_current_config(),
                callbacks={}
            )
            
            # Use the operation's integrated preview loading
            success = preview_operation.load_existing_preview()
            
            if success:
                self.log_debug("Existing preview loaded successfully via operation")
            else:
                self.log_debug("No existing preview found")
            
        except Exception as e:
            self.log_error(f"Error loading existing preview: {e}")
    
    def _update_preview_status(self) -> None:
        """Update preview status information."""
        try:
            # Get current configuration for preview settings
            config = self.get_current_config()
            
            # Update any preview-related status displays
            self.log_debug("Preview status updated")
            
        except Exception as e:
            self.log_warning(f"Failed to update preview status: {e}")
    
    def refresh_preview(self) -> None:
        """Public method to refresh the preview image."""
        try:
            self._load_existing_preview()
            self.log("Preview refreshed successfully", 'info')
        except Exception as e:
            self.log(f"Failed to refresh preview: {e}", 'error')
    
    
    # ==================== HELPER METHODS ====================
    
    def _show_confirmation_dialog(self, title: str, message: str, confirm_text: str = "Ya", cancel_text: str = "Batal") -> bool:
        """Show confirmation dialog and return user choice."""
        try:
            # Get the operation container's dialog method
            operation_container = self.get_component('operation_container')
            if operation_container and 'show_dialog' in operation_container:
                # Use the operation container's dialog functionality
                dialog_result = operation_container['show_dialog'](
                    title=title,
                    message=message,
                    on_confirm=lambda: True,
                    on_cancel=lambda: False,
                    confirm_text=confirm_text,
                    cancel_text=cancel_text
                )
                return dialog_result
            else:
                # Fallback: Log the confirmation and return True
                self.log(f"Konfirmasi: {title} - {message}", 'info')
                return True  # Default to proceed
        except Exception as e:
            self.log(f"Error in confirmation dialog: {e}", 'error')
            return True  # Default to proceed if dialog fails
    
    # ==================== OPERATION EXECUTION METHODS ====================
    
    def _execute_augment_operation(self) -> Dict[str, Any]:
        """Execute the augmentation operation using operation handler with progress updates."""
        try:
            from .operations.augment_operation import AugmentOperation
            
            # Create progress callback for backend integration
            def progress_callback(progress: float, message: str = "", phase: str = ""):
                """Progress callback for backend integration."""
                operation_container = self.get_component('operation_container')
                if operation_container and 'update_progress' in operation_container:
                    operation_container['update_progress'](
                        progress=progress,
                        message=message,
                        phase=phase
                    )
                # Don't log progress messages - they're handled by progress tracker
                # Only log phase transitions and significant events
                pass
            
            # Create handler with current UI components, config, and callbacks
            handler = AugmentOperation(
                ui_module=self,
                config=self.get_current_config(),
                callbacks={
                    'on_success': self._update_operation_summary,
                    'on_progress': progress_callback,
                    'on_phase_change': self._update_operation_phase
                }
            )
            
            # Execute the operation
            result = handler.execute()
            
            # Return standardized result
            if result and result.get('success'):
                return {'success': True, 'message': 'Augmentasi berhasil diselesaikan'}
            else:
                error_msg = result.get('message', 'Augmentasi gagal') if result else 'Augmentasi gagal'
                return {'success': False, 'message': error_msg}
            
        except Exception as e:
            return {'success': False, 'message': f"Error in augmentation operation: {e}"}
    
    def _execute_check_operation(self) -> Dict[str, Any]:
        """Execute the check operation using operation handler."""
        try:
            from .operations.augment_status_operation import AugmentStatusOperation
            
            handler = AugmentStatusOperation(
                ui_module=self,
                config=self.get_current_config(),
                callbacks={'on_success': self._update_operation_summary}
            )
            
            result = handler.execute()
            
            # Return standardized result
            if result and result.get('success'):
                return {'success': True, 'message': 'Pemeriksaan berhasil diselesaikan'}
            else:
                error_msg = result.get('message', 'Pemeriksaan gagal') if result else 'Pemeriksaan gagal'
                return {'success': False, 'message': error_msg}
            
        except Exception as e:
            return {'success': False, 'message': f"Error in check operation: {e}"}
    
    def _execute_cleanup_operation(self) -> Dict[str, Any]:
        """Execute the cleanup operation using operation handler."""
        try:
            from .operations.augment_cleanup_operation import AugmentCleanupOperation
            
            handler = AugmentCleanupOperation(
                ui_module=self,
                config=self.get_current_config(),
                callbacks={'on_success': self._update_operation_summary}
            )
            
            result = handler.execute()
            
            # Return standardized result
            if result and result.get('success'):
                return {'success': True, 'message': 'Pembersihan berhasil diselesaikan'}
            else:
                error_msg = result.get('message', 'Pembersihan gagal') if result else 'Pembersihan gagal'
                return {'success': False, 'message': error_msg}
            
        except Exception as e:
            return {'success': False, 'message': f"Error in cleanup operation: {e}"}
    
    def _execute_preview_operation(self) -> Dict[str, Any]:
        """Execute the preview operation using operation handler."""
        try:
            from .operations.augment_preview_operation import AugmentPreviewOperation
            
            handler = AugmentPreviewOperation(
                ui_module=self,
                config=self.get_current_config(),
                callbacks={'on_success': self._update_operation_summary}
            )
            
            result = handler.execute()
            
            # Return standardized result
            if result and result.get('success'):
                return {'success': True, 'message': 'Pratinjau berhasil ditampilkan'}
            else:
                error_msg = result.get('message', 'Pratinjau gagal') if result else 'Pratinjau gagal'
                return {'success': False, 'message': error_msg}
            
        except Exception as e:
            return {'success': False, 'message': f"Error in preview operation: {e}"}
    
    def _update_operation_summary(self, content: str) -> None:
        """Updates the operation summary container with new content."""
        updater = self.get_component('operation_summary_updater')
        if updater and callable(updater):
            self.log(f"Memperbarui ringkasan operasi.", 'debug')
            updater(content, title="Ringkasan Operasi", icon="📊", visible=True)
        else:
            self.log("Komponen updater ringkasan operasi tidak ditemukan atau tidak dapat dipanggil.", 'warning')
    
    def _update_operation_phase(self, phase: str, message: str = "") -> None:
        """Updates the operation phase for progress tracking."""
        try:
            operation_container = self.get_component('operation_container')
            if operation_container and 'set_phase' in operation_container:
                operation_container['set_phase'](phase, message)
                self.log(f"Phase changed to: {phase} - {message}", 'debug')
        except Exception as e:
            self.log(f"Error updating operation phase: {e}", 'warning')

    def _handle_operation_error(self, operation_name: str, error: Exception) -> None:
        """Handle operation errors with improved error reporting."""
        error_msg = f"Error in {operation_name}: {str(error)}"
        self.log_error(error_msg)
        self.update_operation_log(f"❌ {error_msg}")
        # Display error in UI if possible
        if hasattr(self, 'status_display'):
            self.status_display.value = f"❌ {error_msg}"

    def cleanup(self) -> None:
        """Widget lifecycle cleanup - optimization.md compliance."""
        try:
            # Cleanup any augmentation-specific resources
            if hasattr(self, '_resources'):
                for resource_name, resource in self._resources.items():
                    try:
                        if hasattr(resource, 'close'):
                            resource.close()
                        elif hasattr(resource, 'shutdown'):
                            resource.shutdown()
                        elif hasattr(resource, 'cleanup'):
                            resource.cleanup()
                    except Exception as e:
                        if hasattr(self, 'logger'):
                            self.logger.warning(f"Failed to clean up resource {resource_name}: {e}")
                self._resources.clear()

            # Cleanup UI components if they have cleanup methods
            if hasattr(self, '_ui_components') and self._ui_components:
                # Call component-specific cleanup if available
                if hasattr(self._ui_components, '_cleanup'):
                    self._ui_components._cleanup()

                # Close individual widgets
                for component_name, component in self._ui_components.items():
                    if hasattr(component, 'close'):
                        try:
                            component.close()
                        except Exception:
                            pass  # Ignore cleanup errors

            # Cleanup any augmentation operations
            if hasattr(self, '_operations'):
                for op_name, operation in self._operations.items():
                    try:
                        if hasattr(operation, 'cleanup'):
                            operation.cleanup()
                    except Exception as e:
                        if hasattr(self, 'logger'):
                            self.logger.warning(f"Failed to clean up operation {op_name}: {e}")

            # Call parent cleanup
            if hasattr(super(), 'cleanup'):
                super().cleanup()

            # Minimal logging for cleanup completion
            if hasattr(self, 'logger'):
                self.logger.info("Augmentation module cleanup completed")

        except Exception as e:
            # Critical errors always logged
            if hasattr(self, 'logger'):
                self.logger.error(f"Augmentation module cleanup failed: {e}")

    def __del__(self):
        """Memory management - ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during deletion
