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
            parent_module='dataset'
        )
        self.logger.debug("AugmentationUIModule diinisialisasi.")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this module."""
        return get_default_augmentation_config()
    
    def create_config_handler(self, config: Dict[str, Any]) -> AugmentationConfigHandler:
        """Create config handler instance for this module."""
        return AugmentationConfigHandler(default_config=config)
    
    @suppress_ui_init_logs(duration=3.0)
    def initialize(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """
        Initialize the Augmentation module.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional initialization arguments
            
        Returns:
            True if initialization was successful
        """
        try:
            # Set config if provided before initialization
            if config:
                self._user_config = config
            
            # Initialize using base class which handles everything
            success = super().initialize()
            
            if success:
                # Set UI components in config handler for extraction
                if self._config_handler and hasattr(self._config_handler, 'set_ui_components'):
                    self._config_handler.set_ui_components(self._ui_components)
                
                # Run post-initialization tasks
                self._post_init_tasks()
                
                self.logger.debug("✅ Augmentation module initialization completed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Augmentation module: {e}")
            return False
    
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create UI components for this module."""
        return create_augment_ui(config=config)
    
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
            self.log("🎨 Memulai operasi augmentasi...", 'info')
            return self._execute_augment_operation()
        
        # Show confirmation dialog before augmenting
        confirmation_result = self._show_confirmation_dialog(
            title="Konfirmasi Augmentasi",
            message="Apakah Anda yakin ingin melanjutkan proses augmentasi? Operasi ini akan menambahkan data baru ke dataset existing.",
            confirm_text="Ya, Augmentasi",
            cancel_text="Batal"
        )
        
        if not confirmation_result:
            return {'success': False, 'message': 'Operasi augmentasi dibatalkan oleh pengguna'}
        
        return self._execute_operation_with_wrapper(
            operation_name="Augmentasi Data",
            operation_func=execute_augment,
            button=button,
            validation_func=validate_augment,
            success_message="Augmentasi data berhasil diselesaikan",
            error_message="Kesalahan augmentasi data"
        )
    
    def _operation_check(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle check operation using common wrapper."""
        def validate_system():
            return {'valid': True}  # Check operation is always available
        
        def execute_check():
            self.log("🔍 Memeriksa status data...", 'info')
            return self._execute_check_operation()
        
        return self._execute_operation_with_wrapper(
            operation_name="Pemeriksaan Data",
            operation_func=execute_check,
            button=button,
            validation_func=validate_system,
            success_message="Pemeriksaan data berhasil diselesaikan",
            error_message="Kesalahan pemeriksaan data"
        )
    
    def _operation_cleanup(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle cleanup operation with confirmation dialog."""
        def validate_cleanup():
            return {'valid': True}  # Add validation logic here
        
        def execute_cleanup():
            self.log("🧹 Membersihkan data augmentasi...", 'info')
            return self._execute_cleanup_operation()
        
        # Show confirmation dialog before cleanup
        confirmation_result = self._show_confirmation_dialog(
            title="Konfirmasi Pembersihan",
            message="⚠️ PERHATIAN: Operasi ini akan menghapus semua file augmentasi yang ada. Tindakan ini tidak dapat dibatalkan. Apakah Anda yakin ingin melanjutkan?",
            confirm_text="Ya, Hapus",
            cancel_text="Batal"
        )
        
        if not confirmation_result:
            return {'success': False, 'message': 'Operasi pembersihan dibatalkan oleh pengguna'}
        
        return self._execute_operation_with_wrapper(
            operation_name="Pembersihan Data",
            operation_func=execute_cleanup,
            button=button,
            validation_func=validate_cleanup,
            success_message="Pembersihan data berhasil diselesaikan",
            error_message="Kesalahan pembersihan data"
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
            self.log("🎬 Menghasilkan pratinjau live...", 'info')
            result = self._execute_preview_operation()
            
            # The preview operation now handles loading the preview automatically
            # No need to explicitly call _load_existing_preview
            
            return result
        
        return self._execute_operation_with_wrapper(
            operation_name="Generate Live Preview",
            operation_func=execute_generate,
            button=button,
            validation_func=validate_generate,
            success_message="Live preview berhasil dibuat",
            error_message="Kesalahan pembuatan live preview"
        )
    
    # ==================== POST-INITIALIZATION METHODS ====================
    
    def _post_init_tasks(self) -> None:
        """Run post-initialization tasks like loading existing preview."""
        try:
            # Load existing preview image if available
            self._load_existing_preview()
            
            # Update preview status
            self._update_preview_status()
            
            self.logger.debug("Post-initialization tasks completed successfully")
            
        except Exception as e:
            self.logger.warning(f"Post-initialization tasks failed: {e}")
    
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
                self.logger.debug("Existing preview loaded successfully via operation")
            else:
                self.logger.debug("No existing preview found")
            
        except Exception as e:
            self.logger.error(f"Error loading existing preview: {e}")
    
    def _update_preview_status(self) -> None:
        """Update preview status information."""
        try:
            # Get current configuration for preview settings
            config = self.get_current_config()
            
            # Update any preview-related status displays
            self.logger.debug("Preview status updated")
            
        except Exception as e:
            self.logger.warning(f"Failed to update preview status: {e}")
    
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
                    dialog_type='confirm',
                    buttons=[
                        {'text': confirm_text, 'style': 'primary', 'value': True},
                        {'text': cancel_text, 'style': 'secondary', 'value': False}
                    ]
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
                self.log(f"Progress: {progress:.1f}% - {message}", 'info')
            
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


def initialize_augmentation_ui(display: bool = True) -> AugmentationUIModule:
    """
    Initializes and optionally displays the Augmentation UI Module.

    Args:
        display: If True, the UI will be displayed in the output.

    Returns:
        An instance of the AugmentationUIModule.
    """
    module = AugmentationUIModule()
    module.initialize()
    if display:
        module.display_ui()
    return module