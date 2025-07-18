# -*- coding: utf-8 -*-
"""
File: smartcash/ui/dataset/preprocessing/preprocessing_uimodule.py
Description: Final implementation of the Preprocessing Module using the modern BaseUIModule pattern.
"""

from typing import Dict, Any, Optional, Tuple

from smartcash.ui.core.base_ui_module import BaseUIModule

from .components.preprocessing_ui import create_preprocessing_ui_components
from .configs.preprocessing_config_handler import PreprocessingConfigHandler
from .configs.preprocessing_defaults import get_default_config


class PreprocessingUIModule(BaseUIModule):
    """
    Preprocessing UI Module.
    """

    def __init__(self):
        """Initialize the Preprocessing UI module."""
        super().__init__(
            module_name='preprocessing',
            parent_module='dataset'
        )
        self._required_components = [
            'main_container',
            'header_container',
            'form_container',
            'action_container',
            'operation_container'
        ]
        self.logger.debug("PreprocessingUIModule initialized.")

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this module."""
        return get_default_config()

    def create_config_handler(self, config: Dict[str, Any]) -> PreprocessingConfigHandler:
        """Creates a configuration handler instance."""
        return PreprocessingConfigHandler(config)

    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Creates the UI components for the module."""
        return create_preprocessing_ui_components(config=config)

    def _register_default_operations(self) -> None:
        """Register default operation handlers including preprocessing-specific operations."""
        # Call parent method to register base operations
        super()._register_default_operations()
        
        # Note: Dynamic button handler registration is now handled by BaseUIModule
    
    def _get_module_button_handlers(self) -> Dict[str, Any]:
        """Get Preprocessing module-specific button handlers."""
        # Call parent method to get base handlers (save, reset)
        handlers = super()._get_module_button_handlers()
        
        # Add Preprocessing-specific handlers
        preprocessing_handlers = {
            'preprocess': self._operation_preprocess,
            'check': self._operation_check,
            'cleanup': self._operation_cleanup
        }
        
        handlers.update(preprocessing_handlers)
        return handlers
    
    def initialize(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> bool:
        """
        Initialize the Preprocessing module.
        
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
                
                self.logger.debug("✅ Preprocessing module initialization completed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Preprocessing module: {e}")
            return False
    
    def _handle_button_click(self, handler, button_id):
        """Handle button click events with error handling."""
        try:
            self.logger.debug(f"Button clicked: {button_id}")
            handler(None)  # Pass None as the button parameter
        except Exception as e:
            self.logger.error(f"Error in button '{button_id}' handler: {str(e)}")
            # You might want to show an error message to the user here

    def _initialize_progress_display(self) -> None:
        """Initialize progress display components."""
        try:
            operation_container = self.get_component("operation_container")
            if operation_container:
                self.progress_display = operation_container.get_widget('progress_display')
                if self.progress_display:
                    self.progress_display.set_visibility(False)
        except (KeyError, AttributeError) as e:
            self.logger.warning(f"Could not initialize progress display: {e}")

    def _operation_preprocess(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle preprocessing operation with confirmation dialog and backend integration."""
        def validate_data():
            # Ensure UI components are ready first
            if not hasattr(self, '_ui_components') or not self._ui_components:
                return {'valid': False, 'message': "Komponen UI belum siap, silakan coba lagi"}
            return {'valid': True}
        
        def execute_preprocess():
            self.log("🔄 Memulai preprocessing data...", 'info')
            
            # Get current data stats from backend
            try:
                preprocessed_files, raw_images = self._get_preprocessed_data_stats()
                
                # Show confirmation dialog with current status
                message = (
                    f"Anda akan memulai pra-pemrosesan data.\n\n"
                    f"- Gambar mentah terdeteksi: {raw_images}\n"
                    f"- File yang sudah diproses: {preprocessed_files}\n\n"
                    f"Proses ini mungkin menimpa file yang ada. Lanjutkan?"
                )
                
                # Show confirmation dialog
                op_container = self.get_component('operation_container')
                if op_container and hasattr(op_container, 'show_dialog'):
                    def on_confirm():
                        # Execute actual preprocessing operation
                        return self._execute_preprocess_operation()
                    
                    op_container.show_dialog(
                        title="Konfirmasi Pra-pemrosesan",
                        message=message,
                        on_confirm=on_confirm,
                        confirm_text="Lanjutkan",
                        cancel_text="Batal",
                        danger_mode=True
                    )
                    return {'success': True, 'message': 'Dialog konfirmasi ditampilkan'}
                else:
                    # Fallback: direct execution if dialog not available
                    self.log("Dialog tidak tersedia, menjalankan preprocessing langsung", 'warning')
                    return self._execute_preprocess_operation()
                    
            except Exception as e:
                return {'success': False, 'message': f"Gagal memeriksa status data: {e}"}
        
        return self._execute_operation_with_wrapper(
            operation_name="Preprocessing Data",
            operation_func=execute_preprocess,
            button=button,
            validation_func=validate_data,
            success_message="Preprocessing data berhasil diselesaikan",
            error_message="Kesalahan preprocessing data"
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
        """Handle cleanup operation with confirmation dialog and backend integration."""
        def validate_cleanup():
            preprocessed_files, _ = self._get_preprocessed_data_stats()
            if preprocessed_files == 0:
                return {'valid': False, 'message': "Tidak ada data yang sudah diproses untuk dibersihkan"}
            return {'valid': True, 'preprocessed_files': preprocessed_files}
        
        def execute_cleanup():
            self.log("🧹 Membersihkan data preprocessing...", 'info')
            
            # Get current data stats from backend
            try:
                preprocessed_files, _ = self._get_preprocessed_data_stats()
                
                if preprocessed_files == 0:
                    # Show info dialog that there's nothing to clean
                    op_container = self.get_component('operation_container')
                    if op_container and hasattr(op_container, 'show_dialog'):
                        op_container.show_dialog(
                            title="Tidak Ada untuk Dibersihkan",
                            message="Tidak ada file yang dihasilkan oleh proses preprocessing yang ditemukan.",
                            confirm_text="OK",
                            on_cancel=None  # No cancel button
                        )
                    return {'success': True, 'message': 'Tidak ada file untuk dibersihkan'}
                
                # Show confirmation dialog for cleanup
                message = (
                    f"Anda akan menghapus {preprocessed_files} file yang telah diproses.\n\n"
                    f"Tindakan ini tidak dapat diurungkan. Lanjutkan?"
                )
                
                op_container = self.get_component('operation_container')
                if op_container and hasattr(op_container, 'show_dialog'):
                    def on_confirm():
                        # Execute actual cleanup operation
                        return self._execute_cleanup_operation()
                    
                    op_container.show_dialog(
                        title="Konfirmasi Pembersihan",
                        message=message,
                        on_confirm=on_confirm,
                        confirm_text=f"Ya, Hapus {preprocessed_files} File",
                        cancel_text="Batal",
                        danger_mode=True
                    )
                    return {'success': True, 'message': 'Dialog konfirmasi ditampilkan'}
                else:
                    # Fallback: direct execution if dialog not available
                    self.log("Dialog tidak tersedia, menjalankan cleanup langsung", 'warning')
                    return self._execute_cleanup_operation()
                    
            except Exception as e:
                return {'success': False, 'message': f"Gagal memeriksa status data: {e}"}
        
        return self._execute_operation_with_wrapper(
            operation_name="Pembersihan Data",
            operation_func=execute_cleanup,
            button=button,
            validation_func=validate_cleanup,
            success_message="Pembersihan data berhasil diselesaikan",
            error_message="Kesalahan pembersihan data"
        )

    def _run_operation(
        self, 
        operation_name: str, 
        **kwargs
    ) -> None:
        """Run a preprocessing operation."""
        if not hasattr(self, 'progress_display') or not self.progress_display:
            self._initialize_progress_display()
        super()._run_operation(operation_name, **kwargs)

    def _get_preprocessed_data_stats(self) -> Tuple[int, int]:
        """Gets the count of preprocessed and raw files from the backend."""
        try:
            from smartcash.dataset.preprocessor.api.preprocessing_api import get_preprocessing_status
            
            self.log("Mengecek status dari backend...", 'info')
            status = get_preprocessing_status(config=self.get_current_config())

            if not status.get('service_ready'):
                self.log("Layanan backend tidak siap, mengasumsikan tidak ada file.", 'warning')
                return 0, 0

            stats = status.get('file_statistics', {}).get('train', {})
            preprocessed_files = stats.get('preprocessed_files', 0)
            raw_images = stats.get('raw_images', 0)

            self.log(f"File terdeteksi: {preprocessed_files} diproses, {raw_images} mentah.", 'info')
            return preprocessed_files, raw_images

        except Exception as e:
            self.log(f"Tidak dapat memeriksa keberadaan file yang diproses: {e}", 'error')
            return 0, 0  # Fail safe

    # ==================== OPERATION EXECUTION METHODS ====================

    def _execute_preprocess_operation(self) -> Dict[str, Any]:
        """Execute the preprocessing operation using operation handler."""
        try:
            from .operations.preprocess_operation import PreprocessOperationHandler
            
            # Create handler with current UI components and config
            handler = PreprocessOperationHandler(
                ui_module=self,
                config=self.get_current_config(),
                callbacks={'on_success': self._update_operation_summary}
            )
            
            # Execute the operation
            result = handler.execute()
            
            # Return standardized result
            if result and result.get('success'):
                return {'success': True, 'message': 'Preprocessing berhasil diselesaikan'}
            else:
                error_msg = result.get('message', 'Preprocessing gagal') if result else 'Preprocessing gagal'
                return {'success': False, 'message': error_msg}
            
        except Exception as e:
            return {'success': False, 'message': f"Error in preprocessing operation: {e}"}

    def _update_operation_summary(self, content: str) -> None:
        """Updates the operation summary container with new content."""
        updater = self.get_component('operation_summary_updater')
        if updater and callable(updater):
            self.log(f"Memperbarui ringkasan operasi.", 'debug')
            updater(content, title="Ringkasan Operasi", icon="📊", visible=True)
        else:
            self.log("Komponen updater ringkasan operasi tidak ditemukan atau tidak dapat dipanggil.", 'warning')

    def _execute_check_operation(self) -> Dict[str, Any]:
        """Execute the check operation using operation handler."""
        try:
            from .operations.check_operation import CheckOperationHandler
            
            handler = CheckOperationHandler(
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
            from .operations.cleanup_operation import CleanupOperationHandler
            
            handler = CleanupOperationHandler(
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


def initialize_preprocessing_ui(display: bool = True, **kwargs: Any) -> PreprocessingUIModule:
    """
    Initializes and optionally displays the Preprocessing UI Module.

    Args:
        display: If True, the UI will be displayed in the output.
        **kwargs: Additional arguments to pass to the module.

    Returns:
        An instance of the PreprocessingUIModule.
    """
    module = PreprocessingUIModule(**kwargs)
    module.initialize()
    if display:
        module.display_ui()
    return module
