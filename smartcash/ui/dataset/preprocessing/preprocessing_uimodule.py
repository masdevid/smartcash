# -*- coding: utf-8 -*-
"""
File: smartcash/ui/dataset/preprocessing/preprocessing_uimodule.py
Description: Final implementation of the Preprocessing Module using the modern BaseUIModule pattern.
"""

from typing import Dict, Any, Tuple

from smartcash.ui.core.base_ui_module import BaseUIModule
# Enhanced UI Module Factory removed - use direct instantiation

from smartcash.ui.core.decorators import suppress_ui_init_logs
from .components.preprocessing_ui import create_preprocessing_ui_components
from .configs.preprocessing_config_handler import PreprocessingConfigHandler
from .configs.preprocessing_defaults import get_default_config


class PreprocessingUIModule(BaseUIModule):
    """
    Preprocessing UI Module.
    """
    # 1. Constructor
    def __init__(self, enable_environment: bool = True):
        """
        Initialize the Preprocessing UI module.
        Args:
            enable_environment: Whether to enable environment management features
        """
        super().__init__(
            module_name='preprocessing',
            parent_module='dataset',
            enable_environment=enable_environment
        )
        self._required_components = [
            'main_container',
            'header_container',
            'form_container',
            'action_container',
            'summary_container',
            'operation_container'
        ]
        # Minimal logging for performance
        # Debug information disabled during normal operation
        
        # Initialize resources dictionary to track cleanup targets
        self._resources = {}

    # 2. Config methods
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this module."""
        return get_default_config()

    def create_config_handler(self, config: Dict[str, Any]) -> PreprocessingConfigHandler:
        """Creates a configuration handler instance."""
        return PreprocessingConfigHandler(config)

    def update_config(self, config: dict) -> None:
        """
        Update the configuration for the module and re-initialize config handler if needed.
        Always fallback to default config if input is None or missing required keys.
        """
        if not config:
            config = self.get_default_config()
        # self.log_info(f"[DEBUG] update_config called with config: {config}")
        if hasattr(self, '_config_handler') and self._config_handler:
            self._config_handler.update_config(config)
        else:
            self._config_handler = self.create_config_handler(config)
        # Optionally, update UI components if needed
        if hasattr(self, 'create_ui_components'):
            self._ui_components = self.create_ui_components(config)
        # Mark as not initialized to force re-initialization if needed
        self._is_initialized = False

    # 3. UI/component methods
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Creates the UI components for the module."""
        return create_preprocessing_ui_components(config=config)

    def _register_default_operations(self) -> None:
        """Register default operation handlers including preprocessing-specific operations."""
        super()._register_default_operations()
        # Note: Dynamic button handler registration is now handled by BaseUIModule

    def _get_module_button_handlers(self) -> Dict[str, Any]:
        """Get Preprocessing module-specific button handlers."""
        handlers = super()._get_module_button_handlers()
        preprocessing_handlers = {
            'preprocess': self._operation_preprocess,
            'check': self._operation_check,
            'cleanup': self._operation_cleanup,
        }
        handlers.update(preprocessing_handlers)
        return handlers

    def initialize(self) -> bool:
        """
        Initialize the Preprocessing UI module.
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            # Debug log removed for production
            result = super().initialize()
            return result
        except Exception as e:
            self.log_error(f"Gagal menginisialisasi modul Preprocessing: {str(e)}")
            return False

    def ensure_components_ready(self) -> bool:
        """
        Ensure all required UI components are properly initialized.
        Returns:
            bool: True if all components are ready, False otherwise
        """
        if not hasattr(self, '_ui_components') or not self._ui_components:
            self.log_warning("Komponen UI belum diinisialisasi")
            return False
        if 'operation_container' not in self._ui_components or not self._ui_components['operation_container']:
            self.log_warning("Operation container tidak ditemukan")
            return False
        operation_container = self._ui_components['operation_container']
        if not hasattr(operation_container, 'get') or 'progress_tracker' not in operation_container:
            self.log_warning("Progress tracker tidak ditemukan di operation container")
            return False
        return True

    # 4. Operation methods

    def _operation_preprocess(self, button=None) -> Dict[str, Any]:  # noqa: ARG002
        """Handle preprocessing operation with confirmation dialog and backend integration."""
        def validate_data():
            if not hasattr(self, '_ui_components') or not self._ui_components:
                return {'valid': False, 'message': "Komponen UI belum siap, silakan coba lagi"}
            return {'valid': True}
        def execute_preprocess():
            self.log_info("🔄 Memulai preprocessing data...")
            try:
                preprocessed_files, raw_images = self._get_preprocessed_data_stats()
                message = (
                    f"Anda akan memulai pra-pemrosesan data.\n\n"
                    f"- Gambar mentah terdeteksi: {raw_images}\n"
                    f"- File yang sudah diproses: {preprocessed_files}\n\n"
                    f"Proses ini mungkin menimpa file yang ada. Lanjutkan?"
                )
                op_container = self.get_component('operation_container')
                if op_container and hasattr(op_container, 'show_dialog'):
                    def on_confirm():
                        try:
                            return self._execute_preprocess_operation()
                        except Exception as e:
                            error_msg = f"Error during preprocessing operation: {e}"
                            self.log(error_msg, 'error')
                            self.error_progress(error_msg)
                            raise
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
                    self.log_warning("Dialog tidak tersedia, menjalankan preprocessing langsung")
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
            return {'valid': True}
        def execute_check():
            self.log_info("🔍 Memeriksa status data...")
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
            self.log_info("🧹 Membersihkan data preprocessing...")
            try:
                preprocessed_files, _ = self._get_preprocessed_data_stats()
                if preprocessed_files == 0:
                    op_container = self.get_component('operation_container')
                    if op_container and hasattr(op_container, 'show_dialog'):
                        op_container.show_dialog(
                            title="Tidak Ada untuk Dibersihkan",
                            message="Tidak ada file yang dihasilkan oleh proses preprocessing yang ditemukan.",
                            confirm_text="OK",
                            on_cancel=None
                        )
                    return {'success': True, 'message': 'Tidak ada file untuk dibersihkan'}
                message = (
                    f"Anda akan menghapus {preprocessed_files} file yang telah diproses.\n\n"
                    f"Tindakan ini tidak dapat diurungkan. Lanjutkan?"
                )
                op_container = self.get_component('operation_container')
                if op_container and hasattr(op_container, 'show_dialog'):
                    def on_confirm():
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
                    self.log_warning("Dialog tidak tersedia, menjalankan cleanup langsung")
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

    # 5. Utility and cleanup methods

    def _get_preprocessed_data_stats(self) -> Tuple[int, int]:
        """Gets the count of preprocessed and raw files from the backend."""
        try:
            from smartcash.dataset.preprocessor.api.preprocessing_api import get_preprocessing_status
            # self.log_info("Mengecek status dari backend...")
            status = get_preprocessing_status(config=self.get_current_config())
            if not status.get('service_ready'):
                self.log_warning("Layanan backend tidak siap, mengasumsikan tidak ada file.")
                return 0, 0
            stats = status.get('file_statistics', {}).get('train', {})
            preprocessed_files = stats.get('preprocessed_files', 0)
            raw_images = stats.get('raw_images', 0)
            # self.log_info(f"File terdeteksi: {preprocessed_files} diproses, {raw_images} mentah.")
            return preprocessed_files, raw_images
        except Exception as e:
            self.log_error(f"Tidak dapat memeriksa keberadaan file yang diproses: {e}")
            return 0, 0  # Fail safe

    def _execute_preprocess_operation(self) -> Dict[str, Any]:
        """Execute the preprocessing operation using operation handler."""
        try:
            from .operations.preprocess_operation import PreprocessOperation
            handler = PreprocessOperation(
                ui_module=self,
                config=self.get_current_config(),
                callbacks={'on_success': self._update_operation_summary}
            )
            result = handler.execute()
            if result and result.get('success'):
                return {'success': True, 'message': 'Preprocessing berhasil diselesaikan'}
            else:
                error_msg = result.get('message', 'Preprocessing gagal') if result else 'Preprocessing gagal'
                return {'success': False, 'message': error_msg}
        except Exception as e:
            return {'success': False, 'message': f"Error in preprocessing operation: {e}"}

    def _update_operation_summary(self, content: str) -> None:
        """Updates the operation summary container with new content."""
        try:
            # Get summary container from UI components
            summary_container = self.get_component('summary_container')
            if summary_container and hasattr(summary_container, 'set_content'):
                # Update summary container with backend response
                formatted_content = f"""
                <div style="padding: 10px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
                    <h4 style="color: #2c3e50; margin: 0 0 10px 0;">📊 Operation Summary</h4>
                    <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 4px solid #28a745;">
                        {content}
                    </div>
                </div>
                """
                summary_container.set_content(formatted_content)
                self.log_debug("✅ Summary container updated with operation results")
            else:
                # Fallback: try operation_summary_updater method
                updater = self.get_component('operation_summary_updater')
                if updater and callable(updater):
                    updater(content)
                else:
                    self.log_warning("Summary container tidak ditemukan atau tidak dapat dipanggil.")
        except Exception as e:
            self.log_error(f"Failed to update operation summary: {e}")

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
            if result and result.get('success'):
                return {'success': True, 'message': 'Pembersihan berhasil diselesaikan'}
            else:
                error_msg = result.get('message', 'Pembersihan gagal') if result else 'Pembersihan gagal'
                return {'success': False, 'message': error_msg}
        except Exception as e:
            return {'success': False, 'message': f"Error in cleanup operation: {e}"}

    def cleanup(self) -> None:
        """Widget lifecycle cleanup - optimization.md compliance."""
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                if hasattr(self._ui_components, '_cleanup'):
                    self._ui_components._cleanup()
                for component in self._ui_components.values():
                    if hasattr(component, 'close'):
                        try:
                            component.close()
                        except Exception:
                            pass
            if hasattr(super(), 'cleanup'):
                super().cleanup()
            # self.log_info("Preprocessing module cleanup completed")
        except Exception as e:
            self.log_error(f"Preprocessing module cleanup failed: {e}")

    def __del__(self):
        """Memory management - ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass
