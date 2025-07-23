"""
File: smartcash/ui/dataset/preprocessing/preprocessing_uimodule.py
Description: Refactored Preprocessing UI Module - eliminated overlaps and redundancies.
"""

from typing import Dict, Any, Tuple

from smartcash.ui.core.base_ui_module import BaseUIModule
from .components.preprocessing_ui import create_preprocessing_ui_components
from .configs.preprocessing_config_handler import PreprocessingConfigHandler
from .configs.preprocessing_defaults import get_default_config


class PreprocessingUIModule(BaseUIModule):
    """
    Streamlined Preprocessing UI Module.
    
    This refactored version eliminates redundancies in initialization flow,
    config handling, and operation management.
    """

    def __init__(self, enable_environment: bool = True):
        """Initialize the Preprocessing UI module."""
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
        # Service readiness flag
        self._service_ready = False
        self._existing_data_found = False

    # BaseUIModule Required Methods
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this module."""
        return get_default_config()

    def create_config_handler(self, config: Dict[str, Any]) -> PreprocessingConfigHandler:
        """Create a configuration handler instance."""
        return PreprocessingConfigHandler(config)

    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create the UI components for the module."""
        return create_preprocessing_ui_components(config=config)

    def _post_init_tasks(self) -> None:
        """Run post-initialization tasks including service readiness check."""
        try:
            # Check service readiness using the new API
            from smartcash.dataset.preprocessor.api import check_service_readiness, check_existing_data
            
            config = self.get_current_config()
            data_dir = config.get('data', {}).get('dir', 'data')
            
            # Check service readiness
            readiness_result = check_service_readiness(data_dir)
            if readiness_result.get('success'):
                self._service_ready = readiness_result.get('ready', False)
                
                # Check for existing data if service is ready
                if self._service_ready:
                    existing_data_result = check_existing_data(data_dir)
                    if existing_data_result.get('success'):
                        self._existing_data_found = existing_data_result.get('has_existing_data', False)
                        
                        # Log findings
                        if self._existing_data_found:
                            total_files = existing_data_result.get('total_existing_files', 0)
                            self.log_info(f"🔍 Service ready with {total_files} existing preprocessed files found")
                        else:
                            self.log_info("🔍 Service ready, no existing preprocessed data found")
                    else:
                        self.log_warning("⚠️ Could not check for existing data")
                else:
                    self.log_info("ℹ️ Service not ready - preprocessed directory structure missing")
            else:
                self.log_warning(f"⚠️ Service readiness check failed: {readiness_result.get('message', 'Unknown error')}")
                
        except Exception as e:
            self.log_error(f"❌ Post-init service readiness check failed: {e}")
            self._service_ready = False
            self._existing_data_found = False

    def _get_module_button_handlers(self) -> Dict[str, Any]:
        """Get module-specific button handlers."""
        base_handlers = super()._get_module_button_handlers()
        preprocessing_handlers = {
            'preprocess': self._operation_preprocess,
            'check': self._operation_check,
            'cleanup': self._operation_cleanup,
        }
        base_handlers.update(preprocessing_handlers)
        return base_handlers

    # Operation Handlers
    def _operation_preprocess(self, button=None) -> Dict[str, Any]:
        """Handle preprocessing operation with confirmation dialog."""
        return self._execute_operation_with_wrapper(
            operation_name="Preprocessing Data",
            operation_func=self._execute_preprocess_with_confirmation,
            button=button,
            validation_func=self._validate_ui_ready,
            success_message="Preprocessing data berhasil diselesaikan",
            error_message="Kesalahan preprocessing data"
        )

    def _operation_check(self, button=None) -> Dict[str, Any]:
        """Handle check operation."""
        return self._execute_operation_with_wrapper(
            operation_name="Pemeriksaan Data",
            operation_func=self._execute_check_operation,
            button=button,
            validation_func=lambda: {'valid': True},
            success_message="Pemeriksaan data berhasil diselesaikan",
            error_message="Kesalahan pemeriksaan data"
        )

    def _operation_cleanup(self, button=None) -> Dict[str, Any]:
        """Handle cleanup operation with confirmation dialog."""
        return self._execute_operation_with_wrapper(
            operation_name="Pembersihan Data",
            operation_func=self._execute_cleanup_with_confirmation,
            button=button,
            validation_func=self._validate_cleanup_available,
            success_message="Pembersihan data berhasil diselesaikan",
            error_message="Kesalahan pembersihan data"
        )

    # Private Operation Execution Methods
    def _execute_preprocess_with_confirmation(self) -> Dict[str, Any]:
        """Execute preprocessing with user confirmation, including existing data check."""
        try:
            preprocessed_files, raw_images = self._get_data_stats()
            
            # Build message based on existing data status
            base_message = (
                f"Anda akan memulai pra-pemrosesan data.\n\n"
                f"- Gambar mentah terdeteksi: {raw_images}\n"
                f"- File yang sudah diproses: {preprocessed_files}\n\n"
            )
            
            # Add warning if existing data found
            if self._existing_data_found and preprocessed_files > 0:
                message = (
                    base_message +
                    f"⚠️ PERHATIAN: Ditemukan {preprocessed_files} file yang sudah diproses!\n"
                    f"Proses ini akan menimpa file yang ada dan tidak dapat diurungkan.\n\n"
                    f"Yakin ingin melanjutkan?"
                )
                danger_mode = True
                confirm_text = "Ya, Timpa Data yang Ada"
            else:
                message = base_message + "Lanjutkan dengan pra-pemrosesan?"
                danger_mode = False
                confirm_text = "Lanjutkan"
            
            return self._show_confirmation_dialog(
                title="Konfirmasi Pra-pemrosesan",
                message=message,
                confirm_action=self._execute_preprocess_operation,
                confirm_text=confirm_text,
                danger_mode=danger_mode
            )
        except Exception as e:
            return {'success': False, 'message': f"Gagal memeriksa status data: {e}"}

    def _execute_cleanup_with_confirmation(self) -> Dict[str, Any]:
        """Execute cleanup with user confirmation."""
        try:
            preprocessed_files, _ = self._get_data_stats()
            
            if preprocessed_files == 0:
                return self._show_info_dialog(
                    title="Tidak Ada untuk Dibersihkan",
                    message="Tidak ada file yang dihasilkan oleh proses preprocessing yang ditemukan."
                )
            
            message = (
                f"Anda akan menghapus {preprocessed_files} file yang telah diproses.\n\n"
                f"Tindakan ini tidak dapat diurungkan. Lanjutkan?"
            )
            
            return self._show_confirmation_dialog(
                title="Konfirmasi Pembersihan",
                message=message,
                confirm_action=self._execute_cleanup_operation,
                confirm_text=f"Ya, Hapus {preprocessed_files} File",
                danger_mode=True
            )
        except Exception as e:
            return {'success': False, 'message': f"Gagal memeriksa status data: {e}"}

    def _execute_preprocess_operation(self) -> Dict[str, Any]:
        """Execute the actual preprocessing operation."""
        try:
            from .operations.preprocess_operation import PreprocessOperation
            
            handler = PreprocessOperation(
                ui_module=self,
                config=self.get_current_config(),
                callbacks={'on_success': self._update_operation_summary}
            )
            
            result = handler.execute()
            return self._format_operation_result(result, 'Preprocessing')
            
        except Exception as e:
            return {'success': False, 'message': f"Error in preprocessing operation: {e}"}

    def _execute_check_operation(self) -> Dict[str, Any]:
        """Execute the check operation."""
        try:
            from .operations.check_operation import CheckOperationHandler
            
            handler = CheckOperationHandler(
                ui_module=self,
                config=self.get_current_config(),
                callbacks={'on_success': self._update_operation_summary}
            )
            
            result = handler.execute()
            return self._format_operation_result(result, 'Pemeriksaan')
            
        except Exception as e:
            return {'success': False, 'message': f"Error in check operation: {e}"}

    def _execute_cleanup_operation(self) -> Dict[str, Any]:
        """Execute the cleanup operation."""
        try:
            from .operations.cleanup_operation import CleanupOperationHandler
            
            handler = CleanupOperationHandler(
                ui_module=self,
                config=self.get_current_config(),
                callbacks={'on_success': self._update_operation_summary}
            )
            
            result = handler.execute()
            return self._format_operation_result(result, 'Pembersihan')
            
        except Exception as e:
            return {'success': False, 'message': f"Error in cleanup operation: {e}"}

    # Validation Methods
    def _validate_ui_ready(self) -> Dict[str, Any]:
        """Validate that UI components are ready and service is ready."""
        if not hasattr(self, '_ui_components') or not self._ui_components:
            return {'valid': False, 'message': "Komponen UI belum siap, silakan coba lagi"}
        
        if not self._service_ready:
            return {
                'valid': False, 
                'message': "Service belum siap - struktur direktori preprocessing tidak ditemukan. Pastikan direktori data telah dibuat dengan benar."
            }
        
        return {'valid': True}

    def _validate_cleanup_available(self) -> Dict[str, Any]:
        """Validate that cleanup is available and service is ready."""
        if not self._service_ready:
            return {
                'valid': False, 
                'message': "Service belum siap - tidak dapat melakukan cleanup. Pastikan direktori data telah dibuat dengan benar."
            }
            
        try:
            preprocessed_files, _ = self._get_data_stats()
            if preprocessed_files == 0:
                return {'valid': False, 'message': "Tidak ada data yang sudah diproses untuk dibersihkan"}
            return {'valid': True, 'preprocessed_files': preprocessed_files}
        except Exception as e:
            return {'valid': False, 'message': f"Gagal memeriksa status data: {e}"}

    # Helper Methods
    def _get_data_stats(self) -> Tuple[int, int]:
        """Get preprocessed and raw file counts from backend."""
        try:
            from smartcash.dataset.preprocessor.api.preprocessing_api import get_preprocessing_status
            
            status = get_preprocessing_status(config=self.get_current_config())
            if not status.get('service_ready'):
                self.log_warning("Backend service not ready, assuming no files.")
                return 0, 0
                
            stats = status.get('file_statistics', {}).get('train', {})
            preprocessed_files = stats.get('preprocessed_files', 0)
            raw_images = stats.get('raw_images', 0)
            
            return preprocessed_files, raw_images
        except Exception as e:
            self.log_error(f"Cannot check processed file existence: {e}")
            return 0, 0

    def _show_confirmation_dialog(self, title: str, message: str, confirm_action, 
                                confirm_text: str = "Confirm", danger_mode: bool = False) -> Dict[str, Any]:
        """Show confirmation dialog and execute action on confirm."""
        op_container = self.get_component('operation_container')
        if op_container and hasattr(op_container, 'show_dialog'):
            op_container.show_dialog(
                title=title,
                message=message,
                on_confirm=confirm_action,
                confirm_text=confirm_text,
                cancel_text="Batal",
                danger_mode=danger_mode
            )
            return {'success': True, 'message': 'Dialog konfirmasi ditampilkan'}
        else:
            self.log_warning("Dialog not available, executing action directly")
            return confirm_action()

    def _show_info_dialog(self, title: str, message: str) -> Dict[str, Any]:
        """Show information dialog."""
        op_container = self.get_component('operation_container')
        if op_container and hasattr(op_container, 'show_dialog'):
            op_container.show_dialog(
                title=title,
                message=message,
                confirm_text="OK",
                on_cancel=None
            )
        return {'success': True, 'message': 'Tidak ada file untuk dibersihkan'}

    def _format_operation_result(self, result: Dict[str, Any], operation_name: str) -> Dict[str, Any]:
        """Format operation result consistently."""
        if result and result.get('success'):
            return {'success': True, 'message': f'{operation_name} berhasil diselesaikan'}
        else:
            error_msg = result.get('message', f'{operation_name} gagal') if result else f'{operation_name} gagal'
            return {'success': False, 'message': error_msg}

    def _update_operation_summary(self, content: str) -> None:
        """Update the operation summary container with new content using markdown formatter."""
        try:
            summary_container = self.get_component('summary_container')
            if summary_container and hasattr(summary_container, 'set_content'):
                # Convert markdown content to HTML using the new formatter
                from smartcash.ui.core.utils import format_summary_to_html
                formatted_content = format_summary_to_html(
                    content, 
                    title="📊 Preprocessing Summary", 
                    module_name="preprocessing"
                )
                summary_container.set_content(formatted_content)
                self.log_debug("✅ Summary container updated with preprocessing results")
            else:
                # Fallback to operation summary updater
                updater = self.get_component('operation_summary_updater')
                if updater and callable(updater):
                    # Use the new markdown formatter for consistency 
                    from smartcash.ui.core.utils import format_summary_to_html
                    html_content = format_summary_to_html(
                        content, 
                        title="📊 Preprocessing Summary", 
                        module_name="preprocessing"
                    )
                    updater(html_content)
                else:
                    self.log_warning("Summary container tidak ditemukan atau tidak dapat dipanggil.")
        except Exception as e:
            self.log_error(f"Failed to update operation summary: {e}")

    def cleanup(self) -> None:
        """Widget lifecycle cleanup."""
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
        except Exception as e:
            self.log_error(f"Preprocessing module cleanup failed: {e}")

    def __del__(self):
        """Memory management - ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass