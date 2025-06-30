"""
Configuration operation handler for dataset downloader.
"""
from typing import Dict, Any, Optional
from smartcash.ui.dataset.downloader.utils.ui_utils import log_to_accordion, clear_outputs
from smartcash.ui.dataset.downloader.utils.button_manager import get_button_manager


class ConfigOperation:
    """Handler for configuration operations."""

    def __init__(self, ui_components: Dict[str, Any]):
        """Initialize with UI components."""
        self.ui_components = ui_components
        self._setup_operation_handlers()

    def setup_config_handlers(self) -> None:
        """Set up save and reset configuration handlers."""
        # Setup save button handler
        if 'save_button' in self.ui_components and self.ui_components['save_button']:
            self.ui_components['save_button'].on_click(self._on_save_clicked)
        
        # Setup reset button handler
        if 'reset_button' in self.ui_components and self.ui_components['reset_button']:
            self.ui_components['reset_button'].on_click(self._on_reset_clicked)

    def _on_save_clicked(self, button=None) -> None:
        """Handle save configuration button click."""
        button_manager = get_button_manager(self.ui_components)
        clear_outputs(self.ui_components)
        button_manager.disable_buttons('save_button')
        
        try:
            config_handler = self.ui_components.get('config_handler')
            if config_handler:
                result = config_handler.save_config(self.ui_components)
                if result.get('status') is True:
                    log_to_accordion(self.ui_components, "✅ Konfigurasi berhasil disimpan", 'success')
                else:
                    error_msg = result.get('error', 'Terjadi kesalahan saat menyimpan konfigurasi')
                    log_to_accordion(self.ui_components, f"❌ {error_msg}", 'error')
            else:
                log_to_accordion(self.ui_components, "❌ Config handler tidak tersedia", 'error')
                
        except Exception as e:
            self._handle_config_error(e, "save_config")
        finally:
            button_manager.enable_buttons()

    def _on_reset_clicked(self, button=None) -> None:
        """Handle reset configuration button click."""
        button_manager = get_button_manager(self.ui_components)
        clear_outputs(self.ui_components)
        button_manager.disable_buttons('reset_button')
        
        try:
            config_handler = self.ui_components.get('config_handler')
            if config_handler:
                result = config_handler.reset_config(self.ui_components)
                if result.get('status') is True:
                    log_to_accordion(self.ui_components, "✅ Konfigurasi berhasil direset ke default", 'success')
                else:
                    error_msg = result.get('error', 'Terjadi kesalahan saat mereset konfigurasi')
                    log_to_accordion(self.ui_components, f"❌ {error_msg}", 'error')
            else:
                log_to_accordion(self.ui_components, "❌ Config handler tidak tersedia", 'error')
                
        except Exception as e:
            self._handle_config_error(e, "reset_config")
        finally:
            button_manager.enable_buttons()

    def _handle_config_error(self, error: Exception, operation: str) -> None:
        """Handle configuration-related errors."""
        from .error_handling import handle_downloader_error, create_downloader_error_context
        
        handle_downloader_error(
            error,
            create_downloader_error_context(
                operation=operation,
                ui_components=self.ui_components
            ),
            logger=self.ui_components.get('logger_bridge'),
            ui_components=self.ui_components
        )

    def _setup_operation_handlers(self) -> None:
        """Setup operation-specific handlers."""
        # Register operation handlers
        if '_operation_handlers' not in self.ui_components:
            self.ui_components['_operation_handlers'] = {}
            
        self.ui_components['_operation_handlers'].update({
            'save_config': self._on_save_clicked,
            'reset_config': self._on_reset_clicked,
        })
