"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py
Deskripsi: UNIFIED handler yang menggunakan UI handlers dan backend services
"""

from typing import Dict, Any
from smartcash.ui.dataset.downloader.handlers.base_ui_handler import (
    DownloadUIHandler, CheckUIHandler, CleanupUIHandler
)

class DownloadHandler:
    """Unified handler yang menggabungkan semua UI handlers"""
    
    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any], logger=None):
        self.ui_components = ui_components
        self.config = config
        self.logger = logger
        
        # Create specialized UI handlers
        self.download_handler = DownloadUIHandler(ui_components, logger)
        self.check_handler = CheckUIHandler(ui_components, logger)
        self.cleanup_handler = CleanupUIHandler(ui_components, logger)
        
        # Button mapping
        self.button_handlers = {
            'download_button': self.download_handler.handle_download_click,
            'check_button': self.check_handler.handle_check_click,
            'cleanup_button': self.cleanup_handler.handle_cleanup_click,
            'save_button': self._handle_save_click,
            'reset_button': self._handle_reset_click
        }
    
    def setup_handlers(self) -> Dict[str, Any]:
        """Setup all button handlers"""
        for button_name, handler in self.button_handlers.items():
            button = self.ui_components.get(button_name)
            if button and hasattr(button, 'on_click'):
                # Clear existing handlers
                self._clear_button_handlers(button)
                # Bind new handler
                button.on_click(handler)
        
        self.logger.success("✅ Unified handlers berhasil disetup")
        
        # Add handler references to ui_components
        self.ui_components.update({
            'download_handler': self.download_handler,
            'check_handler': self.check_handler,
            'cleanup_handler': self.cleanup_handler,
            'unified_handler': self
        })
        
        return self.ui_components
    
    def _clear_button_handlers(self, button) -> None:
        """Clear existing button handlers"""
        try:
            if hasattr(button, '_click_handlers') and hasattr(button._click_handlers, 'callbacks'):
                button._click_handlers.callbacks.clear()
        except Exception:
            pass
    
    def _handle_save_click(self, button) -> None:
        """Handle save button click"""
        try:
            self.download_handler._prepare_button_state(button, "💾 Saving...")
            
            config_handler = self.ui_components.get('config_handler')
            if not config_handler:
                self.download_handler._handle_ui_error("Config handler tidak tersedia", button)
                return
            
            success = config_handler.save_config(self.ui_components)
            
            if success:
                self.download_handler._show_ui_success("Konfigurasi berhasil disimpan", button)
            else:
                self.download_handler._handle_ui_error("Gagal menyimpan konfigurasi", button)
                
        except Exception as e:
            self.download_handler._handle_ui_error(f"Error saat menyimpan: {str(e)}", button)
    
    def _handle_reset_click(self, button) -> None:
        """Handle reset button click"""
        try:
            self.download_handler._prepare_button_state(button, "🔄 Resetting...")
            
            config_handler = self.ui_components.get('config_handler')
            if not config_handler:
                self.download_handler._handle_ui_error("Config handler tidak tersedia", button)
                return
            
            success = config_handler.reset_config(self.ui_components)
            
            if success:
                self.download_handler._show_ui_success("Konfigurasi berhasil direset", button)
            else:
                self.download_handler._handle_ui_error("Gagal mereset konfigurasi", button)
                
        except Exception as e:
            self.download_handler._handle_ui_error(f"Error saat reset: {str(e)}", button)

def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup unified download handlers"""
    logger = ui_components.get('logger')
    
    try:
        # Create unified handler
        unified_handler = DownloadHandler(ui_components, config, logger)
        
        # Setup handlers
        ui_components = unified_handler.setup_handlers()
        
        logger.success("✅ Unified download handlers berhasil dikonfigurasi")
        return ui_components
        
    except Exception as e:
        logger.error(f"❌ Error setup unified handlers: {str(e)}")
        return ui_components

# Export
__all__ = ['setup_download_handlers', 'DownloadHandler']