"""
File: smartcash/ui/dataset/preprocessing/handlers/config_event_handlers.py
Deskripsi: Config handlers menggunakan BasePreprocessingHandler untuk DRY implementation
"""

from typing import Dict, Any, Optional
from .base_handler import BasePreprocessingHandler
from smartcash.ui.handlers.config_handlers import ConfigHandler

class ConfigEventHandler(BasePreprocessingHandler):
    """Handler untuk config save/reset operations menggunakan base class"""
    
    def __init__(self, ui_components: Dict[str, Any], config_handler: Optional[ConfigHandler] = None):
        super().__init__(ui_components)
        self.config_handler = config_handler
    
    def setup_handlers(self) -> Dict[str, Any]:
        """Setup save dan reset button handlers"""
        handlers = {}
        
        # Setup save handler
        save_handler = self.setup_button_handler(
            'save_button', 
            self._handle_save_config,
            'save_config'
        )
        if save_handler:
            handlers['save'] = save_handler
        
        # Setup reset handler
        reset_handler = self.setup_button_handler(
            'reset_button',
            self._handle_reset_config, 
            'reset_config'
        )
        if reset_handler:
            handlers['reset'] = reset_handler
        
        self.log_debug("✅ Config handlers setup completed")
        return handlers
    
    def _handle_save_config(self) -> None:
        """Handle save config operation"""
        self.log_info("💾 Menyimpan konfigurasi preprocessing...")
        
        config_handler = self._get_config_handler()
        success = config_handler.save_config(self.ui_components)
        
        if success:
            self.log_success("✅ Konfigurasi berhasil disimpan")
            self.update_status_panel("Konfigurasi disimpan", 'success')
        else:
            raise ValueError("Gagal menyimpan konfigurasi")
    
    def _handle_reset_config(self) -> None:
        """Handle reset config operation"""
        self.log_info("🔄 Mereset konfigurasi ke default...")
        
        config_handler = self._get_config_handler()
        success = config_handler.reset_config(self.ui_components)
        
        if success:
            self.log_success("✅ Konfigurasi berhasil direset")
            self.update_status_panel("Konfigurasi direset", 'success')
        else:
            raise ValueError("Gagal mereset konfigurasi")
    
    def _get_config_handler(self) -> ConfigHandler:
        """Get atau create config handler"""
        if self.config_handler:
            return self.config_handler
        
        from .config_handler import PreprocessingConfigHandler
        handler = PreprocessingConfigHandler()
        handler.set_ui_components(self.ui_components)
        return handler

# Factory function untuk backward compatibility
def setup_config_handlers(ui_components: Dict[str, Any], 
                         config_handler: Optional[ConfigHandler] = None) -> Dict[str, Any]:
    """Factory function untuk setup config handlers"""
    handler = ConfigEventHandler(ui_components, config_handler)
    return handler.setup_handlers()