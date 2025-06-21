"""
File: smartcash/ui/pretrained_model/pretrained_init.py
Deskripsi: Pretrained initializer
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.pretrained_model.handlers.config_handler import PretrainedModelConfigHandler

class PretrainedInit(CommonInitializer):
    """Pretrained initializer"""
    def __init__(self):
        super().__init__('pretrained_model', PretrainedModelConfigHandler)
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan safe widget creation"""
        from smartcash.ui.pretrained_model.components.ui_components import create_pretrained_main_ui
        
        # Clear any existing widgets untuk avoid conflicts
        self._clear_existing_widgets()
        
        try:
            ui_components = create_pretrained_main_ui(config)
            self.logger.info("✅ Berhasil membuat UI components")
        except Exception as e:
            self.logger.error(f"💥 Gagal membuat UI components: {str(e)}")
            try:
                # tambahkan error handling untuk mengatasi kesalahan
                self._handle_create_ui_components_error(e)
            except Exception as e:
                self.logger.error(f"💥 Gagal menangani kesalahan: {str(e)}")
                raise
        
        ui_components.update({
            'pretrained_model_initialized': True,
            'module_name': 'pretrained_model'
        })
        
        return ui_components
    
    def _handle_create_ui_components_error(self, e: Exception):
        """Handle error pada create ui components"""
        # tambahkan logika untuk menangani kesalahan
        self.logger.error(f"💥 Kesalahan tidak terduga: {str(e)}")
        # tambahkan aksi lain jika diperlukan
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan proper error handling"""
        from smartcash.ui.pretrained_model.handlers.pretrained_handlers import setup_pretrained_handlers
        
        try:
            return setup_pretrained_handlers(ui_components, config, env)
        except Exception as e:
            self.logger.error(f"💥 Error setup handlers: {str(e)}")
            return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default config menggunakan ConfigHandler"""
        return self.config_handler_class('pretrained_model').get_default_config() if self.config_handler_class else {}
    
    def _get_critical_components(self) -> List[str]:
        """Critical components untuk validasi"""
        return ['ui', 'download_sync_button', 'log_output', 'progress_tracker']
    
    

# Global instance dan public API
_pretrained_init = PretrainedInit()

def initialize_pretrained_model_ui(env=None, config=None):
    """Public API untuk initialize pretrained model UI"""
    return _pretrained_init.initialize(env=env, config=config)