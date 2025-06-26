"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Preprocessing initializer yang mengimplementasikan CommonInitializer
"""

from typing import Dict, Any, Optional
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.preprocessing.handlers.config_handler import PreprocessingConfigHandler

class PreprocessingInitializer(CommonInitializer):
    """Initializer untuk modul preprocessing yang mengimplementasikan CommonInitializer"""
    
    def __init__(self):
        super().__init__(
            module_name='preprocessing',
            config_handler_class=PreprocessingConfigHandler
        )
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Buat komponen UI untuk preprocessing.
        
        Args:
            config: Konfigurasi untuk inisialisasi UI
            **kwargs: Argumen tambahan
            
        Returns:
            Dictionary berisi komponen UI
        """
        from smartcash.ui.dataset.preprocessing.components.ui_components import create_preprocessing_main_ui
        
        ui_components = create_preprocessing_main_ui(config)
        
        if not isinstance(ui_components, dict):
            raise ValueError("UI components harus berupa dictionary")
            
        if not ui_components:
            raise ValueError("UI components tidak boleh kosong")
            
        ui_components.update({
            'module_name': self.module_name,
            'config_handler': self.config_handler
        })
        
        return ui_components
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Setup event handlers untuk komponen UI.
        
        Args:
            ui_components: Dictionary berisi komponen UI
            config: Konfigurasi yang digunakan
            **kwargs: Argumen tambahan
            
        Returns:
            Dictionary komponen UI yang telah diupdate dengan handlers
        """
        from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handlers import setup_preprocessing_handlers
        
        handlers = setup_preprocessing_handlers(ui_components, config, self.config_handler)
        ui_components['handlers'] = handlers
        return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Dapatkan konfigurasi default untuk modul preprocessing.
        
        Returns:
            Dictionary berisi konfigurasi default
            
        Raises:
            ImportError: Jika modul default config tidak ditemukan
        """
        from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
        return get_default_preprocessing_config()

def initialize_preprocessing_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
    """
    Inisialisasi UI untuk modul preprocessing.
    
    Args:
        config: Konfigurasi opsional untuk inisialisasi
        **kwargs: Argumen tambahan yang akan diteruskan ke initializer
        
    Returns:
        Komponen UI utama
    """
    return PreprocessingInitializer().initialize(config=config, **kwargs)