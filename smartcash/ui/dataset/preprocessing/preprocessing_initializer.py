"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Updated preprocessing initializer untuk mengikuti CommonInitializer yang baru dengan fail-fast approach
"""

from typing import Dict, Any
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.preprocessing.handlers.config_handler import PreprocessingConfigHandler


class PreprocessingInitializer(CommonInitializer):
    """Updated preprocessing initializer untuk CommonInitializer yang baru"""
    
    def __init__(self):
        super().__init__(
            module_name='preprocessing',
            config_handler_class=PreprocessingConfigHandler
        )
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Implementasi method untuk membuat UI components dengan fail-fast approach
        """
        from smartcash.ui.dataset.preprocessing.components.ui_components import create_preprocessing_main_ui
        
        # Create UI components langsung dengan config
        ui_components = create_preprocessing_main_ui(config)
        
        # Validasi simple - fail jika tidak sesuai
        if not isinstance(ui_components, dict):
            raise ValueError("UI components harus berupa dictionary")
        
        if not ui_components:
            raise ValueError("UI components tidak boleh kosong")
        
        # Tambahkan metadata minimal
        ui_components.update({
            'module_name': 'preprocessing',
            'config_handler': self.config_handler
        })
        
        self.logger.debug(f"✅ UI components dibuat: {list(ui_components.keys())}")
        return ui_components
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Setup event handlers untuk UI components
        """
        from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handlers import setup_preprocessing_handlers
        
        # Setup handlers dengan fail-fast
        handlers = setup_preprocessing_handlers(ui_components, config, self.config_handler)
        ui_components['handlers'] = handlers
        self.logger.debug("🔗 Event handlers setup selesai")
        return ui_components
    
    def _get_ui_root(self, ui_components: Dict[str, Any]) -> Any:
        """
        Mendapatkan root UI component untuk display
        """
        # Prioritas: main_ui > accordion > widget pertama
        root_candidates = ['main_ui', 'accordion', 'container']
        
        for candidate in root_candidates:
            if candidate in ui_components:
                return ui_components[candidate]
        
        # Fallback: ambil widget pertama yang ada
        for key, widget in ui_components.items():
            if hasattr(widget, 'children') or hasattr(widget, 'value'):
                return widget
        
        raise ValueError("Tidak ada root UI component yang valid")
        
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Return default configuration for the preprocessing module.
        
        Returns:
            Dict containing default configuration values
        """
        from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
        return get_default_preprocessing_config()
            
        # Fallback default config if config_handler is not available
        return {
            'preprocessing': {
                'enabled': True,
                'steps': []
            },
            'performance': {
                'batch_size': 32
            }
        }


# Entry point function dengan signature sederhana
def initialize_preprocessing_ui(config: Dict[str, Any] = None, **kwargs):
    """
    Entry point untuk inisialisasi preprocessing UI
    
    Args:
        config: Konfigurasi awal (optional)
        **kwargs: Parameter tambahan
        
    Returns:
        Root UI component
    """
    initializer = PreprocessingInitializer()
    return initializer.initialize(config=config, **kwargs)