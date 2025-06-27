"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Preprocessing initializer yang mengimplementasikan CommonInitializer
"""

from typing import Dict, Any, Optional, Type
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.preprocessing.handlers.config_handler import PreprocessingConfigHandler
from smartcash.ui.handlers.config_handlers import ConfigHandler

class PreprocessingInitializer(CommonInitializer):
    """Initializer untuk modul preprocessing yang mengimplementasikan CommonInitializer"""
    
    def __init__(self, module_name: str = 'preprocessing', 
                 config_handler_class: Type[ConfigHandler] = PreprocessingConfigHandler,
                 **kwargs):
        """Initialize the preprocessing initializer.
        
        Args:
            module_name: Nama modul (default: 'preprocessing')
            config_handler_class: Kelas handler konfigurasi
            **kwargs: Argumen tambahan untuk parent class
        """
        super().__init__(
            module_name=module_name,
            config_handler_class=config_handler_class,
            **kwargs
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
        
        try:
            # Create UI components
            ui_components = create_preprocessing_main_ui(config)
            
            if not isinstance(ui_components, dict):
                raise ValueError("UI components harus berupa dictionary")
                
            if not ui_components:
                raise ValueError("UI components tidak boleh kosong")
            
            # Initialize logger bridge
            self._initialize_logger_bridge(ui_components)
            
            # Add required components
            ui_components.update({
                'module_name': self.module_name,
                'config_handler': self.config_handler,
                'logger_bridge': self._logger_bridge
            })
            
            # Log successful initialization
            if self._logger_bridge:
                self._logger_bridge.info(f"✅ UI components berhasil diinisialisasi untuk {self.module_name}")
            
            return ui_components
            
        except Exception as e:
            error_msg = f"❌ Gagal membuat UI components: {str(e)}"
            if hasattr(self, '_logger_bridge') and self._logger_bridge:
                self._logger_bridge.error(error_msg, exc_info=True)
            else:
                print(error_msg)
                import traceback
                traceback.print_exc()
            raise
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Setup event handlers untuk komponen UI.
        
        Args:
            ui_components: Dictionary berisi komponen UI
            config: Konfigurasi yang digunakan
            **kwargs: Argumen tambahan
            
        Returns:
            Dictionary komponen UI yang telah diupdate dengan handlers
            
        Raises:
            ValueError: Jika logger bridge belum diinisialisasi dengan benar
        """
        try:
            # Pastikan logger bridge sudah diinisialisasi
            if not hasattr(self, '_logger_bridge') or not self._logger_bridge:
                self._initialize_logger_bridge(ui_components)
                
            # Pastikan logger bridge tersedia di ui_components
            if 'logger_bridge' not in ui_components or not ui_components['logger_bridge']:
                ui_components['logger_bridge'] = self._logger_bridge
                
            from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handlers import setup_preprocessing_handlers
            
            # Setup handlers dengan logger bridge yang sudah diinisialisasi
            handlers = setup_preprocessing_handlers(ui_components, config, self.config_handler)
            
            if not handlers:
                raise ValueError("Gagal menginisialisasi preprocessing handlers")
                
            ui_components['handlers'] = handlers
            
            # Log successful handler setup
            if self._logger_bridge:
                self._logger_bridge.info("✅ Preprocessing handlers berhasil diinisialisasi")
                
            return ui_components
            
        except Exception as e:
            error_msg = f"❌ Gagal setup preprocessing handlers: {str(e)}"
            if hasattr(self, '_logger_bridge') and self._logger_bridge:
                self._logger_bridge.error(error_msg, exc_info=True)
            else:
                print(error_msg)
                import traceback
                traceback.print_exc()
            raise ValueError(error_msg) from e
    
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