"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Augmentation initializer dengan CommonInitializer pattern terbaru dan fail-fast approach
"""

from typing import Dict, Any, Optional, Type
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.augmentation.handlers.config_handler import AugmentationConfigHandler
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.components.error.error_component import create_error_component
from smartcash.common.logger import get_logger

class AugmentationInitializer(CommonInitializer):
    """Augmentation initializer dengan pattern terbaru dari CommonInitializer"""
    
    def __init__(self, config_handler_class: Type[ConfigHandler] = AugmentationConfigHandler):
        """Initialize augmentation initializer with proper configuration
        
        Args:
            config_handler_class: Optional ConfigHandler class (defaults to AugmentationConfigHandler)
        """
        super().__init__(module_name='augmentation', config_handler_class=config_handler_class)
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Create UI components dengan proper error handling dan validation
        
        Args:
            config: Konfigurasi untuk inisialisasi UI
            **kwargs: Argumen tambahan
            
        Returns:
            Dictionary berisi komponen UI yang valid dengan minimal keys:
            - 'ui': Komponen UI utama
            - 'log_output': Output log widget
            - 'status_panel': Panel status
            
        Raises:
            ValueError: Jika UI components tidak valid atau komponen penting tidak ada
        """
        try:
            from smartcash.ui.dataset.augmentation.components.ui_components import create_augmentation_main_ui
            
            # Create UI components dengan immediate validation
            ui_components = create_augmentation_main_ui(config)
            
            if not isinstance(ui_components, dict):
                raise ValueError(f"UI components harus berupa dictionary, dapat: {type(ui_components)}")
                    
            if not ui_components:
                raise ValueError("UI components tidak boleh kosong")
            
            # Validate critical components exist
            required_components = ['ui', 'log_output', 'status_panel']
            missing = [comp for comp in required_components if comp not in ui_components]
            if missing:
                raise ValueError(f"Komponen UI kritis tidak ditemukan: {missing}")
                
            return ui_components
            
        except Exception as e:
            self.logger.error(f"Gagal membuat komponen UI: {str(e)}", exc_info=True)
            raise
        
        # Add module-specific metadata
        ui_components.update({
            'data_dir': config.get('data', {}).get('dir', 'data'),
            'env': kwargs.get('env'),
            'backend_ready': True,
            'service_integration': True
        })
        
        return ui_components
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Setup event handlers dengan proper logger bridge integration
        
        Args:
            ui_components: Dictionary berisi komponen UI
            config: Konfigurasi yang digunakan
            **kwargs: Argumen tambahan
            
        Returns:
            Dictionary komponen UI yang telah diupdate dengan handlers
            
        Raises:
            ValueError: Jika handler setup gagal
        """
        # Ensure logger bridge is available before setting up handlers
        if not hasattr(self, '_logger_bridge') or not self._logger_bridge:
            raise ValueError("Logger bridge belum diinisialisasi sebelum setup handlers")
                
        # Add logger bridge to ui_components untuk akses handlers
        ui_components['logger_bridge'] = self._logger_bridge
        
        from smartcash.ui.dataset.augmentation.handlers.augmentation_handlers import setup_augmentation_handlers
        
        # Setup handlers dengan error handling
        handlers = setup_augmentation_handlers(ui_components, config, self.config_handler)
        
        if not handlers:
            raise ValueError("Gagal menginisialisasi augmentation handlers")
            
        ui_components['handlers'] = handlers
        
        # Load and update UI with config
        self._load_and_update_ui(ui_components)
        
        return ui_components
    
    def _load_and_update_ui(self, ui_components: Dict[str, Any]) -> None:
        """
        Muat konfigurasi dan perbarui UI dengan error handling yang tepat.
        
        Args:
            ui_components: Dictionary berisi komponen UI
            
        Note:
            Method ini dipanggil setelah setup handlers untuk memastikan
            UI dalam state yang konsisten dengan konfigurasi yang dimuat.
        """
        config_handler = ui_components.get('config_handler')
        if not config_handler:
            self.logger.warning("Config handler tidak tersedia untuk memuat konfigurasi")
            return
            
        try:
            # Pastikan config handler memiliki referensi ke UI components
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            # Muat konfigurasi
            loaded_config = config_handler.load_config()
            
            # Update UI dengan konfigurasi yang dimuat
            if hasattr(config_handler, 'update_ui'):
                config_handler.update_ui(ui_components, loaded_config)
            
            # Simpan konfigurasi yang dimuat
            ui_components['config'] = loaded_config
            
        except Exception as e:
            self.logger.error(f"Gagal memuat konfigurasi: {str(e)}", exc_info=True)
            raise RuntimeError(f"Gagal memuat konfigurasi: {str(e)}") from e
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Dapatkan konfigurasi default untuk modul augmentasi.
        
        Returns:
            Dictionary berisi konfigurasi default
            
        Raises:
            ImportError: Jika modul default config tidak ditemukan
            RuntimeError: Jika gagal memuat konfigurasi default
        """
        try:
            from smartcash.ui.dataset.augmentation.handlers.defaults import get_default_augmentation_config
            return get_default_augmentation_config()
        except ImportError as e:
            self.logger.error("Gagal mengimpor modul konfigurasi default", exc_info=True)
            raise ImportError("Tidak dapat menemukan modul konfigurasi default") from e
        except Exception as e:
            self.logger.error("Gagal memuat konfigurasi default", exc_info=True)
            raise RuntimeError(f"Gagal memuat konfigurasi default: {str(e)}") from e
            

def initialize_augmentation_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
    """Factory function untuk inisialisasi augmentation UI
    
    Args:
        config: Konfigurasi opsional untuk inisialisasi
        **kwargs: Argumen tambahan yang akan diteruskan ke initializer
        
    Returns:
        Widget UI utama yang siap ditampilkan atau dictionary dengan 'ui' key
        
    Example:
        ```python
        ui = initialize_augmentation_ui(config=my_config)
        display(ui)  # or display(ui['ui']) if it's a dict
        ```
    """
    try:
        initializer = AugmentationInitializer()
        result = initializer.initialize(config=config, **kwargs)
        
        # Handle error response
        if isinstance(result, dict) and result.get('error'):
            return result['ui']
        return result
    except Exception as e:
        error_msg = f"❌ Gagal menginisialisasi augmentation UI: {str(e)}"
        return {'ui': create_error_component(error_msg, str(e), "Augmentation Error"), 'error': True}