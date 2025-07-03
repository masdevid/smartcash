"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Augmentation initializer dengan CommonInitializer pattern terbaru dan fail-fast approach

Initialization Flow:
1. Load and validate configuration
2. Create UI components
3. Setup module handlers
4. Return UI with proper error handling
"""

from typing import Dict, Any, Optional, Type, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.augmentation.handlers.config_handler import AugmentationConfigHandler
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.components.error.error_component import create_error_component
from smartcash.ui.handlers.error_handler import create_error_response
from smartcash.common.logger import get_logger

class AugmentationInitializer(CommonInitializer):
    """Augmentation initializer dengan pattern terbaru dari CommonInitializer
    
    Provides a structured approach to initializing the dataset augmentation module with
    proper error handling, logging, and UI component management. Follows the same
    initialization flow as CommonInitializer with additional augmentation-specific
    functionality.
    """
    
    def __init__(self, config_handler_class: Type[ConfigHandler] = AugmentationConfigHandler):
        """Initialize augmentation initializer with proper configuration
        
        Args:
            config_handler_class: Optional ConfigHandler class (defaults to AugmentationConfigHandler)
        """
        super().__init__(module_name='augmentation', config_handler_class=config_handler_class)
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan proper error handling dan validation
        
        Args:
            config: Konfigurasi untuk inisialisasi UI
            env: Optional environment context
            **kwargs: Argumen tambahan
            
        Returns:
            Dictionary berisi komponen UI yang valid
            
        Raises:
            ValueError: Jika UI components tidak valid atau komponen penting tidak ada
        """
        try:
            self.logger.info("🔧 Membuat komponen UI augmentation")
            from smartcash.ui.dataset.augmentation.components.ui_components import create_augmentation_main_ui
            
            # Ensure we have a valid config
            if not config:
                config = self.config_handler.get_default_config()
                
            # Create UI components dengan immediate validation
            ui_components = create_augmentation_main_ui(config)
            
            if not isinstance(ui_components, dict):
                raise ValueError(f"UI components harus berupa dictionary, dapat: {type(ui_components)}")
                    
            if not ui_components:
                raise ValueError("UI components tidak boleh kosong")
            
            # Validate critical components exist
            missing = [name for name in self._get_critical_components() if name not in ui_components]
            if missing:
                raise ValueError(f"Komponen UI kritis tidak ditemukan: {', '.join(missing)}")
            
            # Add module-specific metadata
            ui_components.update({
                'augmentation_initialized': True,
                'module_name': 'augmentation',
                'data_dir': config.get('data', {}).get('dir', 'data'),
                'env': kwargs.get('env'),
                'backend_ready': True,
                'service_integration': True,
                'logger': self.logger  # Ensure logger is available
            })
            
            self.logger.debug(f"UI components created: {list(ui_components.keys())}")
            return ui_components
            
        except Exception as e:
            self.handle_error(f"Failed to create UI components: {str(e)}", exc_info=True)
            return create_error_response("Gagal membuat komponen UI augmentation")
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup event handlers dengan proper logger bridge integration
        
        Args:
            ui_components: Dictionary berisi komponen UI
            config: Konfigurasi yang digunakan
            env: Optional environment context
            **kwargs: Argumen tambahan
            
        Returns:
            Dictionary komponen UI yang telah diupdate dengan handlers
        """
        try:
            self.logger.info("🔄 Menyiapkan handlers augmentation")
            
            # Ensure logger bridge is available
            if not hasattr(self, '_logger_bridge'):
                self._logger_bridge = get_logger('augmentation')
            
            from smartcash.ui.dataset.augmentation.handlers.augmentation_handlers import setup_augmentation_handlers
            
            # Setup handlers dengan error handling
            handlers = setup_augmentation_handlers(ui_components, config, self.config_handler)
            
            # Update UI components with handlers
            if handlers:
                ui_components.update(handlers)
                self.logger.debug(f"Handlers setup complete: {list(handlers.keys()) if handlers else 'No handlers'}")
            
            # Load and update UI with config
            self._load_and_update_ui(ui_components)
            
            return ui_components
            
        except Exception as e:
            self.handle_error(f"Failed to setup module handlers: {str(e)}", exc_info=True)
            return ui_components  # Return original components to avoid breaking the UI
    
    def _load_and_update_ui(self, ui_components: Dict[str, Any]) -> None:
        """Muat konfigurasi dan perbarui UI dengan error handling yang tepat
        
        Args:
            ui_components: Dictionary berisi komponen UI
            
        Note:
            Method ini dipanggil setelah setup handlers untuk memastikan
            UI dalam state yang konsisten dengan konfigurasi yang dimuat
        """
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                self.logger.warning("⚠️ Config handler tidak tersedia untuk memuat konfigurasi")
                return
                
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
            self.logger.debug("Config loaded dan UI updated")
                
        except Exception as e:
            self.handle_error(f"Failed to load and update UI: {str(e)}", exc_info=True)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Dapatkan konfigurasi default untuk modul augmentasi
        
        Returns:
            Dictionary berisi konfigurasi default
        """
        try:
            from smartcash.ui.dataset.augmentation.handlers.defaults import get_default_augmentation_config
            default_config = get_default_augmentation_config()
            self.logger.debug("Default config loaded successfully")
            return default_config
        except ImportError as e:
            self.handle_error("Failed to import default config module", exc_info=True)
            # Return minimal working config to prevent crashes
            return {
                'augmentation': {
                    'enabled': True,
                    'methods': ['flip', 'rotate'],
                    'intensity': 0.5
                },
                'data': {'dir': 'data'}
            }
        except Exception as e:
            self.handle_error(f"Failed to get default config: {str(e)}", exc_info=True)
            # Return minimal working config to prevent crashes
            return {
                'augmentation': {
                    'enabled': True,
                    'methods': ['flip', 'rotate'],
                    'intensity': 0.5
                },
                'data': {'dir': 'data'}
            }
            
    def _get_critical_components(self) -> List[str]:
        """Get list of critical UI components that must exist
        
        Returns:
            List of critical component keys
        """
        return [
            'ui', 'augment_button', 'check_button', 'cleanup_button',
            'log_output', 'status_panel', 'progress_tracker'
        ]
        
    def pre_initialize_checks(self, **kwargs) -> None:
        """Perform pre-initialization checks
        
        Raises:
            Exception: If any pre-initialization check fails
        """
        # Check if we're in a supported environment
        try:
            import IPython
            # Additional checks can be added here
        except ImportError:
            raise RuntimeError("Dataset augmentation requires IPython environment")


# Global instance
_augmentation_initializer = AugmentationInitializer()

def initialize_augmentation_ui(config: Optional[Dict[str, Any]] = None, env=None, **kwargs) -> Dict[str, Any]:
    """Factory function untuk inisialisasi augmentation UI
    
    Args:
        config: Optional configuration dictionary
        env: Optional environment context
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of UI components with 'ui' as the main component
        
    Example:
        ```python
        ui = initialize_augmentation_ui(config=my_config)
        display(ui['ui'])
        ```
    """
    try:
        return _augmentation_initializer.initialize(config=config, env=env, **kwargs)
    except Exception as e:
        error_msg = f"❌ Gagal menginisialisasi augmentation UI: {str(e)}"
        return {'ui': create_error_component(error_msg, str(e), "Augmentation Error"), 'error': True}
