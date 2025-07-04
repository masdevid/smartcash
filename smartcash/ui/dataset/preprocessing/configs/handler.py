"""
File: smartcash/ui/dataset/preprocessing/configs/handler.py
Deskripsi: Config handler untuk preprocessing module dengan SharedConfigHandler integration.
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.handlers.config_handler import SharedConfigHandler
from smartcash.ui.handlers.error_handler import handle_ui_errors


class PreprocessingConfigHandler(SharedConfigHandler):
    """Config handler untuk preprocessing module dengan SharedConfigHandler integration.
    
    Features:
    - 🎯 Complete config management dengan core architecture
    - 💾 Persistence support (optional)
    - 🔗 Shared config support
    - 🔄 UI-Config synchronization
    """
    
    CONFIG_VERSION = "1.0.0"
    
    @handle_ui_errors(error_component_title="Config Handler Initialization Error", log_error=True)
    def __init__(self, 
                 module_name: str = "preprocessing", 
                 parent_module: str = "dataset",
                 default_config: Optional[Dict[str, Any]] = None,
                 persistence_enabled: bool = True,
                 enable_sharing: bool = True):
        """Initialize preprocessing config handler.
        
        Args:
            module_name: Nama module
            parent_module: Parent module
            default_config: Default configuration
            persistence_enabled: Whether to enable config persistence
            enable_sharing: Enable config sharing
        """
        # Initialize parent class
        super().__init__(module_name, parent_module, default_config, 
                        persistence_enabled=persistence_enabled)
        
        # Enable sharing if requested
        if enable_sharing:
            self.enable_sharing(True)
            
        # UI components reference
        self.ui_components = {}
        
        self.logger.debug(f"📋 PreprocessingConfigHandler initialized")
    
    @handle_ui_errors(error_component_title="Config Extraction Error", log_error=True)
    def extract_config_from_ui(self) -> Dict[str, Any]:
        """Extract config dari UI components.
        
        Returns:
            Extracted configuration dictionary
        """
        try:
            from smartcash.ui.dataset.preprocessing.configs.extractor import extract_preprocessing_config
            return extract_preprocessing_config(self.ui_components)
        except Exception as e:
            self.logger.error(f"❌ Error extracting config: {str(e)}")
            return self.get_default_config()
    
    @handle_ui_errors(error_component_title="UI Update Error", log_error=True)
    def update_ui_from_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Update UI components dari config.
        
        Args:
            config: Configuration dictionary to apply
        """
        try:
            from smartcash.ui.dataset.preprocessing.configs.updater import update_preprocessing_ui
            update_preprocessing_ui(self.ui_components, config or self.config)
            self.logger.info("🔄 UI updated with config")
        except Exception as e:
            self.logger.error(f"❌ Error updating UI: {str(e)}")
    
    @handle_ui_errors(error_component_title="Default Config Error", log_error=True)
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config untuk preprocessing.
        
        Returns:
            Default configuration dictionary
        """
        try:
            from smartcash.ui.dataset.preprocessing.configs.defaults import get_default_preprocessing_config
            return get_default_preprocessing_config()
        except Exception as e:
            self.logger.error(f"❌ Error loading defaults: {str(e)}")
            return {
                'preprocessing': {
                    'enabled': True,
                    'resolution': '640x640',
                    'normalization': 'minmax',
                    'preserve_aspect': True,
                    'target_splits': ['train', 'valid'],
                    'batch_size': 32,
                    'validation': True,
                    'move_invalid': True,
                    'invalid_dir': 'invalid'
                },
                'cleanup': {
                    'target': 'preprocessed',
                    'backup': True
                },
                'data': {
                    'dir': 'data'
                }
            }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            from smartcash.ui.dataset.preprocessing.configs.validator import validate_preprocessing_config
            return validate_preprocessing_config(config)
        except Exception as e:
            self.logger.error(f"❌ Error validating config: {str(e)}")
            return True  # Default to accepting config if validation fails
    
    def set_ui_components(self, ui_components: Dict[str, Any]) -> None:
        """Set UI components reference.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        self.ui_components = ui_components


# Factory function
def create_preprocessing_config_handler(ui_components: Optional[Dict[str, Any]] = None,
                                     persistence_enabled: bool = True,
                                     enable_sharing: bool = True) -> PreprocessingConfigHandler:
    """Create preprocessing config handler instance.
    
    Args:
        ui_components: Dictionary containing UI components
        persistence_enabled: Whether to enable config persistence
        enable_sharing: Enable config sharing
        
    Returns:
        PreprocessingConfigHandler instance
    """
    handler = PreprocessingConfigHandler(persistence_enabled=persistence_enabled,
                                       enable_sharing=enable_sharing)
    
    if ui_components:
        handler.set_ui_components(ui_components)
        
    return handler
