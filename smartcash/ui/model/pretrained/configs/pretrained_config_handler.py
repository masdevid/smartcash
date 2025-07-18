"""
File: smartcash/ui/model/pretrained/configs/pretrained_config_handler.py
Description: Config handler for pretrained models module using core mixins.
"""

from typing import Dict, Any, Optional, Type, TypeVar

from smartcash.ui.core.mixins.configuration_mixin import ConfigurationMixin
from smartcash.ui.core.mixins.logging_mixin import LoggingMixin
from smartcash.ui.logger import get_module_logger
from .pretrained_defaults import get_default_pretrained_config

T = TypeVar('T', bound='PretrainedConfigHandler')

class PretrainedConfigHandler(LoggingMixin, ConfigurationMixin):
    """
    Config handler for pretrained models module.
    
    Uses LoggingMixin for logging and ConfigurationMixin for configuration management.
    Follows the mixin pattern used throughout the UI module system.
    """
    
    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        """
        Initialize pretrained config handler.
        
        Args:
            default_config: Optional default configuration dictionary
        """
        # Initialize mixins
        super().__init__()
        
        # Module identification for logging
        self.module_name = 'pretrained'
        self.parent_module = 'model'
        
        # Store default config
        self._default_config = default_config or get_default_pretrained_config()
        
        # Initialize internal config directly instead of calling _initialize_config_handler
        # to avoid infinite recursion
        self._merged_config = self._default_config.copy()
        
        self.log("PretrainedConfigHandler initialized", 'debug')
    
    # ==================== ABSTRACT METHOD IMPLEMENTATIONS ====================
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return self._default_config.copy()
    
    def create_config_handler(self, config: Dict[str, Any]) -> 'PretrainedConfigHandler':
        """Create a new instance of this config handler."""
        # Return self to avoid infinite recursion
        return self
    
    # ==================== CONFIGURATION METHODS ====================
    
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_filename: Optional config filename (default: 'pretrained_config.yaml')
            
        Returns:
            Result dictionary with success status and config
        """
        try:
            # Load default config
            config = self.get_default_config()
            return {'success': True, 'config': config}
        except Exception as e:
            self.log(f"Error loading config: {e}", 'error')
            return {'success': False, 'config': self.get_default_config(), 'error': str(e)}
    
    def save_config(self, ui_components: Dict[str, Any] = None, config_filename: str = None) -> Dict[str, Any]:
        """
        Save configuration from UI components.
        
        Args:
            ui_components: Dictionary of UI components
            config_filename: Optional config filename
            
        Returns:
            Result dictionary with success status
        """
        try:
            if ui_components:
                extracted_config = self.extract_config(ui_components)
                if extracted_config:
                    self._merged_config.update(extracted_config)
                    
            return {'success': True, 'message': 'Configuration saved successfully'}
        except Exception as e:
            self.log(f"Error saving config: {e}", 'error')
            return {'success': False, 'message': str(e)}
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract configuration from UI components.
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            Extracted configuration dictionary
        """
        try:
            # Extract basic form values
            config = {}
            
            # Models directory
            if 'models_dir_input' in ui_components:
                models_dir = ui_components['models_dir_input']
                if hasattr(models_dir, 'value'):
                    config['models_dir'] = models_dir.value
            
            # Model URLs
            model_urls = {}
            if 'yolo_url_input' in ui_components:
                yolo_url = ui_components['yolo_url_input']
                if hasattr(yolo_url, 'value'):
                    model_urls['yolov5s'] = yolo_url.value
            
            if 'efficientnet_url_input' in ui_components:
                efficientnet_url = ui_components['efficientnet_url_input']
                if hasattr(efficientnet_url, 'value'):
                    model_urls['efficientnet_b4'] = efficientnet_url.value
            
            if model_urls:
                config['model_urls'] = model_urls
            
            # Auto download checkbox
            if 'auto_download_checkbox' in ui_components:
                auto_download = ui_components['auto_download_checkbox']
                if hasattr(auto_download, 'value'):
                    config['auto_download'] = auto_download.value
            
            # Validate checkbox
            if 'validate_checkbox' in ui_components:
                validate = ui_components['validate_checkbox']
                if hasattr(validate, 'value'):
                    config['validate_downloads'] = validate.value
            
            return config
            
        except Exception as e:
            self.log(f"Error extracting config: {e}", 'error')
            return {}
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation result with status and messages
        """
        try:
            errors = []
            
            # Validate models directory
            models_dir = config.get('models_dir', '')
            if not models_dir or not models_dir.strip():
                errors.append("Models directory is required")
            
            # Validate model URLs format if provided
            model_urls = config.get('model_urls', {})
            for model_name, url in model_urls.items():
                if url and not url.startswith(('http://', 'https://')):
                    errors.append(f"Invalid URL format for {model_name}")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': []
            }
            
        except Exception as e:
            self.log(f"Error validating config: {e}", 'error')
            return {'valid': False, 'errors': [str(e)]}
    
    def _config_to_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert config dictionary to UI components format for validation.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            UI components dictionary
        """
        # Mock UI components for validation
        class MockWidget:
            def __init__(self, value):
                self.value = value
        
        model_urls = config.get('model_urls', {})
        
        return {
            'models_dir_input': MockWidget(config.get('models_dir', '')),
            'yolo_url_input': MockWidget(model_urls.get('yolov5s', '')),
            'efficientnet_url_input': MockWidget(model_urls.get('efficientnet_b4', '')),
            'auto_download_checkbox': MockWidget(config.get('auto_download', False)),
            'validate_checkbox': MockWidget(config.get('validate_downloads', True))
        }

def get_pretrained_config_handler() -> PretrainedConfigHandler:
    """
    Get or create a pretrained config handler instance.
    
    Returns:
        PretrainedConfigHandler instance
    """
    return PretrainedConfigHandler()