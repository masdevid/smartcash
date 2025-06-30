"""
File: smartcash/ui/model/backbone/handlers/config_handler.py
Deskripsi: Handler untuk configuration management yang extends ConfigCellHandler
"""

from typing import Dict, Any, Optional, Type, TypeVar, Generic
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler

T = TypeVar('T', bound='BackboneConfigHandler')

class BackboneConfigHandler(ConfigCellHandler):
    """Handler untuk backbone model configuration management extending ConfigCellHandler"""
    
    def __init__(self, logger_bridge: Any):
        """Initialize config handler
        
        Args:
            logger_bridge: Logger instance untuk logging
        """
        # Store logger bridge first
        self.logger_bridge = logger_bridge
        
        # Initialize parent with module names
        super().__init__(
            module_name='backbone',
            parent_module='model'
        )
        
        # Ensure config file is set up
        self._setup_config_file()
        
        # Load or initialize config
        loaded_config = self.load()
        self.config = loaded_config if loaded_config is not None else self._get_default_config()
        
        # Save default config if file didn't exist
        if loaded_config is None:
            self.save()
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from file
        
        Returns:
            Configuration dictionary or None if not found
        """
        try:
            config = super().load()
            if config and 'model' in config:
                self.logger_bridge.info(f"📁 Loaded config from {self.file_path}")
                return config
            return None
        except Exception as e:
            self.logger_bridge.error(f"❌ Error loading config: {str(e)}")
            return None
    
    def save(self) -> bool:
        """Save current configuration to file
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            success = super().save()
            if success:
                self.logger_bridge.success(f"✅ Configuration saved to {self.file_path}")
            else:
                self.logger_bridge.error("❌ Failed to save configuration")
            return success
        except Exception as e:
            self.logger_bridge.error(f"❌ Error saving config: {str(e)}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, str]:
        """Validate configuration structure
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not config.get('model'):
            return False, "Missing 'model' section in configuration"
        
        model_config = config['model']
        required_fields = ['backbone', 'detection_layers', 'layer_mode']
        
        for field in required_fields:
            if field not in model_config:
                return False, f"Missing required field: {field}"
        
        # Validate values
        valid_backbones = ['efficientnet_b4', 'cspdarknet']
        if model_config.get('backbone') not in valid_backbones:
            return False, f"Invalid backbone: {model_config.get('backbone')}"
        
        valid_modes = ['single', 'multilayer']
        if model_config.get('layer_mode') not in valid_modes:
            return False, f"Invalid layer mode: {model_config.get('layer_mode')}"
        
        return True, "Configuration is valid"
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Override update to add validation
        
        Args:
            new_config: New configuration
        """
        # Validate before update
        is_valid, message = self.validate_config(new_config)
        if not is_valid:
            self.logger_bridge.error(f"❌ Invalid configuration: {message}")
            return
        
        # Call parent update
        super().update(new_config)
        self.logger_bridge.info("📡 Configuration updated")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge configurations
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        import copy
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default backbone configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'model': {
                'backbone': 'efficientnet_b4',
                'model_name': 'smartcash_yolov5',
                'detection_layers': ['banknote'],
                'layer_mode': 'single',
                'num_classes': 7,
                'img_size': 640,
                'feature_optimization': {
                    'enabled': False
                },
                'mixed_precision': True,
                'device': 'auto'
            }
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default backbone configuration (instance method)
        
        Returns:
            Default configuration dictionary
        """
        return self.get_default_config()
    
    def _setup_config_file(self) -> None:
        """Set up the configuration file path using fixed model_config.yaml"""
        # Use parent's implementation with custom filename
        super()._setup_config_file(filename='model_config.yaml')
        
        # Log the file path
        if hasattr(self, 'logger_bridge'):
            self.logger_bridge.info(f"Using config file: {self.file_path}")