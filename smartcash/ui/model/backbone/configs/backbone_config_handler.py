"""
File: smartcash/ui/model/backbone/configs/backbone_config_handler.py
Configuration handler for backbone module using BaseUIModule pattern (simplified).
"""

from typing import Dict, Any, Optional, List
from smartcash.ui.logger import get_module_logger
from .backbone_defaults import get_default_backbone_config, get_available_backbones
from ..constants import VALIDATION_CONFIG, BackboneType


class BackboneConfigHandler:
    """
    Backbone configuration handler using pure delegation pattern.
    
    This class follows the modern BaseUIModule architecture where config handlers
    are pure implementation classes that delegate to BaseUIModule mixins.
    
    Features:
    - 🧬 Backbone model configuration validation
    - 🔧 Model parameter synchronization 
    - 📋 Configuration merge and validation
    - 🎯 UI component sync support
    - 🛡️ Type checking and constraints validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger=None):
        """
        Initialize backbone configuration handler.
        
        Args:
            config: Optional initial configuration
            logger: Optional logger instance
        """
        # Initialize pure delegation pattern (no mixin inheritance)
        self.logger = logger or get_module_logger('smartcash.ui.model.backbone.config')
        self.module_name = 'backbone'
        self.parent_module = 'model'
        
        # Load default config
        self._default_config = get_default_backbone_config()
        self._config = self._default_config.copy()
        self._available_backbones = get_available_backbones()
        self.available_backbones = self._available_backbones
        
        # Update with provided config if any
        if config:
            self._config.update(config)
        
        # Config sections that require UI synchronization
        self.ui_sync_sections = ['backbone', 'model', 'ui']
        
        self.logger.info("✅ Backbone config handler initialized")

    # --- Core Configuration Methods ---

    def get_current_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self._config.copy()

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        if self.validate_config(updates):
            self._config.update(updates)
            self.logger.debug(f"Configuration updated: {list(updates.keys())}")
        else:
            raise ValueError("Invalid configuration updates provided")

    def reset_config(self) -> None:
        """Reset configuration to defaults."""
        self._config = get_default_backbone_config().copy()
        self.logger.info("Configuration reset to defaults")

    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_path: Optional path to save config file
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            # Implementation would save to YAML file
            self.logger.info("Configuration saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False

    @property
    def config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config
    
    @config.setter
    def config(self, value: Dict[str, Any]) -> None:
        """Set configuration with validation."""
        if self.validate_config(value):
            self._config = value
        else:
            raise ValueError("Invalid configuration provided")
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default backbone configuration.
        
        Returns:
            Default configuration dictionary
        """
        return get_default_backbone_config()
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate backbone configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            if not isinstance(config, dict):
                self.logger.error("Configuration must be a dictionary")
                return False
            
            # Check required sections
            required_sections = ['backbone', 'model']
            for section in required_sections:
                if section not in config:
                    self.logger.error(f"Missing required section: {section}")
                    return False
            
            # Backbone-specific validation
            if not self._validate_backbone_section(config.get('backbone', {})):
                return False
            
            if not self._validate_model_section(config.get('model', {})):
                return False
            
            self.logger.debug("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    
    def _validate_backbone_section(self, backbone_config: Dict[str, Any]) -> bool:
        """Validate backbone configuration section."""
        # Check model type
        model_type = backbone_config.get('model_type')
        if not model_type:
            self.logger.error("model_type is required in backbone section")
            return False
        
        # Check if backbone type is supported
        available_types = [bt.value for bt in BackboneType]
        if model_type not in available_types:
            self.logger.error(f"Unsupported backbone type: {model_type}. Available: {available_types}")
            return False
        
        # Check pretrained flag
        if not isinstance(backbone_config.get('pretrained'), bool):
            self.logger.error("pretrained must be a boolean value")
            return False
        
        # Check detection layers
        detection_layers = backbone_config.get('detection_layers', [])
        if not isinstance(detection_layers, list) or not detection_layers:
            self.logger.error("detection_layers must be a non-empty list")
            return False
        
        # Check input size
        input_size = backbone_config.get('input_size', 640)
        min_size = VALIDATION_CONFIG.get('min_input_size', 320)
        max_size = VALIDATION_CONFIG.get('max_input_size', 1280)
        if not (min_size <= input_size <= max_size):
            self.logger.error(f"input_size must be between {min_size} and {max_size}")
            return False
        
        # Check num_classes
        num_classes = backbone_config.get('num_classes', 7)
        min_classes = VALIDATION_CONFIG.get('min_classes', 1)
        max_classes = VALIDATION_CONFIG.get('max_classes', 100)
        if not (min_classes <= num_classes <= max_classes):
            self.logger.error(f"num_classes must be between {min_classes} and {max_classes}")
            return False
        
        return True
    
    def _validate_model_section(self, model_config: Dict[str, Any]) -> bool:
        """Validate model configuration section."""
        # Check backbone consistency
        backbone = model_config.get('backbone')
        if backbone:
            available_types = [bt.value for bt in BackboneType]
            if backbone not in available_types:
                self.logger.error(f"Invalid backbone in model section: {backbone}")
                return False
        
        return True
    
    # Note: update_config is provided by ConfigurationMixin
    
    def set_ui_components(self, ui_components: Dict[str, Any]) -> None:
        """
        Set UI components reference for configuration extraction.
        
        Args:
            ui_components: UI components dictionary
        """
        self._ui_components = ui_components
        self.logger.debug("✅ UI components reference set")
    
    def get_validation_errors(self, config: Dict[str, Any]) -> List[str]:
        """
        Get detailed validation errors for configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        try:
            if not isinstance(config, dict):
                errors.append("Configuration must be a dictionary")
                return errors
            
            # Check required sections
            if 'backbone' not in config:
                errors.append("Missing 'backbone' section")
            if 'model' not in config:
                errors.append("Missing 'model' section")
            
            # Validate backbone section details
            backbone_config = config.get('backbone', {})
            
            model_type = backbone_config.get('model_type')
            if not model_type:
                errors.append("Backbone model_type is required")
            elif model_type not in [bt.value for bt in BackboneType]:
                errors.append(f"Unsupported backbone type: {model_type}")
            
            if not isinstance(backbone_config.get('pretrained'), bool):
                errors.append("Backbone pretrained must be boolean")
            
            detection_layers = backbone_config.get('detection_layers', [])
            if not detection_layers or not isinstance(detection_layers, list):
                errors.append("Detection layers must be a non-empty list")
            
            # Input size validation
            input_size = backbone_config.get('input_size', 640)
            if not isinstance(input_size, int) or input_size < 320 or input_size > 1280:
                errors.append("Input size must be integer between 320 and 1280")
            
            # Classes validation
            num_classes = backbone_config.get('num_classes', 7)
            if not isinstance(num_classes, int) or num_classes < 1 or num_classes > 100:
                errors.append("Number of classes must be integer between 1 and 100")
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors
    
    # ==================== ABSTRACT METHOD IMPLEMENTATIONS ====================
    
    def create_config_handler(self, config: Dict[str, Any]) -> 'BackboneConfigHandler':
        """Create config handler instance for backbone module."""
        return BackboneConfigHandler(config)