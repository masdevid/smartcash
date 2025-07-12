"""
File: smartcash/ui/model/backbone/configs/backbone_config_handler.py
Configuration handler for backbone module following UIModule pattern.
"""

from typing import Dict, Any, Optional, List
from smartcash.ui.core.handlers.config_handler import ConfigHandler
from smartcash.ui.logger import get_module_logger
from .backbone_defaults import get_default_backbone_config, get_available_backbones
from ..constants import VALIDATION_CONFIG, BackboneType


class BackboneConfigHandler(ConfigHandler):
    """
    Configuration handler for backbone module.
    
    Features:
    - 🧬 Backbone model configuration validation
    - 🔧 Model parameter synchronization
    - 📋 Configuration merge and validation
    - 🎯 UI component sync support
    - 🛡️ Type checking and constraints validation
    """
    
    def __init__(self):
        """Initialize backbone configuration handler."""
        super().__init__(
            module_name='backbone',
            parent_module='model'
        )
        self.logger = get_module_logger("smartcash.ui.model.backbone.configs")
        self.available_backbones = get_available_backbones()
        
        # Config sections that require UI synchronization
        self.ui_sync_sections = ['backbone', 'model', 'ui']
        
        self.logger.debug("✅ BackboneConfigHandler initialized")
    
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
            
            # Validate backbone section
            backbone_config = config.get('backbone', {})
            if not self._validate_backbone_section(backbone_config):
                return False
            
            # Validate model section
            model_config = config.get('model', {})
            if not self._validate_model_section(model_config):
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
        pretrained = backbone_config.get('pretrained')
        if not isinstance(pretrained, bool):
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
    
    def sync_to_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """
        Synchronize configuration to UI components.
        
        Args:
            ui_components: UI components dictionary
            config: Configuration to sync
            
        Returns:
            True if sync successful, False otherwise
        """
        try:
            if not ui_components or not config:
                self.logger.warning("Missing UI components or config for sync")
                return False
            
            # Update form container if available
            form_container = ui_components.get('form_container')
            if form_container and hasattr(form_container, 'update_from_config'):
                form_container.update_from_config(config)
                self.logger.debug("✅ Form container updated from config")
            
            # Update summary container with backbone info
            summary_container = ui_components.get('summary_container')
            if summary_container and hasattr(summary_container, 'update_content'):
                summary_content = self._generate_summary_content(config)
                summary_container.update_content(summary_content)
                self.logger.debug("✅ Summary container updated")
            
            return True
            
        except Exception as e:
            self.logger.error(f"UI sync error: {e}")
            return False
    
    def sync_from_ui(self, ui_components: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Synchronize configuration from UI components.
        
        Args:
            ui_components: UI components dictionary
            
        Returns:
            Configuration dictionary if sync successful, None otherwise
        """
        try:
            if not ui_components:
                self.logger.warning("No UI components available for sync")
                return None
            
            # Get current config from form container
            form_container = ui_components.get('form_container')
            if form_container and hasattr(form_container, 'get_form_values'):
                form_values = form_container.get_form_values()
                
                # Convert form values to config structure
                config = self._form_values_to_config(form_values)
                
                self.logger.debug("✅ Configuration synced from UI")
                return config
            
            self.logger.warning("Form container not available for sync")
            return None
            
        except Exception as e:
            self.logger.error(f"UI to config sync error: {e}")
            return None
    
    def _form_values_to_config(self, form_values: Dict[str, Any]) -> Dict[str, Any]:
        """Convert form values to configuration structure."""
        backbone_type = form_values.get('backbone_type', 'efficientnet_b4')
        pretrained = form_values.get('pretrained', True)
        feature_optimization = form_values.get('feature_optimization', True)
        mixed_precision = form_values.get('mixed_precision', True)
        input_size = form_values.get('input_size', 640)
        num_classes = form_values.get('num_classes', 7)
        
        return {
            'backbone': {
                'model_type': backbone_type,
                'pretrained': pretrained,
                'feature_optimization': feature_optimization,
                'mixed_precision': mixed_precision,
                'detection_layers': ['banknote'],
                'layer_mode': 'single',
                'input_size': input_size,
                'num_classes': num_classes,
                'early_training': {
                    'enabled': True,
                    'validation_from_pretrained': True,
                    'auto_build': False
                }
            },
            'model': {
                'backbone': backbone_type,
                'pretrained': pretrained,
                'detection_layers': ['banknote'],
                'layer_mode': 'single',
                'feature_optimization': feature_optimization,
                'mixed_precision': mixed_precision,
                'input_size': input_size,
                'num_classes': num_classes
            },
            'ui': {
                'show_advanced_options': form_values.get('show_advanced_options', False),
                'auto_validate': form_values.get('auto_validate', True),
                'show_model_info': form_values.get('show_model_info', True),
                'summary_panel_enabled': True
            }
        }
    
    def _generate_summary_content(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary content for display."""
        backbone_config = config.get('backbone', {})
        model_type = backbone_config.get('model_type', 'efficientnet_b4')
        
        # Get backbone info
        backbone_info = self.available_backbones.get(model_type, {})
        
        return {
            'title': 'Backbone Configuration Summary',
            'sections': {
                'Model Information': {
                    'Backbone': backbone_info.get('display_name', model_type),
                    'Description': backbone_info.get('description', 'N/A'),
                    'Pretrained': 'Yes' if backbone_config.get('pretrained') else 'No',
                    'Recommended': 'Yes' if backbone_info.get('recommended') else 'No'
                },
                'Configuration': {
                    'Input Size': f"{backbone_config.get('input_size', 640)}px",
                    'Classes': backbone_config.get('num_classes', 7),
                    'Feature Optimization': 'Enabled' if backbone_config.get('feature_optimization') else 'Disabled',
                    'Mixed Precision': 'Enabled' if backbone_config.get('mixed_precision') else 'Disabled'
                },
                'Performance': {
                    'Memory Usage': backbone_info.get('memory_usage', 'N/A'),
                    'Inference Speed': backbone_info.get('inference_speed', 'N/A'),
                    'Accuracy': backbone_info.get('accuracy', 'N/A'),
                    'Output Channels': str(backbone_info.get('output_channels', []))
                }
            }
        }
    
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