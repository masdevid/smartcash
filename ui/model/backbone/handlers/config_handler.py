"""
File: smartcash/ui/model/backbone/handlers/config_handler.py
Deskripsi: Handler untuk backbone model configuration management
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Type, TypeVar, Generic
from pathlib import Path

T = TypeVar('T', bound='BackboneConfigHandler')

class BackboneConfigHandler:
    """Handler untuk backbone model configuration management"""
    
    CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.smartcash', 'config')
    CONFIG_EXT = '.yaml'
    
    def __init__(self, logger_bridge: Any = None):
        """Initialize config handler
        
        Args:
            logger_bridge: Optional logger instance for logging
        """
        self.logger_bridge = logger_bridge
        self.module_name = 'backbone'
        self.parent_module = 'model'
        self.config = {}
        self._setup_config_file()
        self.config = self.load() or self._get_default_config()
        
        # Save default config if file didn't exist
        if not os.path.exists(self.file_path):
            self.save()
    
    @property
    def file_path(self) -> str:
        """Get the path to the config file"""
        return os.path.join(self.CONFIG_DIR, f"{self.module_name}{self.CONFIG_EXT}")
    
    def _setup_config_file(self) -> None:
        """Ensure config directory and file exist"""
        os.makedirs(self.CONFIG_DIR, exist_ok=True)
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                yaml.safe_dump({}, f)
    
    def load(self) -> Optional[Dict[str, Any]]:
        """Load configuration from file
        
        Returns:
            Configuration dictionary or None if not found
        """
        try:
            if os.path.exists(self.file_path):
                with open(self.file_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                    if self.logger_bridge:
                        self.logger_bridge.info(f"📁 Loaded config from {self.file_path}")
                    return config
            return None
        except Exception as e:
            if self.logger_bridge:
                self.logger_bridge.error(f"❌ Error loading config: {str(e)}")
            return None
    
    def save(self) -> bool:
        """Save current configuration to file
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            with open(self.file_path, 'w') as f:
                yaml.safe_dump(self.config, f, default_flow_style=False, sort_keys=False)
            if self.logger_bridge:
                self.logger_bridge.info(f"💾 Config saved to {self.file_path}")
            return True
        except Exception as e:
            if self.logger_bridge:
                self.logger_bridge.error(f"❌ Error saving config: {str(e)}")
            return False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration
        
        Returns:
            Default configuration dictionary
        """
        return {
            'model': {
                'backbone': 'efficientnet-b4',
                'pretrained': True,
                'freeze_backbone': False,
                'num_classes': 2,
                'input_size': 512,
                'dropout': 0.2
            },
            'training': {
                'batch_size': 16,
                'epochs': 50,
                'learning_rate': 1e-4,
                'weight_decay': 1e-4,
                'early_stopping_patience': 10,
                'lr_reduce_patience': 5,
                'min_lr': 1e-6
            },
            'augmentation': {
                'horizontal_flip': True,
                'vertical_flip': True,
                'rotation_range': 20,
                'width_shift_range': 0.1,
                'height_shift_range': 0.1,
                'zoom_range': 0.2,
                'brightness_range': [0.8, 1.2],
                'fill_mode': 'reflect',
                'validation_split': 0.2
            }
        }
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values
        
        Args:
            updates: Dictionary with configuration updates
        """
        def _update_dict(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = _update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = _update_dict(self.config, updates)
        self.save()
    
    def get_config(self, key: str = None, default: Any = None) -> Any:
        """Get configuration value by key
        
        Args:
            key: Dot-separated key path (e.g., 'training.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value or default if not found
        """
        if not key:
            return self.config
            
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values"""
        self.config = self._get_default_config()
        self.save()
    
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