"""
Configuration management mixin for UI modules.

Provides standard configuration merging, validation, save/reset functionality.
"""

from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
# Removed problematic imports for now


class ConfigurationMixin(ABC):
    """
    Mixin providing common configuration management functionality.
    
    This mixin provides:
    - Configuration merging with defaults
    - Save/reset/load operations
    - UI synchronization
    - Standard error handling
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config_handler: Optional[Any] = None
        self._merged_config: Dict[str, Any] = {}
        
    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this module."""
        pass
    
    @abstractmethod
    def create_config_handler(self, config: Dict[str, Any]) -> Any:
        """Create config handler instance for this module."""
        pass
    
    def _merge_with_defaults(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge user configuration with default values.
        
        Args:
            user_config: User-provided configuration
            
        Returns:
            Merged configuration dictionary
        """
        try:
            default_config = self.get_default_config()
            
            # Deep merge configurations
            merged = default_config.copy()
            
            def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
                for key, value in override.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        base[key] = deep_merge(base[key], value)
                    else:
                        base[key] = value
                return base
            
            return deep_merge(merged, user_config)
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error merging configurations: {e}")
            return user_config
    
    def _initialize_config_handler(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize configuration handler with merged config."""
        try:
            # Get default config first
            default_config = self.get_default_config()
            
            # Merge provided config with defaults
            if config:
                merged_config = self._merge_with_defaults(config)
            else:
                merged_config = default_config
            
            # Create config handler
            self._config_handler = self.create_config_handler(merged_config)
            
            # Store merged config internally
            self._merged_config = merged_config
            
            if hasattr(self, 'logger'):
                self.logger.debug("✅ Config handler initialized")
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to initialize config handler: {e}")
            raise
    
    def save_config(self) -> Dict[str, Any]:
        """
        Save current configuration from UI.
        
        Returns:
            Operation result dictionary
        """
        try:
            if not self._config_handler:
                raise RuntimeError("Config handler not available")
            
            # Extract configuration from UI
            config = self._config_handler.extract_config_from_ui()
            
            # Validate configuration
            if hasattr(self._config_handler, 'validate_config'):
                validation_result = self._config_handler.validate_config(config)
                if not validation_result.get('valid', True):
                    raise ValueError(validation_result.get('message', 'Configuration validation failed'))
            
            # Save configuration (pass config as name parameter, not as config data)
            if hasattr(self._config_handler, 'save_config'):
                # First update the handler's config with the extracted config
                if hasattr(self._config_handler, 'update_config'):
                    self._config_handler.update_config(config)
                
                # Then save (without passing config as name parameter)
                save_result = self._config_handler.save_config()
                if not save_result.get('success', False):
                    raise RuntimeError(save_result.get('message', 'Configuration save failed'))
            
            # Update internal config
            self._merged_config.update(config)
            
            if hasattr(self, 'logger'):
                self.logger.info("✅ Configuration saved successfully")
                
            return {'success': True, 'message': 'Configuration saved successfully'}
            
        except Exception as e:
            error_msg = f"Failed to save configuration: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def reset_config(self) -> Dict[str, Any]:
        """
        Reset configuration to defaults.
        
        Returns:
            Operation result dictionary
        """
        try:
            if not self._config_handler:
                raise RuntimeError("Config handler not available")
            
            # Get default configuration
            default_config = self.get_default_config()
            
            # Update internal config
            self._merged_config = default_config.copy()
            
            # Update config handler
            if hasattr(self._config_handler, 'update_config'):
                self._config_handler.update_config(default_config)
            
            # Sync to UI if components are available
            if hasattr(self, '_ui_components') and self._ui_components:
                if hasattr(self._config_handler, 'sync_to_ui'):
                    self._config_handler.sync_to_ui(self._ui_components, default_config)
            
            if hasattr(self, 'logger'):
                self.logger.info("✅ Configuration reset to defaults")
                
            return {'success': True, 'message': 'Configuration reset to defaults'}
            
        except Exception as e:
            error_msg = f"Failed to reset configuration: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Operation result dictionary
        """
        try:
            if not self._config_handler:
                raise RuntimeError("Config handler not available")
            
            # Load configuration
            if hasattr(self._config_handler, 'load_config'):
                load_result = self._config_handler.load_config(config_path)
                if not load_result.get('success', False):
                    raise RuntimeError(load_result.get('message', 'Configuration load failed'))
                
                config = load_result.get('config', {})
            else:
                # Fallback to basic file loading
                import json
                import yaml
                
                with open(config_path, 'r') as f:
                    if config_path.endswith('.json'):
                        config = json.load(f)
                    elif config_path.endswith(('.yaml', '.yml')):
                        config = yaml.safe_load(f)
                    else:
                        raise ValueError(f"Unsupported configuration file format: {config_path}")
            
            # Merge with defaults
            merged_config = self._merge_with_defaults(config)
            self._merged_config = merged_config
            
            # Sync to UI if components are available
            if hasattr(self, '_ui_components') and self._ui_components:
                if hasattr(self._config_handler, 'sync_to_ui'):
                    self._config_handler.sync_to_ui(self._ui_components, merged_config)
            
            if hasattr(self, 'logger'):
                self.logger.info(f"✅ Configuration loaded from {config_path}")
                
            return {'success': True, 'message': f'Configuration loaded from {config_path}'}
            
        except Exception as e:
            error_msg = f"Failed to load configuration from {config_path}: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self._merged_config.copy()
    
    def update_config_value(self, key: str, value: Any) -> None:
        """
        Update a single configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: New value
        """
        try:
            # Support dot notation for nested keys
            keys = key.split('.')
            config = self._merged_config
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
            
            # Update config handler if available
            if self._config_handler and hasattr(self._config_handler, 'update_config'):
                self._config_handler.update_config({key: value})
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to update config value {key}: {e}")
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        try:
            # Support dot notation for nested keys
            keys = key.split('.')
            config = self._merged_config
            
            for k in keys:
                if isinstance(config, dict) and k in config:
                    config = config[k]
                else:
                    return default
            
            return config
            
        except Exception:
            return default