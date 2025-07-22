"""
Helper functions for Colab UI.

This module contains utility functions for configuration and validation.
"""

from typing import Dict, Any

def get_colab_default_config() -> Dict[str, Any]:
    """Get default configuration for the Colab module.
    
    Returns:
        Default configuration dictionary
    """
    # Use unified configuration from main config module
    from smartcash.ui.setup.colab.configs.colab_defaults import get_default_colab_config
    return get_default_colab_config()

# Legacy alias for backward compatibility
DEFAULT_CONFIG = get_colab_default_config()

def validate_colab_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate Colab configuration against defined rules.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ValueError: If configuration validation fails
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")
    
    # Create a copy to avoid modifying the original
    validated = get_colab_default_config()
    
    # Deep merge the configuration 
    def deep_merge(default: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = default.copy()
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    validated = deep_merge(validated, config)
    
    # Validate key sections exist
    required_sections = ['environment', 'setup', 'paths']
    for section in required_sections:
        if section not in validated:
            raise ValueError(f"Required configuration section '{section}' is missing")
    
    # Validate environment section
    env_config = validated.get('environment', {})
    if not isinstance(env_config.get('auto_mount_drive'), bool):
        validated['environment']['auto_mount_drive'] = True
        
    # Validate paths section  
    paths_config = validated.get('paths', {})
    if not isinstance(paths_config.get('drive_base'), str) or not paths_config.get('drive_base'):
        validated['paths']['drive_base'] = "/content/drive/MyDrive/SmartCash"
    
    return validated

def update_colab_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration from UI components.
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        Updated configuration dictionary
    """
    config = {}
    
    # Extract values from form widgets
    form_widgets = ui_components.get('form_widgets', {})
    
    if 'auto_detect' in form_widgets:
        config['auto_detect'] = form_widgets['auto_detect'].value
    
    if 'drive_path' in form_widgets:
        config['drive_path'] = form_widgets['drive_path'].value
    
    if 'project_name' in form_widgets:
        config['project_name'] = form_widgets['project_name'].value
    
    return config
