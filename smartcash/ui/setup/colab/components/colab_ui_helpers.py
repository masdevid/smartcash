"""
Helper functions for Colab UI.

This module contains utility functions for configuration and validation.
"""

from typing import Dict, Any

# Default configuration
DEFAULT_CONFIG = {
    'auto_detect': True,
    'drive_path': '/content/drive/MyDrive',
    'project_name': 'SmartCash',
    'environment': 'colab',
    'show_summary': True
}

def get_colab_default_config() -> Dict[str, Any]:
    """Get default configuration for the Colab module.
    
    Returns:
        Default configuration dictionary
    """
    return DEFAULT_CONFIG.copy()

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
    validated.update(config)
    
    # Validate individual fields
    if not isinstance(validated['auto_detect'], bool):
        raise ValueError("auto_detect must be a boolean")
        
    if not isinstance(validated['drive_path'], str) or not validated['drive_path']:
        raise ValueError("drive_path must be a non-empty string")
        
    if not isinstance(validated['project_name'], str) or not validated['project_name']:
        raise ValueError("project_name must be a non-empty string")
    
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
