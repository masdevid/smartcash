"""
File: smartcash/ui/dataset/downloader/handlers/config_extractor.py
Deskripsi: Ekstraksi konfigurasi downloader dari UI components sesuai dengan dataset_config.yaml
"""

from typing import Dict, Any
from datetime import datetime
from .downloader_config_constants import (
    get_default_config_structure,
    UI_FIELD_MAPPINGS,
    get_nested_value,
    set_nested_value
)

def extract_downloader_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Ekstraksi konfigurasi downloader yang konsisten dengan dataset_config.yaml"""
    # Start with default configuration structure
    config = get_default_config_structure()
    
    # Update timestamp
    current_time = datetime.now().isoformat()
    config['updated_at'] = current_time
    config['history']['updated_at'] = current_time
    
    # Extract values from UI components using centralized mapping
    for ui_key, config_path, config_key, default_value in UI_FIELD_MAPPINGS:
        # Get value from UI component
        ui_value = getattr(ui_components.get(ui_key, type('', (), {'value': default_value})()), 'value', default_value)
        
        # Handle string stripping for text inputs
        if isinstance(ui_value, str):
            ui_value = ui_value.strip()
        
        # Set the value in config using dot notation
        full_path = f"{config_path}.{config_key}" if config_path else config_key
        set_nested_value(config, full_path, ui_value)
    
    return config

def extract_simplified_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract simplified configuration for operations that don't need full config structure.
    
    Args:
        ui_components: Dictionary of UI components
        
    Returns:
        Simplified configuration dictionary
    """
    # One-liner value extraction dengan fallback
    get_value = lambda key, default: getattr(ui_components.get(key, type('', (), {'value': default})()), 'value', default)
    
    return {
        'data': {
            'roboflow': {
                'workspace': get_value('workspace_input', '').strip(),
                'project': get_value('project_input', '').strip(),
                'version': get_value('version_input', '').strip(),
                'api_key': get_value('api_key_input', '').strip(),
            }
        },
        'validation': {
            'validate_download': get_value('validate_checkbox', True),
            'backup_existing': get_value('backup_checkbox', False),
        }
    }
