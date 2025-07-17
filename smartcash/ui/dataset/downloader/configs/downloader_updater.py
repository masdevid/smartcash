"""
File: smartcash/ui/dataset/downloader/handlers/config_updater.py
Deskripsi: Pembaruan UI components dari konfigurasi downloader sesuai dengan dataset_config.yaml
"""

from typing import Dict, Any

def update_downloader_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config downloader sesuai dengan dataset_config.yaml"""
    from .downloader_config_constants import UI_FIELD_MAPPINGS, get_nested_value
    
    # One-liner component update dengan safe access
    safe_update = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Apply all mappings using centralized constants
    for ui_key, config_path, config_key, default_value in UI_FIELD_MAPPINGS:
        # Get value from config using dot notation
        full_path = f"{config_path}.{config_key}" if config_path else config_key
        config_value = get_nested_value(config, full_path, default_value)
        
        # Update UI component
        safe_update(ui_key, config_value)
    
    # Special handling untuk version field (string conversion)
    try:
        version_value = get_nested_value(config, 'data.roboflow.version', '3')
        safe_update('version_input', str(version_value))
    except Exception:
        safe_update('version_input', '3')  # Default fallback


def reset_downloader_ui(ui_components: Dict[str, Any]) -> None:
    """Reset UI components ke default konfigurasi downloader"""
    try:
        # Preserve current API key
        current_api_key = getattr(ui_components.get('api_key_input'), 'value', '').strip()
        
        from smartcash.ui.dataset.downloader.configs.downloader_defaults import get_default_downloader_config
        default_config = get_default_downloader_config()
        
        # Preserve API key di config
        if current_api_key:
            default_config['data']['roboflow']['api_key'] = current_api_key
        
        update_downloader_ui(ui_components, default_config)
    except Exception:
        # Fallback with preserved API key
        current_api_key = getattr(ui_components.get('api_key_input'), 'value', '').strip()
        _apply_basic_defaults(ui_components)
        # Restore API key after basic defaults
        if current_api_key and 'api_key_input' in ui_components:
            ui_components['api_key_input'].value = current_api_key


def _apply_basic_defaults(ui_components: Dict[str, Any]) -> None:
    """Apply basic defaults ke UI components jika config manager tidak tersedia"""
    basic_defaults = {
        'workspace_input': 'smartcash-wo2us',
        'project_input': 'rupiah-emisi-2022',
        'version_input': '3',
        'api_key_input': '',
        'validate_checkbox': True,
        'backup_checkbox': False,
        'target_dir': 'data',
        'temp_dir': 'data/downloads',
        'organize_dataset': True,
        'rename_files': True,
        'retry_count': 3,
        'timeout': 30,
        'chunk_size': 8192,
        'uuid_enabled': True,
        'validation_enabled': True,
        'auto_cleanup_downloads': False
    }
    
    for key, value in basic_defaults.items():
        if key in ui_components and hasattr(ui_components[key], 'value'):
            try:
                ui_components[key].value = value
            except Exception:
                pass  # Silent fail untuk widget issues


def update_api_key_status(ui_components: Dict[str, Any], api_key_info: Dict[str, Any]) -> None:
    """Update API key status display dengan info dari colab secrets"""
    try:
        from smartcash.ui.core.mixins.colab_secrets_mixin import ColabSecretsMixin
        
        secrets_mixin = ColabSecretsMixin()
        api_key_status_widget = ui_components.get('api_key_status')
        if api_key_status_widget and hasattr(api_key_status_widget, 'value'):
            api_key_status_widget.value = secrets_mixin.create_api_key_info_html(api_key_info)
    except Exception as e:
        logger.debug(f"Error updating API key status: {e}")
        pass  # Silent fail untuk widget update issues


def validate_ui_inputs(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validate UI inputs dan return validation result"""
    from .downloader_config_constants import REQUIRED_FIELDS, OPTIONAL_FIELDS_WITH_WARNINGS
    
    try:
        # Extract values untuk validation
        get_value = lambda key: getattr(ui_components.get(key), 'value', '').strip()
        
        values = {
            'workspace': get_value('workspace_input'),
            'project': get_value('project_input'),
            'version': get_value('version_input'),
            'api_key': get_value('api_key_input')
        }
        
        # Validate required fields using centralized constants
        errors = []
        warnings = []
        
        # Check required fields
        for field_path, error_message in REQUIRED_FIELDS.items():
            field_key = field_path.split('.')[-1]  # Get the last part (e.g., 'workspace')
            if not values.get(field_key):
                errors.append(error_message)
        
        # Check optional fields with warnings
        for field_path, warning_message in OPTIONAL_FIELDS_WITH_WARNINGS.items():
            field_key = field_path.split('.')[-1]  # Get the last part (e.g., 'api_key')
            if not values.get(field_key):
                warnings.append(warning_message)
        
        valid = len(errors) == 0
        
        return {
            'valid': valid,
            'status': valid,
            'errors': errors,
            'warnings': warnings,
            'values': {
                **values,
                'api_key_masked': '****' if values['api_key'] else ''
            }
        }
    except Exception as e:
        # Fallback validation
        return {
            'valid': False,
            'status': False,
            'errors': [f"Error validasi: {str(e)}"],
            'warnings': [],
            'values': {
                'workspace': getattr(ui_components.get('workspace_input'), 'value', '').strip(),
                'project': getattr(ui_components.get('project_input'), 'value', '').strip(),
                'version': getattr(ui_components.get('version_input'), 'value', '').strip(),
                'api_key_masked': '****'
            }
        }