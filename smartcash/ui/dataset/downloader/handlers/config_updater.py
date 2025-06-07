"""
File: smartcash/ui/dataset/downloader/handlers/config_updater.py
Deskripsi: Pembaruan UI components dari konfigurasi downloader sesuai dengan dataset_config.yaml
"""

from typing import Dict, Any

def update_downloader_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config downloader sesuai dengan dataset_config.yaml"""
    roboflow_config = config.get('data', {}).get('roboflow', {})
    download_config = config.get('download', {})
    uuid_config = config.get('uuid_renaming', {})
    validation_config = config.get('validation', {})
    cleanup_config = config.get('cleanup', {})
    file_naming_config = config.get('data', {}).get('file_naming', {})
    
    # One-liner component update dengan safe access
    safe_update = lambda key, value: setattr(ui_components[key], 'value', value) if key in ui_components and hasattr(ui_components[key], 'value') else None
    
    # Update settings dengan mapping approach
    field_mappings = [
        # Roboflow dataset settings
        ('workspace_input', roboflow_config, 'workspace', 'smartcash-wo2us'),
        ('project_input', roboflow_config, 'project', 'rupiah-emisi-2022'),
        ('version_input', roboflow_config, 'version', '3'),
        ('api_key_input', roboflow_config, 'api_key', ''),
        
        # Download options
        ('validate_checkbox', download_config, 'validate_download', True),
        ('backup_checkbox', download_config, 'backup_existing', False),
        ('target_dir', download_config, 'target_dir', 'data'),
        ('temp_dir', download_config, 'temp_dir', 'data/downloads'),
        ('organize_dataset', download_config, 'organize_dataset', True),
        ('rename_files', download_config, 'rename_files', True),
        ('retry_count', download_config, 'retry_count', 3),
        ('timeout', download_config, 'timeout', 30),
        ('chunk_size', download_config, 'chunk_size', 8192),
        ('parallel_downloads', download_config, 'parallel_downloads', False),
        ('max_workers', download_config, 'max_workers', 4),
        
        # UUID renaming settings
        ('uuid_enabled', uuid_config, 'enabled', True),
        ('uuid_backup_before_rename', uuid_config, 'backup_before_rename', False),
        ('uuid_batch_size', uuid_config, 'batch_size', 1000),
        ('uuid_parallel_workers', uuid_config, 'parallel_workers', 4),
        ('uuid_validate_consistency', uuid_config, 'validate_consistency', True),
        ('uuid_progress_reporting', uuid_config, 'progress_reporting', True),
        
        # File naming settings
        ('uuid_format', file_naming_config, 'uuid_format', True),
        ('naming_strategy', file_naming_config, 'naming_strategy', 'research_uuid'),
        ('preserve_original', file_naming_config, 'preserve_original', False),
        
        # Validation settings
        ('validation_enabled', validation_config, 'enabled', True),
        ('check_file_integrity', validation_config, 'check_file_integrity', True),
        ('verify_image_format', validation_config, 'verify_image_format', True),
        ('validate_labels', validation_config, 'validate_labels', True),
        ('check_dataset_structure', validation_config, 'check_dataset_structure', True),
        ('max_image_size_mb', validation_config, 'max_image_size_mb', 50),
        ('generate_report', validation_config, 'generate_report', True),
        
        # Cleanup settings
        ('auto_cleanup_downloads', cleanup_config, 'auto_cleanup_downloads', False),
        ('preserve_original_structure', cleanup_config, 'preserve_original_structure', True),
        ('backup_dir', cleanup_config, 'backup_dir', 'data/backup/downloads'),
        ('keep_download_logs', cleanup_config, 'keep_download_logs', True),
        ('cleanup_on_error', cleanup_config, 'cleanup_on_error', True)
    ]
    
    # Apply all mappings dengan one-liner approach
    [safe_update(component_key, source_config.get(config_key, default_value)) for component_key, source_config, config_key, default_value in field_mappings]
    
    # Special handling untuk version field (string conversion)
    try:
        version_value = roboflow_config.get('version', '3')
        safe_update('version_input', str(version_value))
    except Exception:
        safe_update('version_input', '3')  # Default fallback


def reset_downloader_ui(ui_components: Dict[str, Any]) -> None:
    """Reset UI components ke default konfigurasi downloader"""
    try:
        from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config
        default_config = get_default_downloader_config()
        update_downloader_ui(ui_components, default_config)
    except Exception:
        # Fallback to basic reset jika config manager gagal
        _apply_basic_defaults(ui_components)


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
            ui_components[key].value = value


def update_api_key_status(ui_components: Dict[str, Any], api_key_info: Dict[str, Any]) -> None:
    """Update API key status display dengan info dari colab secrets"""
    try:
        from smartcash.ui.dataset.downloader.utils.colab_secrets import create_api_key_info_html
        
        api_key_status_widget = ui_components.get('api_key_status')
        if api_key_status_widget and hasattr(api_key_status_widget, 'value'):
            api_key_status_widget.value = create_api_key_info_html(api_key_info)
    except Exception:
        pass  # Silent fail untuk widget update issues


def validate_ui_inputs(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validate UI inputs dan return validation result"""
    errors = []
    warnings = []
    
    # Extract values untuk validation
    workspace = getattr(ui_components.get('workspace_input'), 'value', '').strip()
    project = getattr(ui_components.get('project_input'), 'value', '').strip()
    version = getattr(ui_components.get('version_input'), 'value', '').strip()
    api_key = getattr(ui_components.get('api_key_input'), 'value', '').strip()
    
    # Required field validation
    if not workspace:
        errors.append("Workspace wajib diisi")
    elif len(workspace) < 3:
        errors.append("Workspace minimal 3 karakter")
    
    if not project:
        errors.append("Project wajib diisi")
    elif len(project) < 3:
        errors.append("Project minimal 3 karakter")
    
    if not version:
        errors.append("Version wajib diisi")
    
    if not api_key:
        errors.append("API Key wajib diisi")
    elif len(api_key) < 10:
        errors.append("API Key terlalu pendek (minimal 10 karakter)")
    
    # Format validation
    if version and not version.isdigit():
        warnings.append("Version biasanya berupa angka")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'values': {
            'workspace': workspace,
            'project': project,
            'version': version,
            'api_key_masked': f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}" if len(api_key) > 8 else '****'
        }
    }