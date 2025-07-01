"""
File: smartcash/ui/dataset/downloader/handlers/config_updater.py
Deskripsi: Pembaruan UI components dari konfigurasi downloader sesuai dengan dataset_config.yaml
"""

from typing import Dict, Any, Optional, List, TypeVar, Callable, Union

from smartcash.ui.utils.ui_logger import get_module_logger
from smartcash.ui.handlers.error_handler import handle_ui_errors, create_error_response
from smartcash.ui.utils.error_utils import ErrorHandler, ErrorContext
from smartcash.ui.utils.fallback_utils import safe_execute
from smartcash.common.worker_utils import get_optimal_worker_count

# Initialize module logger
logger = get_module_logger('smartcash.ui.dataset.downloader.config_updater')
T = TypeVar('T')

@handle_ui_errors(error_component_title="UI Update Error")
def update_downloader_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config downloader sesuai dengan dataset_config.yaml"""
    logger.debug("Updating UI components from config")
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
        ('max_workers', download_config, 'max_workers', get_optimal_worker_count('io')),
        
        # UUID renaming settings
        ('uuid_enabled', uuid_config, 'enabled', True),
        ('uuid_backup_before_rename', uuid_config, 'backup_before_rename', False),
        ('uuid_batch_size', uuid_config, 'batch_size', 1000),
        ('uuid_parallel_workers', uuid_config, 'parallel_workers', get_optimal_worker_count('mixed')),
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
    updated_count = 0
    for component_key, source_config, config_key, default_value in field_mappings:
        try:
            if component_key in ui_components and hasattr(ui_components[component_key], 'value'):
                ui_components[component_key].value = source_config.get(config_key, default_value)
                updated_count += 1
        except Exception as e:
            logger.debug(f"Could not update {component_key}: {e}")
    
    logger.debug(f"Updated {updated_count} UI components from config")
    
    # Special handling untuk version field (string conversion)
    try:
        version_value = roboflow_config.get('version', '3')
        safe_update('version_input', str(version_value))
    except Exception:
        safe_update('version_input', '3')  # Default fallback


@handle_ui_errors(error_component_title="UI Reset Error")
def reset_downloader_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Reset UI components ke default konfigurasi downloader
    
    Returns:
        Dict[str, Any]: Status response with format {'status': str, 'error': Optional[str]}
    """
    # Create error context for better tracing
    ctx = ErrorContext(
        component="reset_downloader_ui",
        operation="reset_ui"
    )
    try:
        # Preserve current API key
        current_api_key = getattr(ui_components.get('api_key_input'), 'value', '').strip()
        logger.debug("Resetting UI components to defaults")
        
        from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config
        default_config = get_default_downloader_config()
        
        # Preserve API key di config
        if current_api_key:
            default_config['data']['roboflow']['api_key'] = current_api_key
            logger.debug("Preserved API key during reset")
        
        update_downloader_ui(ui_components, default_config)
        return {'status': 'success'}
    except Exception as e:
        logger.error(f"Error resetting UI components: {str(e)}", exc_info=True)
        # Fallback with preserved API key
        try:
            current_api_key = getattr(ui_components.get('api_key_input'), 'value', '').strip()
            _apply_basic_defaults(ui_components)
            # Restore API key after basic defaults
            if current_api_key and 'api_key_input' in ui_components:
                ui_components['api_key_input'].value = current_api_key
            return {'status': 'success', 'warning': 'Used fallback defaults'}
        except Exception as e2:
            return {'status': 'error', 'error': f"Failed to reset UI: {str(e2)}"}


@handle_ui_errors(error_component_title="Basic Defaults Error")
def _apply_basic_defaults(ui_components: Dict[str, Any]) -> None:
    """Apply basic defaults ke UI components jika config manager tidak tersedia"""
    # Create error context for better tracing
    ctx = ErrorContext(
        component="_apply_basic_defaults",
        operation="apply_defaults"
    )
    logger.debug("Applying basic defaults to UI components")
    
    # Import here to avoid circular imports
    from smartcash.ui.dataset.downloader.handlers.defaults import (
        get_default_workspace, get_default_project, get_default_version,
        is_uuid_enabled_by_default, is_validation_enabled_by_default
    )
    
    basic_defaults = {
        'workspace_input': get_default_workspace(),
        'project_input': get_default_project(),
        'version_input': get_default_version(),
        'api_key_input': '',
        'validate_checkbox': is_validation_enabled_by_default(),
        'backup_checkbox': False,
        'target_dir': 'data',
        'temp_dir': 'data/downloads',
        'organize_dataset': True,
        'rename_files': True,
        'retry_count': 3,
        'timeout': 30,
        'chunk_size': 8192,
        'uuid_enabled': is_uuid_enabled_by_default(),
        'validation_enabled': is_validation_enabled_by_default(),
        'auto_cleanup_downloads': False
    }
    
    updated_count = 0
    for key, value in basic_defaults.items():
        if key in ui_components and hasattr(ui_components[key], 'value'):
            try:
                ui_components[key].value = value
                updated_count += 1
            except Exception as e:
                logger.debug(f"Could not set default for {key}: {str(e)}")
    
    logger.debug(f"Applied {updated_count} basic defaults to UI components")


@handle_ui_errors(error_component_title="API Key Status Error")
def update_api_key_status(ui_components: Dict[str, Any], api_key_info: Dict[str, Any]) -> Dict[str, Any]:
    """Update API key status display dengan info dari colab secrets
    
    Returns:
        Dict[str, Any]: Status response with format {'status': str, 'error': Optional[str]}
    """
    # Create error context for better tracing
    ctx = ErrorContext(
        component="update_api_key_status",
        operation="update_key_status"
    )
    
    # Use ErrorHandler for consistent error handling
    handler = ErrorHandler(
        context=ctx,
        logger=logger
    )
    try:
        from smartcash.ui.dataset.downloader.utils.colab_secrets import create_api_key_info_html
        
        api_key_status_widget = ui_components.get('api_key_status')
        if not api_key_status_widget or not hasattr(api_key_status_widget, 'value'):
            error_msg = "API key status widget not found or missing value attribute"
            handler.handle_error(message=error_msg, error_level="warning")
            return {'status': 'warning', 'warning': error_msg}
            
        # Update widget with HTML content
        api_key_status_widget.value = create_api_key_info_html(api_key_info)
        logger.debug(f"Updated API key status: {api_key_info.get('source')}, valid={api_key_info.get('valid')}")
        return {'status': 'success'}
    except ImportError as e:
        error_msg = f"Missing required module: {str(e)}"
        handler.handle_error(error=e, message=error_msg)
        return {'status': 'error', 'error': error_msg}
    except AttributeError as e:
        error_msg = f"Invalid widget attribute: {str(e)}"
        handler.handle_error(error=e, message=error_msg)
        return {'status': 'error', 'error': error_msg}
    except Exception as e:
        error_msg = f"Error updating API key status: {str(e)}"
        handler.handle_error(error=e, message=error_msg)
        return {'status': 'error', 'error': error_msg}


@handle_ui_errors(error_component_title="Validation Error")
def validate_ui_inputs(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validate UI inputs dan return validation result"""
    errors = []
    warnings = []
    
    # Create error context for better tracing
    ctx = ErrorContext(
        component="validate_ui_inputs",
        operation="validate_inputs"
    )
    
    # Use ErrorHandler for consistent error handling
    handler = ErrorHandler(
        context=ctx,
        logger=logger
    )
    
    logger.debug("Validating UI inputs")
    
    try:
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
    
        validation_result = {
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
        
        logger.debug(f"Validation result: valid={validation_result['valid']}, errors={len(errors)}, warnings={len(warnings)}")
        return validation_result
    except Exception as e:
        logger.error(f"Error validating UI inputs: {str(e)}", exc_info=True)
        return {
            'valid': False,
            'errors': [f"Validation error: {str(e)}"],
            'warnings': [],
            'values': {}
        }

@handle_ui_errors(error_component_title="Status Panel Error")
def update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
    """Update status panel with proper error handling using fail-fast approach"""
    # Create error context for better tracing
    ctx = ErrorContext(
        component="update_status_panel",
        operation="update_status"
    )
    
    # Use ErrorHandler for consistent error handling
    handler = ErrorHandler(
        context=ctx,
        logger=logger
    )
    
    # Validate inputs first - fail fast if invalid
    if not message:
        error_msg = "Cannot update status panel with empty message"
        handler.handle_error(message=error_msg, error_level="warning")
        logger.warning(error_msg)
        return
        
    if not isinstance(status_type, str) or status_type not in ['info', 'warning', 'error', 'success']:
        error_msg = f"Invalid status type: {status_type}"
        handler.handle_error(message=error_msg, error_level="warning")
        # Default to info if invalid
        status_type = 'info'
    
    # Safe update with proper error handling
    if 'status_panel' not in ui_components or not hasattr(ui_components['status_panel'], 'update'):
        logger.info(f"Status update (no panel): [{status_type}] {message}")
        return
        
    # Use safe_execute to handle any exceptions during update
    safe_execute(
        lambda: ui_components['status_panel'].update(
            create_error_response(
                error_message=message,
                title="Status Update",
                error_type=status_type,
                include_traceback=False
            )
        ),
        error_handler=lambda e: handler.handle_error(
            error=e,
            message=f"Failed to update status panel: {str(e)}"
        )
    )
    
    logger.debug(f"Updated status panel: [{status_type}] {message}")