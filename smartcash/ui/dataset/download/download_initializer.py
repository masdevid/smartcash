"""
File: smartcash/ui/dataset/download/download_initializer.py
Deskripsi: Fixed download initializer dengan integrasi latest progress_tracking dan button_state_manager
"""

import logging
from typing import Dict, Any, Optional
from smartcash.ui.utils.ui_logger_namespace import ENV_CONFIG_LOGGER_NAMESPACE
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
from smartcash.ui.dataset.download.components.main_ui import create_download_ui
from smartcash.ui.dataset.download.handlers.config_handlers import setup_config_handlers
from smartcash.ui.dataset.download.handlers.button_handlers import setup_button_handlers

# Global state management
_DOWNLOAD_INITIALIZED = False
_CACHED_UI_COMPONENTS = None

def initialize_download_ui(env=None, config=None, force_refresh=False):
    """
    Main initializer untuk download UI dengan latest progress tracking integration.
    
    Args:
        env: Environment context (optional)
        config: Configuration override
        force_refresh: Force refresh cached UI
        
    Returns:
        Main UI widget dengan latest integrations
    """
    global _DOWNLOAD_INITIALIZED, _CACHED_UI_COMPONENTS
    
    # Return cached UI jika sudah initialized
    if _DOWNLOAD_INITIALIZED and _CACHED_UI_COMPONENTS and not force_refresh:
        return _get_cached_ui_or_refresh(config)
    
    try:
        # 1. Setup comprehensive log suppression
        _setup_comprehensive_log_suppression()
        
        # 2. Get dan merge config
        merged_config = _get_merged_config(config)
        
        # 3. Create UI components dengan latest progress tracking
        ui_components = _create_ui_components_safe(merged_config)
        
        # 4. Setup logger bridge
        logger_bridge = _setup_logger_bridge_safe(ui_components)
        ui_components['logger'] = logger_bridge
        ui_components['logger_namespace'] = ENV_CONFIG_LOGGER_NAMESPACE
        
        # 5. Setup config handlers
        ui_components = setup_config_handlers(ui_components, merged_config)
        
        # 6. Setup button handlers dengan latest integration
        ui_components = setup_button_handlers(ui_components, env)
        
        # 7. Validation dan final setup
        validation_result = _validate_and_finalize_setup(ui_components)
        
        if not validation_result['valid']:
            logger_bridge.warning(f"⚠️ Setup validation issues: {validation_result['message']}")
        
        # 8. Cache dan return
        _CACHED_UI_COMPONENTS = ui_components
        _DOWNLOAD_INITIALIZED = True
        
        logger_bridge.success("🎉 Download UI berhasil diinisialisasi dengan latest integrations")
        
        return ui_components['ui']
        
    except Exception as e:
        return _create_error_fallback_ui(f"Initialization error: {str(e)}")

def _setup_comprehensive_log_suppression():
    """Setup comprehensive log suppression untuk clean output."""
    suppressed_loggers = [
        'requests', 'urllib3', 'http.client', 'requests.packages.urllib3',
        'googleapiclient', 'google.auth', 'google_auth_httplib2',
        'ipykernel', 'tornado', 'asyncio', 'matplotlib'
    ]
    
    for logger_name in suppressed_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False

def _get_merged_config(config):
    """Get dan merge configuration dari berbagai sumber."""
    try:
        from smartcash.common.config.manager import get_config_manager
        
        # Default config
        default_config = {
            'roboflow': {
                'workspace': 'smartcash-wo2us',
                'project': 'rupiah-emisi-2022',
                'version': '3'
            }
        }
        
        # Load saved config
        config_manager = get_config_manager()
        saved_config = config_manager.get_config('download') or {}
        
        # Merge: default -> saved -> provided
        merged = default_config.copy()
        if saved_config:
            _deep_merge(merged, saved_config)
        if config:
            _deep_merge(merged, config)
        
        return merged
        
    except Exception:
        return config or {}

def _deep_merge(target, source):
    """Deep merge dictionaries."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value

def _create_ui_components_safe(config):
    """Create UI components dengan error handling dan latest progress integration."""
    try:
        ui_components = create_download_ui(config)
        
        # Verify latest progress tracking integration
        required_progress_methods = [
            'show_for_operation', 'update_progress', 'complete_operation', 
            'error_operation', 'reset_all', 'tracker'
        ]
        
        missing_methods = [method for method in required_progress_methods if method not in ui_components]
        
        if missing_methods:
            # Add fallback progress methods
            _add_progress_fallbacks(ui_components, missing_methods)
        
        return ui_components
        
    except Exception as e:
        raise Exception(f"UI component creation failed: {str(e)}")

def _add_progress_fallbacks(ui_components: Dict[str, Any], missing_methods: list) -> None:
    """Add fallback progress methods untuk missing integrations."""
    
    fallback_implementations = {
        'show_for_operation': lambda operation: None,
        'update_progress': lambda progress_type, value, message, color=None: None,
        'complete_operation': lambda message: None,
        'error_operation': lambda message: None,
        'reset_all': lambda: None,
        'tracker': None  # Will be handled separately if needed
    }
    
    for method in missing_methods:
        if method in fallback_implementations:
            ui_components[method] = fallback_implementations[method]

def _setup_logger_bridge_safe(ui_components):
    """Setup logger bridge dengan error handling."""
    try:
        logger_bridge = create_ui_logger_bridge(ui_components, ENV_CONFIG_LOGGER_NAMESPACE)
        return logger_bridge
    except Exception as e:
        # Fallback logger
        import logging
        fallback_logger = logging.getLogger('download_fallback')
        fallback_logger.error(f"Logger bridge setup failed: {str(e)}")
        return fallback_logger

def _validate_and_finalize_setup(ui_components):
    """Validate dan finalize setup dengan comprehensive checks."""
    validation_result = {
        'valid': True,
        'message': '',
        'issues': []
    }
    
    # Check required components
    required_components = [
        'ui', 'download_button', 'check_button', 'cleanup_button',
        'save_button', 'reset_button', 'logger'
    ]
    
    missing_components = [comp for comp in required_components if comp not in ui_components]
    
    if missing_components:
        validation_result['issues'].append(f"Missing components: {', '.join(missing_components)}")
    
    # Check progress tracking integration
    progress_methods = ['show_for_operation', 'update_progress', 'complete_operation', 'error_operation']
    missing_progress = [method for method in progress_methods if method not in ui_components]
    
    if missing_progress:
        validation_result['issues'].append(f"Missing progress methods: {', '.join(missing_progress)}")
    
    # Check button state manager integration
    if 'button_state_manager' not in ui_components:
        try:
            from smartcash.ui.utils.button_state_manager import get_button_state_manager
            ui_components['button_state_manager'] = get_button_state_manager(ui_components)
        except Exception as e:
            validation_result['issues'].append(f"Button state manager setup failed: {str(e)}")
    
    # Compile validation result
    validation_result['valid'] = len(validation_result['issues']) == 0
    validation_result['message'] = '; '.join(validation_result['issues']) if validation_result['issues'] else 'All validations passed'
    
    return validation_result

def _get_cached_ui_or_refresh(config):
    """Get cached UI atau refresh jika diperlukan."""
    global _CACHED_UI_COMPONENTS
    
    if not _CACHED_UI_COMPONENTS:
        return initialize_download_ui(config=config, force_refresh=True)
    
    # Check apakah cached UI masih valid
    if 'ui' not in _CACHED_UI_COMPONENTS:
        return initialize_download_ui(config=config, force_refresh=True)
    
    # Update config jika disediakan
    if config:
        try:
            _CACHED_UI_COMPONENTS = setup_config_handlers(_CACHED_UI_COMPONENTS, config)
        except Exception:
            pass
    
    return _CACHED_UI_COMPONENTS['ui']

def _create_error_fallback_ui(error_message: str):
    """Create enhanced error fallback UI dengan latest styling."""
    import ipywidgets as widgets
    
    error_html = f"""
    <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffc107; 
                border-radius: 8px; color: #856404; margin: 10px 0;">
        <h4 style="color: #721c24; margin-top: 0;">⚠️ Error Inisialisasi Download UI</h4>
        <div style="margin: 15px 0;">
            <strong>Error Detail:</strong><br>
            <code style="background: #f8f9fa; padding: 5px; border-radius: 3px; font-size: 12px;">
                {error_message}
            </code>
        </div>
        <div style="margin: 15px 0;">
            <strong>🔧 Solusi yang Bisa Dicoba:</strong>
            <ol style="margin: 10px 0; padding-left: 20px;">
                <li>Restart kernel Colab dan jalankan ulang cell</li>
                <li>Clear output semua cell dan jalankan dari awal</li>
                <li>Periksa koneksi internet dan Google Drive</li>
                <li>Pastikan semua dependencies terinstall dengan benar</li>
            </ol>
        </div>
        <div style="margin: 15px 0; padding: 10px; background: #e3f2fd; border-radius: 4px;">
            <strong>💡 Debug Info:</strong><br>
            <small>Latest progress tracking integration: {_check_progress_integration_status()}</small><br>
            <small>Button state manager: {_check_button_manager_status()}</small>
        </div>
    </div>
    """
    
    return widgets.HTML(error_html)

def _check_progress_integration_status() -> str:
    """Check status integrasi progress tracking."""
    try:
        from smartcash.ui.components.progress_tracking import create_progress_tracking_container
        container = create_progress_tracking_container()
        required_methods = ['show_for_operation', 'update_progress', 'complete_operation']
        has_methods = all(method in container for method in required_methods)
        return "✅ Available" if has_methods else "⚠️ Partial"
    except Exception:
        return "❌ Not Available"

def _check_button_manager_status() -> str:
    """Check status button state manager."""
    try:
        from smartcash.ui.utils.button_state_manager import get_button_state_manager
        return "✅ Available"
    except Exception:
        return "❌ Not Available"

def reset_download_ui_cache():
    """Reset cached UI untuk force refresh pada pemanggilan berikutnya."""
    global _DOWNLOAD_INITIALIZED, _CACHED_UI_COMPONENTS
    _DOWNLOAD_INITIALIZED = False
    _CACHED_UI_COMPONENTS = None

def get_download_ui_status() -> Dict[str, Any]:
    """Get status download UI untuk debugging."""
    global _DOWNLOAD_INITIALIZED, _CACHED_UI_COMPONENTS
    
    status = {
        'initialized': _DOWNLOAD_INITIALIZED,
        'cached_available': _CACHED_UI_COMPONENTS is not None,
        'progress_integration': _check_progress_integration_status(),
        'button_manager': _check_button_manager_status()
    }
    
    if _CACHED_UI_COMPONENTS:
        status.update({
            'has_ui': 'ui' in _CACHED_UI_COMPONENTS,
            'has_logger': 'logger' in _CACHED_UI_COMPONENTS,
            'has_progress_methods': all(
                method in _CACHED_UI_COMPONENTS 
                for method in ['show_for_operation', 'update_progress', 'complete_operation']
            ),
            'component_count': len(_CACHED_UI_COMPONENTS)
        })
    
    return status

# Export functions untuk backward compatibility
__all__ = [
    'initialize_download_ui',
    'reset_download_ui_cache', 
    'get_download_ui_status'
]