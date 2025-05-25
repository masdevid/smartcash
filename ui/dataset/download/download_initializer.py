"""
File: smartcash/ui/dataset/download/download_initializer.py
Deskripsi: Enhanced silent initializer dengan comprehensive error handling dan modular setup
"""

from typing import Dict, Any, Optional
from smartcash.common.config.manager import get_config_manager
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge

# Konstanta namespace
from smartcash.ui.utils.ui_logger_namespace import DOWNLOAD_LOGGER_NAMESPACE, KNOWN_NAMESPACES
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[DOWNLOAD_LOGGER_NAMESPACE]

# Import handlers dan components
from smartcash.ui.dataset.download.handlers.button_handlers import setup_button_handlers
from smartcash.ui.dataset.download.handlers.config_handlers import setup_config_handlers
from smartcash.ui.dataset.download.handlers.progress_handlers import setup_progress_handlers
from smartcash.ui.dataset.download.components import create_download_ui

# Flag global untuk mencegah inisialisasi ulang
_DOWNLOAD_MODULE_INITIALIZED = False
_CACHED_UI_COMPONENTS = None

def initialize_dataset_download_ui(env=None, config=None, force_refresh=False) -> Any:
    """
    Enhanced inisialisasi UI download dataset dengan caching dan error recovery.
    
    Args:
        env: Environment context
        config: Custom configuration
        force_refresh: Force refresh UI components
        
    Returns:
        UI widget untuk download dataset
    """
    global _DOWNLOAD_MODULE_INITIALIZED, _CACHED_UI_COMPONENTS
    
    # Return cached UI jika sudah initialized dan tidak force refresh
    if _DOWNLOAD_MODULE_INITIALIZED and _CACHED_UI_COMPONENTS and not force_refresh:
        return _get_cached_ui_or_refresh(config)
    
    try:
        # Setup log suppression sebelum initialization
        _setup_comprehensive_log_suppression()
        
        # Get dan merge config (silent)
        merged_config = _get_merged_config(config)
        
        # Create UI components dengan error handling
        ui_components = _create_ui_components_safe(merged_config)
        if not ui_components:
            return _create_error_fallback_ui("Failed to create UI components")
        
        # Setup enhanced logger bridge
        logger_bridge = _setup_logger_bridge_safe(ui_components)
        ui_components['logger'] = logger_bridge
        ui_components['logger_namespace'] = DOWNLOAD_LOGGER_NAMESPACE
        ui_components['download_initialized'] = True
        
        # Setup handlers dengan comprehensive error handling
        ui_components = _setup_handlers_comprehensive(ui_components, merged_config, env)
        
        # Validation dan final setup
        validation_result = _validate_and_finalize_setup(ui_components)
        if not validation_result['valid']:
            return _create_error_fallback_ui(validation_result['message'])
        
        # Cache UI components
        _CACHED_UI_COMPONENTS = ui_components
        _DOWNLOAD_MODULE_INITIALIZED = True
        
        # Log sukses hanya jika logger tersedia
        if logger_bridge:
            logger_bridge.success("✅ UI download dataset siap digunakan")
        
        return ui_components['ui']
        
    except Exception as e:
        return _create_error_fallback_ui(f"Initialization error: {str(e)}")

def _get_cached_ui_or_refresh(config=None) -> Any:
    """Get cached UI atau refresh dengan config baru."""
    global _CACHED_UI_COMPONENTS
    
    try:
        if not _CACHED_UI_COMPONENTS:
            return initialize_dataset_download_ui(config=config, force_refresh=True)
        
        # Update config jika ada perubahan
        if config:
            _update_cached_config(config)
        
        return _CACHED_UI_COMPONENTS['ui']
        
    except Exception as e:
        return _create_error_fallback_ui(f"Cache refresh error: {str(e)}")

def _get_merged_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Get merged configuration dengan safe error handling."""
    try:
        config_manager = get_config_manager()
        
        # Load saved config dengan safe method
        saved_config = {}
        try:
            if hasattr(config_manager, 'get_config'):
                saved_config = config_manager.get_config('dataset') or {}
        except Exception:
            pass
        
        # Start dengan default config
        merged_config = {
            'workspace': 'smartcash-wo2us',
            'project': 'rupiah-emisi-2022', 
            'version': '3',
            'validate_dataset': True,
            'organize_dataset': True
        }
        
        # Merge dengan saved config
        if saved_config:
            merged_config.update(saved_config)
        
        # Merge dengan parameter config
        if config:
            merged_config.update(config)
        
        return merged_config
        
    except Exception:
        # Fallback ke minimal config
        return {
            'workspace': 'smartcash-wo2us',
            'project': 'rupiah-emisi-2022',
            'version': '3'
        }

def _create_ui_components_safe(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create UI components dengan comprehensive error handling."""
    try:
        return create_download_ui(config)
    except Exception:
        # Try dengan minimal config
        try:
            minimal_config = {
                'workspace': 'smartcash-wo2us',
                'project': 'rupiah-emisi-2022',
                'version': '3'
            }
            return create_download_ui(minimal_config)
        except Exception:
            return None

def _setup_logger_bridge_safe(ui_components: Dict[str, Any]) -> Optional[Any]:
    """Setup logger bridge dengan error handling."""
    try:
        return create_ui_logger_bridge(ui_components, DOWNLOAD_LOGGER_NAMESPACE)
    except Exception:
        # Return None jika logger bridge gagal, UI masih bisa berfungsi
        return None

def _setup_handlers_comprehensive(ui_components: Dict[str, Any], 
                                config: Dict[str, Any], 
                                env) -> Dict[str, Any]:
    """Setup handlers dengan comprehensive error recovery."""
    
    setup_results = {
        'config_handlers': False,
        'progress_handlers': False, 
        'button_handlers': False
    }
    
    # Setup config handlers
    try:
        ui_components = setup_config_handlers(ui_components, config)
        setup_results['config_handlers'] = True
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.warning(f"⚠️ Config handlers setup failed: {str(e)}")
    
    # Setup progress handlers
    try:
        ui_components = setup_progress_handlers(ui_components)
        setup_results['progress_handlers'] = True
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.warning(f"⚠️ Progress handlers setup failed: {str(e)}")
    
    # Setup button handlers (most critical)
    try:
        ui_components = setup_button_handlers(ui_components, env)
        setup_results['button_handlers'] = True
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Button handlers setup failed: {str(e)}")
    
    # Store setup results untuk debugging
    ui_components['_setup_results'] = setup_results
    
    return ui_components

def _validate_and_finalize_setup(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validate setup dan finalize configuration."""
    
    # Critical components validation
    critical_components = ['ui', 'download_button', 'check_button']
    missing_critical = [comp for comp in critical_components if comp not in ui_components]
    
    if missing_critical:
        return {
            'valid': False,
            'message': f"Critical components missing: {', '.join(missing_critical)}"
        }
    
    # Button functionality validation
    button_validation = _validate_button_functionality(ui_components)
    if not button_validation['valid']:
        return button_validation
    
    # Setup final configurations
    ui_components['module_initialized'] = True
    ui_components['initialization_timestamp'] = _get_timestamp()
    
    return {'valid': True, 'message': 'Setup validation passed'}

def _validate_button_functionality(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validate button functionality."""
    
    button_keys = ['download_button', 'check_button', 'cleanup_button', 'reset_button', 'save_button']
    functional_buttons = []
    
    for button_key in button_keys:
        if (button_key in ui_components and 
            ui_components[button_key] is not None and
            hasattr(ui_components[button_key], 'on_click')):
            functional_buttons.append(button_key)
    
    # Minimal requirement: download dan check button harus functional
    if 'download_button' not in functional_buttons:
        return {
            'valid': False,
            'message': 'Download button tidak functional'
        }
    
    if 'check_button' not in functional_buttons:
        return {
            'valid': False,
            'message': 'Check button tidak functional'  
        }
    
    return {
        'valid': True,
        'functional_buttons': functional_buttons,
        'total_functional': len(functional_buttons)
    }

def _update_cached_config(new_config: Dict[str, Any]) -> None:
    """Update cached UI components dengan config baru."""
    global _CACHED_UI_COMPONENTS
    
    if not _CACHED_UI_COMPONENTS:
        return
    
    try:
        # Update field values jika component ada
        field_mapping = {
            'workspace': 'workspace',
            'project': 'project',
            'version': 'version',
            'output_dir': 'output_dir'
        }
        
        for config_key, ui_key in field_mapping.items():
            if (config_key in new_config and 
                ui_key in _CACHED_UI_COMPONENTS and
                hasattr(_CACHED_UI_COMPONENTS[ui_key], 'value')):
                _CACHED_UI_COMPONENTS[ui_key].value = new_config[config_key]
                
    except Exception:
        # Silent fail untuk config update
        pass

def _setup_comprehensive_log_suppression() -> None:
    """Setup comprehensive log suppression untuk initialization."""
    import logging
    import sys
    
    # Suppress common loggers yang muncul saat initialization
    loggers_to_suppress = [
        'smartcash.common.environment',
        'smartcash.common.config.manager',
        'smartcash.common.logger',
        'smartcash.ui.utils.logger_bridge',
        'requests', 'urllib3', 'http.client',
        'ipywidgets', 'traitlets',
        'matplotlib', 'PIL', 'tensorflow'
    ]
    
    for logger_name in loggers_to_suppress:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
        logger.handlers.clear()

def _create_error_fallback_ui(error_message: str):
    """Create enhanced error fallback UI dengan actionable information."""
    import ipywidgets as widgets
    
    error_html = f"""
    <div style="padding: 20px; background: #fff3cd; border: 1px solid #ffc107; 
                border-radius: 8px; color: #856404; margin: 10px 0; max-width: 800px;">
        <h4 style="color: #721c24; margin-top: 0;">⚠️ Error Inisialisasi UI Download</h4>
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
                <li>Pastikan tidak ada error pada cell-cell sebelumnya</li>
            </ol>
        </div>
        <div style="margin: 15px 0; padding: 10px; background: #e8f4fd; border-radius: 5px;">
            <strong>💡 Quick Fix:</strong> Jalankan <code>reset_download_module()</code> kemudian coba lagi
        </div>
    </div>
    """
    
    return widgets.HTML(error_html)

def _get_timestamp() -> str:
    """Get current timestamp untuk tracking."""
    import datetime
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Public utility functions
def reset_download_module() -> None:
    """Reset module initialization - untuk recovery dari error."""
    global _DOWNLOAD_MODULE_INITIALIZED, _CACHED_UI_COMPONENTS
    _DOWNLOAD_MODULE_INITIALIZED = False
    _CACHED_UI_COMPONENTS = None

def get_download_ui_components(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Helper untuk mendapatkan UI components tanpa inisialisasi penuh.
    
    Args:
        config: Optional configuration
        
    Returns:
        Dictionary UI components
    """
    try:
        merged_config = _get_merged_config(config)
        ui_components = _create_ui_components_safe(merged_config)
        
        if not ui_components:
            return {}
        
        # Setup basic logger bridge
        logger_bridge = _setup_logger_bridge_safe(ui_components)
        ui_components['logger'] = logger_bridge
        ui_components['logger_namespace'] = DOWNLOAD_LOGGER_NAMESPACE
        ui_components['download_initialized'] = True
        
        return ui_components
        
    except Exception:
        return {}

def get_module_status() -> Dict[str, Any]:
    """Get status informasi module untuk debugging."""
    global _DOWNLOAD_MODULE_INITIALIZED, _CACHED_UI_COMPONENTS
    
    status = {
        'initialized': _DOWNLOAD_MODULE_INITIALIZED,
        'cached_available': _CACHED_UI_COMPONENTS is not None,
        'timestamp': _get_timestamp()
    }
    
    if _CACHED_UI_COMPONENTS:
        status.update({
            'setup_results': _CACHED_UI_COMPONENTS.get('_setup_results', {}),
            'logger_available': 'logger' in _CACHED_UI_COMPONENTS,
            'ui_available': 'ui' in _CACHED_UI_COMPONENTS,
            'initialization_time': _CACHED_UI_COMPONENTS.get('initialization_timestamp', 'Unknown')
        })
    
    return status