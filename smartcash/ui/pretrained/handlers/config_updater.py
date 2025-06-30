# File: smartcash/ui/pretrained/handlers/config_updater.py
"""
File: smartcash/ui/pretrained/handlers/config_updater.py
Deskripsi: Config updater untuk pretrained models dengan safe update approach
"""

from typing import Dict, Any, Optional, Callable, TypeVar, Type, cast
from functools import wraps

# Type variables for generic function typing
T = TypeVar('T')
P = TypeVar('P')

# Type alias for logger bridge to support different logger implementations
LoggerBridge = Any

def with_error_handling(logger_bridge: Optional[LoggerBridge] = None):
    """Decorator untuk menangani error dan logging secara konsisten"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"❌ Error in {func.__name__}: {str(e)}"
                if logger_bridge and hasattr(logger_bridge, 'error'):
                    logger_bridge.error(error_msg, exc_info=True)
                else:
                    print(f"[ERROR] {error_msg}")
                raise  # Re-raise to allow caller to handle
        return wrapper
    return decorator


def _log_debug(logger_bridge: Optional[LoggerBridge], message: str) -> None:
    """Log debug message using logger_bridge if available"""
    if logger_bridge and hasattr(logger_bridge, 'debug'):
        logger_bridge.debug(message)


def _log_info(logger_bridge: Optional[LoggerBridge], message: str) -> None:
    """Log info message using logger_bridge if available"""
    if logger_bridge and hasattr(logger_bridge, 'info'):
        logger_bridge.info(message)


def _log_warning(logger_bridge: Optional[LoggerBridge], message: str) -> None:
    """Log warning message using logger_bridge if available"""
    if logger_bridge and hasattr(logger_bridge, 'warning'):
        logger_bridge.warning(message)


def _log_error(logger_bridge: Optional[LoggerBridge], message: str, exc_info: bool = False) -> None:
    """Log error message using logger_bridge if available"""
    if logger_bridge and hasattr(logger_bridge, 'error'):
        logger_bridge.error(message, exc_info=exc_info)


def update_pretrained_ui(
    ui_components: Dict[str, Any], 
    config: Dict[str, Any],
    logger_bridge: Optional[LoggerBridge] = None
) -> None:
    """🔄 Update UI components dengan konfigurasi yang dimuat
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Dictionary konfigurasi untuk diterapkan
        logger_bridge: Logger bridge instance untuk logging (opsional)
        
    Raises:
        ValueError: Jika update UI gagal atau data tidak valid
    """
    if not ui_components or not isinstance(ui_components, dict):
        error_msg = "UI components tidak valid untuk update"
        _log_error(logger_bridge, error_msg)
        raise ValueError(error_msg)
    
    if not config or not isinstance(config, dict):
        error_msg = "Config tidak valid untuk update UI"
        _log_error(logger_bridge, error_msg)
        raise ValueError(error_msg)
    
    pretrained_config = config.get('pretrained_models', {})
    
    # Update models directory
    if 'models_dir_input' in ui_components and 'models_dir' in pretrained_config:
        ui_components['models_dir_input'].value = pretrained_config['models_dir']
        _log_debug(logger_bridge, f"Updated models_dir: {pretrained_config['models_dir']}")
    
    # Update drive models directory
    if 'drive_models_dir_input' in ui_components and 'drive_models_dir' in pretrained_config:
        ui_components['drive_models_dir_input'].value = pretrained_config['drive_models_dir']
        _log_debug(logger_bridge, f"Updated drive_models_dir: {pretrained_config['drive_models_dir']}")
    
    # Update pretrained type
    if 'pretrained_type_dropdown' in ui_components and 'pretrained_type' in pretrained_config:
        ui_components['pretrained_type_dropdown'].value = pretrained_config['pretrained_type']
        _log_debug(logger_bridge, f"Updated pretrained_type: {pretrained_config['pretrained_type']}")
    
    # Update sync drive checkbox
    if 'sync_drive_checkbox' in ui_components and 'sync_drive' in pretrained_config:
        ui_components['sync_drive_checkbox'].value = pretrained_config['sync_drive']
        _log_debug(logger_bridge, f"Updated sync_drive: {pretrained_config['sync_drive']}")
    
    _log_info(logger_bridge, "UI berhasil diupdate dengan konfigurasi yang dimuat")


def reset_pretrained_ui(
    ui_components: Dict[str, Any],
    logger_bridge: Optional[LoggerBridge] = None
) -> None:
    """🔄 Reset UI components ke nilai default
    
    Args:
        ui_components: Dictionary berisi komponen UI untuk direset
        logger_bridge: Logger bridge instance untuk logging (opsional)
        
    Raises:
        ValueError: Jika reset UI gagal
    """
    if not ui_components or not isinstance(ui_components, dict):
        error_msg = "UI components tidak valid untuk reset"
        _log_error(logger_bridge, error_msg)
        raise ValueError(error_msg)
    
    try:
        from smartcash.ui.pretrained.handlers.defaults import get_default_pretrained_config
        
        # Get default config
        default_config = get_default_pretrained_config()
        
        # Apply default config to UI
        update_pretrained_ui(ui_components, default_config, logger_bridge)
        
        _log_info(logger_bridge, "UI berhasil direset ke nilai default")
        
    except ImportError as e:
        error_msg = f"Gagal mengimpor modul defaults: {str(e)}"
        _log_error(logger_bridge, error_msg, exc_info=True)
        raise ImportError(error_msg) from e
    except Exception as e:
        error_msg = f"Gagal mereset UI: {str(e)}"
        _log_error(logger_bridge, error_msg, exc_info=True)
        raise ValueError(error_msg) from e


def apply_config_to_ui(
    ui_components: Dict[str, Any], 
    config_key: str, 
    config_value: Any,
    logger_bridge: Optional[LoggerBridge] = None
) -> bool:
    """🎯 Apply single config value ke UI component yang sesuai
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config_key: Key konfigurasi yang akan diterapkan
        config_value: Value konfigurasi yang akan diterapkan
        logger_bridge: Logger bridge instance untuk logging (opsional)
        
    Returns:
        True jika berhasil, False jika gagal
    """
    try:
        # Mapping config key ke UI component
        config_mapping = {
            'models_dir': 'models_dir_input',
            'sync_drive': 'sync_drive_checkbox',
            'model_urls': {
                'yolov5s': 'yolo_url_input',
                'efficientnet': 'efficientnet_url_input'
            }
        }
        
        # Handle model URLs specially
        if config_key == 'model_urls' and isinstance(config_value, dict):
            url_mapping = config_mapping.get('model_urls', {})
            updated = False
            for model_name, url in config_value.items():
                ui_key = url_mapping.get(model_name)
                if ui_key and ui_key in ui_components:
                    ui_components[ui_key].value = url
                    _log_debug(logger_bridge, f"Applied {model_name} URL: {url}")
                    updated = True
            return updated
            
        # Handle regular config values
        ui_key = config_mapping.get(config_key)
        if not ui_key or ui_key not in ui_components:
            _log_warning(logger_bridge, f"UI component tidak ditemukan untuk key: {config_key}")
            return False
        
        # Apply value ke UI component
        ui_components[ui_key].value = config_value
        _log_debug(logger_bridge, f"Applied {config_key}: {config_value}")
        return True
        
    except Exception as e:
        error_msg = f"Error applying config {config_key}: {str(e)}"
        _log_error(logger_bridge, error_msg, exc_info=True)
        return False