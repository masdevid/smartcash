# File: smartcash/ui/pretrained/handlers/config_extractor.py
"""
File: smartcash/ui/pretrained/handlers/config_extractor.py
Deskripsi: Config extractor untuk pretrained models dengan fail-fast approach
"""

from typing import Dict, Any, Optional, TypeVar, Callable
from functools import wraps

from smartcash.ui.utils.ui_logger import UILogger, get_module_logger

# Type variables for generic function typing
T = TypeVar('T')

def with_error_handling(logger: Optional[UILogger] = None):
    """Decorator untuk menangani error dan logging secara konsisten"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"❌ Error in {func.__name__}: {str(e)}"
                if logger:
                    logger.error(error_msg, exc_info=True)
                else:
                    logger = get_module_logger('smartcash.ui.pretrained.handlers.config_extractor')
                    logger.error(error_msg, exc_info=True)
                raise  # Re-raise to allow caller to handle
        return wrapper
    return decorator


def _log_debug(logger: Optional[UILogger], message: str) -> None:
    """Log debug message using UILogger"""
    if logger:
        logger.debug(f"🔧 {message}")
    else:
        logger = get_module_logger('smartcash.ui.pretrained.handlers.config_extractor')
        logger.debug(f"🔧 {message}")


def _log_info(logger: Optional[UILogger], message: str) -> None:
    """Log info message using UILogger"""
    if logger:
        logger.info(f"ℹ️ {message}")
    else:
        logger = get_module_logger('smartcash.ui.pretrained.handlers.config_extractor')
        logger.info(f"ℹ️ {message}")


def _log_warning(logger: Optional[UILogger], message: str) -> None:
    """Log warning message using UILogger"""
    if logger:
        logger.warning(f"⚠️ {message}")
    else:
        logger = get_module_logger('smartcash.ui.pretrained.handlers.config_extractor')
        logger.warning(f"⚠️ {message}")


def _log_error(logger: Optional[UILogger], message: str, exc_info: bool = False) -> None:
    """Log error message using UILogger"""
    if logger:
        logger.error(f"❌ {message}", exc_info=exc_info)
    else:
        logger = get_module_logger('smartcash.ui.pretrained.handlers.config_extractor')
        logger.error(f"❌ {message}", exc_info=exc_info)


def extract_pretrained_config(
    ui_components: Dict[str, Any],
    logger: Optional[UILogger] = None
) -> Dict[str, Any]:
    """🔧 Extract konfigurasi dari UI components pretrained models
    
    Args:
        ui_components: Dictionary berisi komponen UI
        logger_bridge: Logger bridge instance untuk logging (opsional)
        
    Returns:
        Dictionary berisi ekstraksi konfigurasi pretrained models
        
    Raises:
        ValueError: Jika ekstraksi config gagal atau data tidak valid
    """
    if not ui_components or not isinstance(ui_components, dict):
        error_msg = "UI components tidak valid untuk ekstraksi config"
        _log_error(logger, error_msg)
        raise ValueError(error_msg)
    
    try:
        from smartcash.ui.pretrained.handlers.defaults import DEFAULT_MODEL_URLS
        
        config = {'pretrained_models': {}}
        pretrained_config = config['pretrained_models']
        
        # Extract models directory
        if 'models_dir_input' in ui_components:
            models_dir = ui_components['models_dir_input'].value.strip()
            pretrained_config['models_dir'] = models_dir if models_dir else '/data/pretrained'
            _log_debug(logger, f"Extracted models_dir: {pretrained_config['models_dir']}")
        else:
            pretrained_config['models_dir'] = '/data/pretrained'
            _log_debug(logger, f"Using default models_dir: {pretrained_config['models_dir']}")
        
        # Hardcode model type to yolov5s
        pretrained_config['pretrained_type'] = 'yolov5s'
        _log_debug(logger, f"Set pretrained_type to: {pretrained_config['pretrained_type']}")
        
        # Extract model download URLs
        model_urls = {}
        
        # YOLOv5 URL
        if 'yolo_url_input' in ui_components:
            yolo_url = ui_components['yolo_url_input'].value.strip()
            if yolo_url and yolo_url != DEFAULT_MODEL_URLS['yolov5s']:
                model_urls['yolov5s'] = yolo_url
                _log_debug(logger, f"Using custom YOLOv5 URL: {yolo_url}")
        
        # EfficientNet URL
        if 'efficientnet_url_input' in ui_components:
            effnet_url = ui_components['efficientnet_url_input'].value.strip()
            if effnet_url and effnet_url != DEFAULT_MODEL_URLS['efficientnet']:
                model_urls['efficientnet'] = effnet_url
                _log_debug(logger, f"Using custom EfficientNet URL: {effnet_url}")
        
        if model_urls:
            pretrained_config['model_urls'] = model_urls
            _log_debug(logger, f"Using custom model URLs: {model_urls}")
        
        # Disable sync drive by default
        pretrained_config['sync_drive'] = False
        _log_debug(logger, "Sync drive disabled by default")
        
        _log_info(logger, "Config berhasil diekstrak dari UI components")
        return config
        
    except ImportError as e:
        error_msg = f"Gagal mengimpor modul defaults: {str(e)}"
        _log_error(logger_bridge, error_msg, exc_info=True)
        raise ImportError(error_msg) from e
    except Exception as e:
        error_msg = f"Gagal mengekstrak config dari UI: {str(e)}"
        _log_error(logger_bridge, error_msg, exc_info=True)
        raise ValueError(error_msg) from e


def validate_pretrained_config(
    config: Dict[str, Any],
    logger: Optional[UILogger] = None
) -> bool:
    """🔍 Validasi konfigurasi pretrained models
    
    Args:
        config: Dictionary konfigurasi untuk divalidasi
        logger_bridge: Logger bridge instance untuk logging (opsional)
        
    Returns:
        True jika valid, False jika tidak
    """
    try:
        if not isinstance(config, dict):
            _log_warning(logger, "Config bukan dictionary")
            return False
        
        pretrained_models = config.get('pretrained_models', {})
        if not isinstance(pretrained_models, dict):
            _log_warning(logger, "pretrained_models bukan dictionary")
            return False
        
        # Validasi required fields
        required_fields = ['models_dir', 'pretrained_type']
        for field in required_fields:
            if field not in pretrained_models or not pretrained_models[field]:
                _log_warning(logger, f"Required field missing: {field}")
                return False
        
        # Validasi pretrained_type values - Simplified to YOLOv5s only
        valid_types = ['yolov5s']
        if pretrained_models['pretrained_type'] not in valid_types:
            _log_warning(
                logger, 
                f"Invalid pretrained_type: {pretrained_models['pretrained_type']}, using yolov5s"
            )
            return False
        
        _log_info(logger, "Config validation passed")
        return True
        
    except Exception as e:
        error_msg = f"Error validating config: {str(e)}"
        _log_error(logger_bridge, error_msg, exc_info=True)
        return False