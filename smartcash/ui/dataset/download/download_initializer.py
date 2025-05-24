"""
File: smartcash/ui/dataset/download/download_initializer.py
Deskripsi: Silent initializer yang mencegah logs muncul sebelum UI siap
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

def initialize_dataset_download_ui(env=None, config=None) -> Any:
    """Inisialisasi UI download dataset dengan silent setup."""
    global _DOWNLOAD_MODULE_INITIALIZED
    
    # Silent mode - tidak create logger sampai UI ready
    if _DOWNLOAD_MODULE_INITIALIZED:
        return _get_existing_ui_or_recreate(config)
    
    try:
        # Get dan merge config (silent)
        config_manager = get_config_manager()
        dataset_config = {}
        try:
            dataset_config = config_manager.get_config('dataset') if hasattr(config_manager, 'get_config') else {}
        except Exception:
            pass
        
        if config:
            dataset_config.update(config)
        
        # Create UI components terlebih dahulu
        ui_components = create_download_ui(dataset_config)
        
        # Baru setup logger bridge setelah UI ready
        logger_bridge = create_ui_logger_bridge(ui_components, DOWNLOAD_LOGGER_NAMESPACE)
        ui_components['logger'] = logger_bridge
        ui_components['logger_namespace'] = DOWNLOAD_LOGGER_NAMESPACE
        ui_components['download_initialized'] = True
        
        # Setup handlers dalam urutan yang tepat (silent)
        ui_components = _silent_setup_handlers(ui_components, dataset_config, env)
        
        # Validation tanpa verbose logging
        _silent_validate_ui_setup(ui_components)
        
        # Mark sebagai initialized
        _DOWNLOAD_MODULE_INITIALIZED = True
        
        # Log sukses hanya setelah UI fully ready
        logger = ui_components.get('logger')
        if logger:
            logger.success("✅ UI download dataset siap digunakan")
        
        return ui_components['ui']
        
    except Exception as e:
        return _create_error_fallback_ui(str(e))

def _get_existing_ui_or_recreate(config=None):
    """Get existing UI atau recreate jika diperlukan."""
    try:
        # Try to recreate dengan config baru
        config_manager = get_config_manager()
        dataset_config = {}
        try:
            dataset_config = config_manager.get_config('dataset') if hasattr(config_manager, 'get_config') else {}
        except Exception:
            pass
        
        if config:
            dataset_config.update(config)
        
        ui_components = create_download_ui(dataset_config)
        logger_bridge = create_ui_logger_bridge(ui_components, DOWNLOAD_LOGGER_NAMESPACE)
        ui_components['logger'] = logger_bridge
        ui_components['logger_namespace'] = DOWNLOAD_LOGGER_NAMESPACE
        ui_components['download_initialized'] = True
        
        ui_components = _silent_setup_handlers(ui_components, dataset_config, None)
        
        return ui_components['ui']
        
    except Exception as e:
        return _create_error_fallback_ui(str(e))

def _silent_setup_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env) -> Dict[str, Any]:
    """Setup handlers dalam mode silent."""
    try:
        # Setup config handlers (silent)
        ui_components = setup_config_handlers(ui_components, config)
        
        # Setup progress handlers (silent)
        ui_components = setup_progress_handlers(ui_components)
        
        # Setup button handlers (silent)
        ui_components = setup_button_handlers(ui_components, env)
        
        return ui_components
        
    except Exception:
        # Silent fail - return ui_components as is
        return ui_components

def _silent_validate_ui_setup(ui_components: Dict[str, Any]) -> None:
    """Validate UI setup tanpa logging yang mengganggu."""
    required_components = [
        'ui', 'download_button', 'check_button', 'reset_button', 
        'cleanup_button', 'save_button', 'progress_container'
    ]
    
    missing_components = [comp for comp in required_components if comp not in ui_components]
    
    # Hanya log jika ada missing components yang critical
    if missing_components and len(missing_components) > 2:
        logger = ui_components.get('logger')
        if logger:
            logger.warning(f"⚠️ Some components missing: {', '.join(missing_components[:3])}")

def _create_error_fallback_ui(error_message: str):
    """Create minimal error fallback UI."""
    import ipywidgets as widgets
    
    error_html = f"""
    <div style="padding: 20px; background: #f8d7da; border: 1px solid #f5c6cb; 
                border-radius: 5px; color: #721c24; margin: 10px 0;">
        <h4>❌ Error Inisialisasi UI Download</h4>
        <p><strong>Error:</strong> {error_message}</p>
        <p>Silakan restart kernel dan coba lagi.</p>
    </div>
    """
    
    return widgets.HTML(error_html)

def get_download_ui_components(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Helper untuk mendapatkan UI components tanpa inisialisasi penuh."""
    try:
        if config is None:
            config_manager = get_config_manager()
            config = {}
            try:
                config = config_manager.get_config('dataset') if hasattr(config_manager, 'get_config') else {}
            except Exception:
                pass
    except Exception:
        config = config or {}
    
    # Buat UI components
    ui_components = create_download_ui(config)
    
    # Setup logger bridge
    logger_bridge = create_ui_logger_bridge(ui_components, DOWNLOAD_LOGGER_NAMESPACE)
    ui_components['logger'] = logger_bridge
    ui_components['logger_namespace'] = DOWNLOAD_LOGGER_NAMESPACE
    ui_components['download_initialized'] = True
    
    # Setup basic handlers (silent)
    ui_components = _silent_setup_handlers(ui_components, config, None)
    
    return ui_components

def reset_download_module() -> None:
    """Reset module initialization flag - untuk testing."""
    global _DOWNLOAD_MODULE_INITIALIZED
    _DOWNLOAD_MODULE_INITIALIZED = False

def suppress_initial_logs():
    """Suppress initial logs dari common modules."""
    import logging
    import sys
    
    # Suppress backend loggers yang muncul saat restart
    loggers_to_suppress = [
        'smartcash.common.environment',
        'smartcash.common.config.manager', 
        'smartcash.common.logger',
        'smartcash.ui.utils.logger_bridge',
        'requests',
        'urllib3'
    ]
    
    for logger_name in loggers_to_suppress:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
        # Clear handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # Redirect stdout temporarily saat initialization
    original_stdout = sys.stdout
    
    class InitSuppressor:
        def write(self, text):
            # Suppress common initialization messages
            suppress_patterns = [
                '✅ Colab detected',
                '📁 Drive',
                'Environment Summary',
                'Initializing',
                'Setup complete'
            ]
            
            if not any(pattern in text for pattern in suppress_patterns):
                original_stdout.write(text)
        
        def flush(self):
            original_stdout.flush()
    
    return original_stdout, InitSuppressor()

# Auto-suppress pada import
_original_stdout, _suppressor = suppress_initial_logs()