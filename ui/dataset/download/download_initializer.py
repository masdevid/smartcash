"""
File: smartcash/ui/dataset/download/download_initializer.py
Deskripsi: Streamlined initializer dengan minimal debug dan error handling yang diperbaiki
"""

from typing import Dict, Any, Optional
from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
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
    """Inisialisasi UI download dataset dengan streamlined setup dan fixed error handling."""
    global _DOWNLOAD_MODULE_INITIALIZED
    
    logger = get_logger(DOWNLOAD_LOGGER_NAMESPACE)
    
    if _DOWNLOAD_MODULE_INITIALIZED:
        logger.debug("UI download dataset sudah diinisialisasi")
    else:
        logger.info("🚀 Inisialisasi UI download dataset")
        _DOWNLOAD_MODULE_INITIALIZED = True
    
    try:
        # Get dan merge config
        config_manager = get_config_manager()
        dataset_config = config_manager.get_config('dataset') if hasattr(config_manager, 'get_config') else {}
        
        if config:
            dataset_config.update(config)
        
        # Create UI components
        ui_components = create_download_ui(dataset_config)
        
        # Setup logger bridge
        logger_bridge = create_ui_logger_bridge(ui_components, DOWNLOAD_LOGGER_NAMESPACE)
        ui_components['logger'] = logger_bridge
        ui_components['logger_namespace'] = DOWNLOAD_LOGGER_NAMESPACE
        ui_components['download_initialized'] = True
        
        # Setup handlers dalam urutan yang tepat
        ui_components = setup_config_handlers(ui_components, dataset_config)
        ui_components = setup_progress_handlers(ui_components)
        ui_components = setup_button_handlers(ui_components, env)
        
        # Validation tanpa verbose logging
        _validate_ui_setup(ui_components)
        
        logger.success("✅ UI download dataset siap digunakan")
        return ui_components['ui']
        
    except Exception as e:
        logger.error(f"❌ Error saat inisialisasi UI: {str(e)}")
        return _create_error_fallback_ui(str(e))

def _validate_ui_setup(ui_components: Dict[str, Any]) -> None:
    """Validate UI setup dengan minimal logging."""
    logger = ui_components.get('logger')
    
    # Check required components
    required_components = [
        'ui', 'download_button', 'check_button', 'reset_button', 
        'cleanup_button', 'save_button', 'progress_container'
    ]
    
    missing_components = [comp for comp in required_components if comp not in ui_components]
    
    if missing_components and logger:
        logger.warning(f"⚠️ Missing components: {', '.join(missing_components)}")
    
    # Check setup flags
    setup_flags = [
        ui_components.get('progress_setup', False),
        ui_components.get('download_initialized', False)
    ]
    
    if not all(setup_flags) and logger:
        logger.warning("⚠️ Tidak semua handler berhasil disetup")

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
    logger = get_logger(DOWNLOAD_LOGGER_NAMESPACE)
    
    try:
        if config is None:
            config_manager = get_config_manager()
            config = config_manager.get_config('dataset') if hasattr(config_manager, 'get_config') else {}
    except Exception as e:
        logger.warning(f"⚠️ Gagal memuat konfigurasi: {str(e)}")
        config = config or {}
    
    # Buat UI components
    ui_components = create_download_ui(config)
    
    # Setup logger bridge
    logger_bridge = create_ui_logger_bridge(ui_components, DOWNLOAD_LOGGER_NAMESPACE)
    ui_components['logger'] = logger_bridge
    ui_components['logger_namespace'] = DOWNLOAD_LOGGER_NAMESPACE
    ui_components['download_initialized'] = True
    
    # Setup basic handlers
    ui_components = setup_config_handlers(ui_components, config)
    ui_components = setup_progress_handlers(ui_components)
    
    return ui_components

def reset_download_module() -> None:
    """Reset module initialization flag - untuk testing."""
    global _DOWNLOAD_MODULE_INITIALIZED
    _DOWNLOAD_MODULE_INITIALIZED = False