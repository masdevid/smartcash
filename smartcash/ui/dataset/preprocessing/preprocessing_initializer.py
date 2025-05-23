"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Streamlined initializer untuk preprocessing UI dengan integrasi logger dan config yang diperbaiki
"""

from typing import Dict, Any, Optional
from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge

# Konstanta namespace
from smartcash.ui.utils.ui_logger_namespace import PREPROCESSING_LOGGER_NAMESPACE, KNOWN_NAMESPACES
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[PREPROCESSING_LOGGER_NAMESPACE]

# Import handlers dan components
from smartcash.ui.dataset.preprocessing.handlers.main_handler import setup_main_handler
from smartcash.ui.dataset.preprocessing.handlers.config_handler import setup_config_handlers
from smartcash.ui.dataset.preprocessing.handlers.cleanup_handler import setup_cleanup_handler
from smartcash.ui.dataset.preprocessing.handlers.progress_handler import setup_progress_handler
from smartcash.ui.dataset.preprocessing.components.ui_components import create_preprocessing_main_ui

# Flag global untuk mencegah inisialisasi ulang
_PREPROCESSING_MODULE_INITIALIZED = False

def initialize_dataset_preprocessing_ui(env=None, config=None) -> Any:
    """Inisialisasi UI preprocessing dataset dengan streamlined setup dan fixed error handling."""
    global _PREPROCESSING_MODULE_INITIALIZED
    
    logger = get_logger(PREPROCESSING_LOGGER_NAMESPACE)
    
    if _PREPROCESSING_MODULE_INITIALIZED:
        logger.debug("UI preprocessing dataset sudah diinisialisasi")
    else:
        logger.info("🚀 Inisialisasi UI preprocessing dataset")
        _PREPROCESSING_MODULE_INITIALIZED = True
    
    try:
        # Get dan merge config
        config_manager = get_config_manager()
        preprocessing_config = config_manager.get_config('preprocessing') if hasattr(config_manager, 'get_config') else {}
        
        if config:
            preprocessing_config.update(config)
        
        # Create UI components
        ui_components = create_preprocessing_main_ui(preprocessing_config)
        
        # Setup logger bridge
        logger_bridge = create_ui_logger_bridge(ui_components, PREPROCESSING_LOGGER_NAMESPACE)
        ui_components['logger'] = logger_bridge
        ui_components['logger_namespace'] = PREPROCESSING_LOGGER_NAMESPACE
        ui_components['preprocessing_initialized'] = True
        ui_components['config_manager'] = config_manager
        
        # Setup handlers dalam urutan yang tepat
        ui_components = setup_config_handlers(ui_components, preprocessing_config)
        ui_components = setup_progress_handler(ui_components)
        ui_components = setup_cleanup_handler(ui_components)
        ui_components = setup_main_handler(ui_components, env)
        
        # Validation tanpa verbose logging
        _validate_ui_setup(ui_components)
        
        logger.success("✅ UI preprocessing dataset siap digunakan")
        return ui_components['ui']
        
    except Exception as e:
        logger.error(f"❌ Error saat inisialisasi UI: {str(e)}")
        return _create_error_fallback_ui(str(e))

def _validate_ui_setup(ui_components: Dict[str, Any]) -> None:
    """Validate UI setup dengan minimal logging."""
    logger = ui_components.get('logger')
    
    # Check required components
    required_components = [
        'ui', 'preprocess_button', 'cleanup_button', 'save_button', 
        'reset_button', 'progress_container', 'log_output'
    ]
    
    missing_components = [comp for comp in required_components if comp not in ui_components]
    
    if missing_components and logger:
        logger.warning(f"⚠️ Missing components: {', '.join(missing_components)}")
    
    # Check setup flags
    setup_flags = [
        ui_components.get('progress_setup', False),
        ui_components.get('preprocessing_initialized', False)
    ]
    
    if not all(setup_flags) and logger:
        logger.warning("⚠️ Tidak semua handler berhasil disetup")

def _create_error_fallback_ui(error_message: str):
    """Create minimal error fallback UI."""
    import ipywidgets as widgets
    
    error_html = f"""
    <div style="padding: 20px; background: #f8d7da; border: 1px solid #f5c6cb; 
                border-radius: 5px; color: #721c24; margin: 10px 0;">
        <h4>❌ Error Inisialisasi UI Preprocessing</h4>
        <p><strong>Error:</strong> {error_message}</p>
        <p>Silakan restart kernel dan coba lagi.</p>
    </div>
    """
    
    return widgets.HTML(error_html)

def get_preprocessing_ui_components(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Helper untuk mendapatkan UI components tanpa inisialisasi penuh."""
    logger = get_logger(PREPROCESSING_LOGGER_NAMESPACE)
    
    try:
        if config is None:
            config_manager = get_config_manager()
            config = config_manager.get_config('preprocessing') if hasattr(config_manager, 'get_config') else {}
    except Exception as e:
        logger.warning(f"⚠️ Gagal memuat konfigurasi: {str(e)}")
        config = config or {}
    
    # Buat UI components
    ui_components = create_preprocessing_main_ui(config)
    
    # Setup logger bridge
    logger_bridge = create_ui_logger_bridge(ui_components, PREPROCESSING_LOGGER_NAMESPACE)
    ui_components['logger'] = logger_bridge
    ui_components['logger_namespace'] = PREPROCESSING_LOGGER_NAMESPACE
    ui_components['preprocessing_initialized'] = True
    ui_components['config_manager'] = get_config_manager()
    
    # Setup basic handlers
    ui_components = setup_config_handlers(ui_components, config)
    ui_components = setup_progress_handler(ui_components)
    
    return ui_components

def reset_preprocessing_module() -> None:
    """Reset module initialization flag - untuk testing."""
    global _PREPROCESSING_MODULE_INITIALIZED
    _PREPROCESSING_MODULE_INITIALIZED = False

# Alias untuk kompatibilitas
initialize_preprocessing_ui = initialize_dataset_preprocessing_ui