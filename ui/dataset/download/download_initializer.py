"""
File: smartcash/ui/dataset/download/download_initializer.py
Deskripsi: Initializer yang disederhanakan untuk UI download dataset dengan integrasi logger terbaru
"""

from typing import Dict, Any, Optional
from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge

# Konstanta namespace
from smartcash.ui.utils.ui_logger_namespace import DOWNLOAD_LOGGER_NAMESPACE, KNOWN_NAMESPACES
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[DOWNLOAD_LOGGER_NAMESPACE]
# Import handlers
from smartcash.ui.dataset.download.handlers.button_handlers import setup_button_handlers
from smartcash.ui.dataset.download.handlers.config_handlers import setup_config_handlers
from smartcash.ui.dataset.download.handlers.progress_handlers import setup_progress_handlers
from smartcash.ui.dataset.download.components import create_download_ui

# Flag global untuk mencegah inisialisasi ulang
_DOWNLOAD_MODULE_INITIALIZED = False

def initialize_dataset_download_ui(env=None, config=None) -> Any:
    """
    Inisialisasi UI untuk dataset downloader dengan logger terintegrasi.
    
    Args:
        env: Environment manager (opsional)`
        config: Konfigurasi aplikasi (opsional)
        
    Returns:
        Widget UI utama
    """
    global _DOWNLOAD_MODULE_INITIALIZED
    
    logger = get_logger(DOWNLOAD_LOGGER_NAMESPACE)
    
    if _DOWNLOAD_MODULE_INITIALIZED:
        logger.debug("UI download dataset sudah diinisialisasi")
    else:
        logger.info("🚀 Inisialisasi UI download dataset")
        _DOWNLOAD_MODULE_INITIALIZED = True
    
    try:
        # Get config dengan fallback
        config_manager = get_config_manager()
        dataset_config = config_manager.get_config('dataset') if hasattr(config_manager, 'get_config') else {}
        
        if config:
            dataset_config.update(config)
            
        # Create UI components
        ui_components = create_download_ui(dataset_config)
        
        # Setup logger bridge untuk UI
        logger_bridge = create_ui_logger_bridge(ui_components, DOWNLOAD_LOGGER_NAMESPACE)
        ui_components['logger'] = logger_bridge
        ui_components['logger_namespace'] = DOWNLOAD_LOGGER_NAMESPACE
        ui_components['download_initialized'] = True
        
        # Setup handlers dalam urutan yang benar
        ui_components = setup_config_handlers(ui_components, dataset_config)
        ui_components = setup_progress_handlers(ui_components)
        ui_components = setup_button_handlers(ui_components, env)
        
        logger.success("✅ UI download dataset berhasil diinisialisasi")
        return ui_components['ui']
        
    except Exception as e:
        logger.error(f"❌ Error saat inisialisasi UI: {str(e)}")
        raise

def get_download_ui_components(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Helper untuk mendapatkan UI components tanpa inisialisasi penuh.
    
    Args:
        config: Konfigurasi UI (opsional)
        
    Returns:
        Dictionary komponen UI
    """
    logger = get_logger(DOWNLOAD_LOGGER_NAMESPACE)
    
    try:
        # Get config dari manager jika tidak diberikan
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