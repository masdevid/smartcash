"""
File: smartcash/ui/dataset/download/download_initializer.py
Deskripsi: Initializer yang diperbaiki dengan flow yang tepat dan connection testing
"""

from typing import Dict, Any, Optional
from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge

# Konstanta namespace
from smartcash.ui.utils.ui_logger_namespace import DOWNLOAD_LOGGER_NAMESPACE, KNOWN_NAMESPACES
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[DOWNLOAD_LOGGER_NAMESPACE]

# Import handlers dan components
from smartcash.ui.dataset.download.handlers.button_handlers import setup_button_handlers, debug_button_connections
from smartcash.ui.dataset.download.handlers.config_handlers import setup_config_handlers
from smartcash.ui.dataset.download.handlers.progress_handlers import setup_progress_handlers
from smartcash.ui.dataset.download.components import create_download_ui
from smartcash.ui.dataset.download.utils.download_executor import test_download_connection

# Flag global untuk mencegah inisialisasi ulang
_DOWNLOAD_MODULE_INITIALIZED = False

def initialize_dataset_download_ui(env=None, config=None) -> Any:
    """
    Inisialisasi UI download dataset dengan connection testing dan error handling yang kuat.
    
    Args:
        env: Environment manager (opsional)
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
        # 📋 Get dan merge config
        config_manager = get_config_manager()
        dataset_config = config_manager.get_config('dataset') if hasattr(config_manager, 'get_config') else {}
        
        if config:
            dataset_config.update(config)
        
        # 🎨 Create UI components
        logger.info("🎨 Membuat komponen UI...")
        ui_components = create_download_ui(dataset_config)
        
        # 🔗 Setup logger bridge
        logger_bridge = create_ui_logger_bridge(ui_components, DOWNLOAD_LOGGER_NAMESPACE)
        ui_components['logger'] = logger_bridge
        ui_components['logger_namespace'] = DOWNLOAD_LOGGER_NAMESPACE
        ui_components['download_initialized'] = True
        
        # ⚙️ Setup handlers dalam urutan yang tepat
        logger.info("⚙️ Mengsetup handlers...")
        
        # 1. Config handlers (harus pertama untuk setup environment)
        ui_components = setup_config_handlers(ui_components, dataset_config)
        
        # 2. Progress handlers (setup observer system)
        ui_components = setup_progress_handlers(ui_components)
        
        # 3. Button handlers (terakhir, tergantung pada yang lain)
        ui_components = setup_button_handlers(ui_components, env)
        
        # 🧪 Test connections
        logger.info("🧪 Testing service connections...")
        _test_all_connections(ui_components)
        
        # ✅ Final validation
        _validate_ui_setup(ui_components)
        
        logger.success("✅ UI download dataset berhasil diinisialisasi")
        return ui_components['ui']
        
    except Exception as e:
        logger.error(f"❌ Error saat inisialisasi UI: {str(e)}")
        # Return minimal UI sebagai fallback
        return _create_error_fallback_ui(str(e))

def _test_all_connections(ui_components: Dict[str, Any]) -> None:
    """Test semua koneksi dan dependencies."""
    logger = ui_components.get('logger')
    
    # Test download service connection
    download_ok = test_download_connection(ui_components)
    
    # Test button connections
    if logger:
        logger.debug("🔍 Testing button connections...")
    debug_button_connections(ui_components)
    
    # Test progress system
    progress_system = ui_components.get('_progress_system')
    progress_ok = progress_system is not None and 'handlers' in progress_system
    
    # Test environment manager
    env_manager = ui_components.get('env_manager')
    env_ok = env_manager is not None
    
    # Log test results
    if logger:
        logger.info("🧪 Connection test results:")
        logger.info(f"   • Download service: {'✅' if download_ok else '❌'}")
        logger.info(f"   • Progress system: {'✅' if progress_ok else '❌'}")
        logger.info(f"   • Environment manager: {'✅' if env_ok else '❌'}")
        
        if not all([download_ok, progress_ok, env_ok]):
            logger.warning("⚠️ Beberapa komponen tidak terhubung dengan baik")

def _validate_ui_setup(ui_components: Dict[str, Any]) -> None:
    """Validate UI setup completion."""
    logger = ui_components.get('logger')
    
    # Check required components
    required_components = [
        'ui', 'download_button', 'check_button', 'reset_button', 
        'cleanup_button', 'save_button', 'progress_container'
    ]
    
    missing_components = [comp for comp in required_components if comp not in ui_components]
    
    if missing_components:
        if logger:
            logger.warning(f"⚠️ Missing UI components: {', '.join(missing_components)}")
    
    # Check handlers setup
    setup_flags = [
        ui_components.get('progress_setup', False),
        ui_components.get('download_initialized', False)
    ]
    
    if not all(setup_flags):
        if logger:
            logger.warning("⚠️ Tidak semua handler berhasil disetup")
    
    # Final status
    if logger:
        total_components = len(required_components)
        working_components = total_components - len(missing_components)
        logger.info(f"📊 UI Setup: {working_components}/{total_components} komponen aktif")

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

def reset_download_module() -> None:
    """Reset module initialization flag - untuk testing."""
    global _DOWNLOAD_MODULE_INITIALIZED
    _DOWNLOAD_MODULE_INITIALIZED = False