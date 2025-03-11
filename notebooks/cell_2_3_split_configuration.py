# Cell 2.3 - Dataset Split Configuration
# Konfigurasi pembagian dataset untuk training, validation, dan testing
import sys
import atexit
from pathlib import Path
from IPython.display import display

# Pastikan smartcash ada di path
if not any('smartcash' in p for p in sys.path):
    sys.path.append('.')

# Setup environment dan konfigurasi
try:
    from smartcash.utils.config_manager import ConfigManager
    from smartcash.utils.environment_manager import EnvironmentManager
    from smartcash.utils.logger import get_logger
    from smartcash.utils.observer.observer_manager import ObserverManager
    
    # Setup logger dan environment
    logger = get_logger("dataset_split")
    env_manager = EnvironmentManager()
    
    # Pastikan semua observer dari sesi sebelumnya dibersihkan
    observer_manager = ObserverManager()
    observer_manager.unregister_group("split_observers")
    
    # Load konfigurasi
    config = ConfigManager.load_config(
        filename="configs/base_config.yaml", 
        fallback_to_pickle=True,
        logger=logger
    )
    
    logger.info("✅ Environment dan konfigurasi berhasil dimuat")
except Exception as e:
    print(f"ℹ️ Fallback ke konfigurasi default: {str(e)}")
    config = {}

# Import komponen UI dan handler
try:
    from smartcash.ui_components.split_config import create_split_config_ui
    from smartcash.ui_handlers.split_config import setup_split_config_handlers
except ImportError as e:
    print(f"❌ Error: {str(e)}")
    print("🔄 Memuat fallback UI...")
    
    # Fallback jika komponen tidak tersedia
    import ipywidgets as widgets
    def create_split_config_ui():
        return {'ui': widgets.HTML("<h3>⚠️ Komponen UI tidak tersedia</h3><p>Pastikan semua modul terinstall dengan benar</p>")}
    
    def setup_split_config_handlers(ui_components, config=None):
        return ui_components

# Buat dan setup komponen UI
ui_components = create_split_config_ui()
ui_components = setup_split_config_handlers(ui_components, config)

# Register cleanup untuk melepas observer
if 'cleanup' in ui_components:
    atexit.register(ui_components['cleanup'])

# Tampilkan UI
display(ui_components['ui'])