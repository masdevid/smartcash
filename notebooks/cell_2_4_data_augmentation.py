# Cell 2.4 - Data Augmentation
# Augmentasi dataset untuk meningkatkan variasi dan jumlah data training
import sys
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
    from smartcash.utils.observer import EventDispatcher, EventTopics
    
    # Setup logger dan environment
    logger = get_logger("data_augmentation")
    env_manager = EnvironmentManager()
    
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
    from smartcash.ui_components.augmentation import create_augmentation_ui
    from smartcash.ui_handlers.augmentation import setup_augmentation_handlers
except ImportError as e:
    print(f"❌ Error: {str(e)}")
    print("🔄 Memuat fallback UI...")
    
    # Fallback jika komponen tidak tersedia
    import ipywidgets as widgets
    def create_augmentation_ui():
        return {'ui': widgets.HTML("<h3>⚠️ Komponen UI tidak tersedia</h3><p>Pastikan semua modul terinstall dengan benar</p>")}
    
    def setup_augmentation_handlers(ui_components, config=None):
        return ui_components

# Buat dan setup komponen UI
ui_components = create_augmentation_ui()
ui_components = setup_augmentation_handlers(ui_components, config)

# Tampilkan UI
display(ui_components['ui'])