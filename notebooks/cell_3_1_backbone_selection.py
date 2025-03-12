# Cell 3.1 - Backbone Selection
# Pilih backbone untuk model SmartCash

# Setup import paths
import sys, os, atexit
from pathlib import Path
from IPython.display import display

# Tambahkan current directory ke path jika smartcash tidak ditemukan
if not any('smartcash' in p for p in sys.path):
    module_path = str(Path().absolute())
    if module_path not in sys.path:
        sys.path.append(module_path)

# Setup environment dan konfigurasi
try:
    from smartcash.utils.config_manager import ConfigManager
    from smartcash.utils.environment_manager import EnvironmentManager
    from smartcash.utils.logger import get_logger

    # Setup logger dan environment
    logger = get_logger("backbone_selection")
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
    config = None

# Import komponen UI dan handler
try:
    from smartcash.ui_components.training_config import create_training_config_ui
    from smartcash.ui_handlers.training_config import setup_training_config_handlers
except ImportError as e:
    print(f"❌ Error: {str(e)}")
    print("🔄 Memuat fallback UI...")

    # Fallback jika komponen tidak tersedia
    import ipywidgets as widgets
    def create_training_config_ui():
        return {'ui': widgets.HTML("<h3>⚠️ Komponen UI tidak tersedia</h3><p>Pastikan semua modul terinstall dengan benar</p>")}

    def setup_training_config_handlers(ui_components, config=None):
        return ui_components

# Buat dan setup komponen UI
ui_components = create_training_config_ui()
ui_components = setup_training_config_handlers(ui_components, config)

# Tampilkan UI
display(ui_components['ui'])