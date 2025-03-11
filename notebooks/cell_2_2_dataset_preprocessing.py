# Cell 2.2 - Dataset Preprocessing
# Preprocessing dataset untuk training model SmartCash
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
    
    # Setup logger dan environment
    logger = get_logger("dataset_preprocessing")
    env_manager = EnvironmentManager()
    
    # Load konfigurasi, cari file di beberapa lokasi standar
    config_files = [
        "configs/base_config.yaml",
        "configs/dataset_config.yaml",
        "config.yaml"
    ]
    
    config = None
    for file_path in config_files:
        try:
            config = ConfigManager.load_config(
                filename=file_path, 
                fallback_to_pickle=True,
                logger=logger
            )
            if config:
                logger.info(f"✅ Konfigurasi berhasil dimuat dari {file_path}")
                break
        except Exception as e:
            pass
    
    # Jika tidak ada file yang ditemukan, buat config default
    if not config:
        logger.warning("⚠️ Tidak ada file konfigurasi ditemukan, menggunakan default")
        config = {
            'data': {
                'preprocessing': {
                    'img_size': [640, 640],
                    'cache_dir': '.cache/smartcash',
                    'num_workers': 4,
                    'normalize_enabled': True,
                    'cache_enabled': True
                }
            },
            'data_dir': 'data'
        }
except Exception as e:
    print(f"⚠️ Fallback ke konfigurasi default: {str(e)}")
    config = {}

# Import komponen UI dan handler
try:
    from smartcash.ui_components.preprocessing import create_preprocessing_ui
    from smartcash.ui_handlers.preprocessing import setup_preprocessing_handlers
except ImportError as e:
    print(f"❌ Error: {str(e)}")
    print("🔄 Memuat fallback UI...")
    
    # Fallback jika komponen tidak tersedia
    import ipywidgets as widgets
    def create_preprocessing_ui():
        return {'ui': widgets.HTML("<h3>⚠️ Komponen UI tidak tersedia</h3><p>Pastikan semua modul terinstall dengan benar</p>")}
    
    def setup_preprocessing_handlers(ui_components, config=None):
        return ui_components

# Buat dan setup komponen UI
ui_components = create_preprocessing_ui()
ui_components = setup_preprocessing_handlers(ui_components, config)

# Tampilkan UI
display(ui_components['ui'])