# Cell 2.1 - Dataset Download
# Persiapan dataset untuk training model SmartCash

import sys
import atexit
from pathlib import Path
from IPython.display import display
import logging
# Pastikan smartcash ada di path
if not any('smartcash' in p for p in sys.path):
    sys.path.append('.')

# Load konfigurasi dengan ConfigManager
from smartcash.utils.config_manager import get_config_manager
from smartcash.utils.logging_factory import LoggingFactory

# Load konfigurasi
logger = LoggingFactory.get_logger('dataset_download')
config_manager = get_config_manager(logger=logger)
config = config_manager.load_config('dataset_config.yaml')

try:
    
    from smartcash.utils.observer.cleanup_utils import setup_cell_cleanup, register_notebook_cleanup
    
    
    
except Exception as e:
    print(f"⚠️ Error loading config: {str(e)}")

# Import UI components
from smartcash.ui_components.dataset_download import create_dataset_download_ui
from smartcash.ui_handlers.dataset_download import setup_download_handlers

# Buat dan setup UI
ui_components = create_dataset_download_ui()
ui_components = setup_download_handlers(ui_components, config)

# Setup cleanup otomatis menggunakan utilitas baru
if 'cleanup' in ui_components and callable(ui_components['cleanup']):
    # Daftarkan ke atexit
    cleanup_func = ui_components['cleanup']
    atexit.register(cleanup_func)
    
    # Alternatif, gunakan util yang baru (jika tersedia)
    try:
        setup_cell_cleanup(ui_components)
    except:
        pass

# Tambahkan cleanup untuk dataset_manager jika ada (untuk downloader dan observers)
if 'dataset_manager' in ui_components and hasattr(ui_components['dataset_manager'], 'unregister_observers'):
    manager = ui_components['dataset_manager']
    
    # Daftarkan ke atexit untuk cleanup otomatis saat notebook ditutup
    def cleanup_manager():
        try:
            manager.unregister_observers(group="download_observers")
            
            # Jika ada downloader, cleanup observer di downloader
            if hasattr(manager, 'loading_facade') and hasattr(manager.loading_facade, 'downloader'):
                if hasattr(manager.loading_facade.downloader, 'unregister_observers'):
                    manager.loading_facade.downloader.unregister_observers()
        except Exception as e:
            print(f"⚠️ Error saat cleanup observer: {str(e)}")
    
    atexit.register(cleanup_manager)
    
    # Tambahkan ke ui_components untuk akses manual
    ui_components['cleanup_manager'] = cleanup_manager

# Tampilkan UI
display(ui_components['ui'])