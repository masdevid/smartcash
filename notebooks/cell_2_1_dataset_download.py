# Cell 2.1 - Dataset Download
# Persiapan dataset untuk training model SmartCash

import sys
from pathlib import Path
from IPython.display import display

# Pastikan smartcash ada di path
if not any('smartcash' in p for p in sys.path):
    sys.path.append('.')

# Load konfigurasi dengan ConfigManager
try:
    from smartcash.utils.config_manager import ConfigManager
    
    # Pastikan direktori configs ada
    Path("configs").mkdir(parents=True, exist_ok=True)
    
    # Load konfigurasi
    config = ConfigManager.load_config(
        filename="configs/base_config.yaml", 
        fallback_to_pickle=True,
        default_config={
            'data': {
                'source': 'roboflow',
                'roboflow': {'workspace': 'smartcash-wo2us', 'project': 'rupiah-emisi-2022', 'version': '3'},
            },
            'data_dir': 'data'
        }
    )
    
except Exception as e:
    print(f"⚠️ Error loading config: {str(e)}")
    config = {
        'data': {
            'source': 'roboflow',
            'roboflow': {'workspace': 'smartcash-wo2us', 'project': 'rupiah-emisi-2022', 'version': '3'},
        },
        'data_dir': 'data'
    }

# Import UI components
from smartcash.ui_components.dataset_download import create_dataset_download_ui
from smartcash.ui_handlers.dataset_download import setup_download_handlers

# Buat dan setup UI
ui_components = create_dataset_download_ui()
ui_components = setup_download_handlers(ui_components, config)

# Tampilkan UI
display(ui_components['ui'])

# Register cleanup untuk melepas observer
if 'cleanup' in ui_components:
    import atexit
    atexit.register(ui_components['cleanup'])