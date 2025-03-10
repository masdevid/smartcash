# Cell 2.0 - Dataset Preparation
# Persiapan dataset untuk training model SmartCash

from smartcash.ui_components.dataset import create_dataset_preparation_ui
from smartcash.ui_handlers.dataset import setup_dataset_handlers
from IPython.display import display

# Baca konfigurasi jika tersedia
try:
    from smartcash.configs import get_config_manager
    config_manager = get_config_manager("configs/base_config.yaml")
    config = config_manager.get_config()
except:
    config = None  # Gunakan konfigurasi default dalam handler

# Create dan setup UI components
ui_components = create_dataset_preparation_ui()
setup_dataset_handlers(ui_components, config)

# Tampilkan UI
display(ui_components['ui'])