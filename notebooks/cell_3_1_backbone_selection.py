# Cell 3.0 - Training Configuration
# Konfigurasi parameter training model SmartCash

# Setup import paths (jika diperlukan)
import sys, os
from pathlib import Path

# Tambahkan current directory ke path jika smartcash tidak ditemukan
if not any('smartcash' in p for p in sys.path):
    module_path = str(Path().absolute())
    if module_path not in sys.path:
        sys.path.append(module_path)

# Fallback import (buat fungsi dummy jika modul tidak tersedia)
try:
    from smartcash.ui_components.training_config import create_training_config_ui
    from smartcash.ui_handlers.training_config import setup_training_config_handlers
except ImportError:
    import ipywidgets as widgets
    from IPython.display import display, HTML
    
    # Pesan fallback jika modul belum tersedia
    def create_fallback_ui():
        ui = widgets.VBox([
            widgets.HTML("<h2>⚠️ SmartCash modules belum tersedia</h2>"),
            widgets.HTML("<p>Anda perlu menyelesaikan setup project terlebih dahulu.</p>"),
            widgets.Button(description="Setup Project", button_style="primary")
        ])
        return {'ui': ui}
    
    create_training_config_ui = create_fallback_ui
    setup_training_config_handlers = lambda ui_components, config=None: ui_components

# Baca konfigurasi jika tersedia
try:
    from smartcash.config import get_config_manager
    config_manager = get_config_manager("configs/base_config.yaml")
    config = config_manager.get_config()
except:
    config = None  # Gunakan konfigurasi default dalam handler

# Create dan setup UI components
ui_components = create_training_config_ui()
setup_training_config_handlers(ui_components, config)

# Tampilkan UI
from IPython.display import display
display(ui_components['ui'])