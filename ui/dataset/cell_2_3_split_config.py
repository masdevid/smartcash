"""
File: cell_2_3_split_config.py
Deskripsi: Cell untuk konfigurasi pembagian dataset SmartCash dengan visualisasi yang disederhanakan
"""

# Import dasar
import sys
if '.' not in sys.path: sys.path.append('.')

try:
    # Setup environment dan komponen UI
    from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui
    
    # Setup environment dan load config
    env, config = setup_notebook_environment("split_config", "configs/dataset_config.yaml")
    
    # Setup komponen UI
    ui_components = setup_ui_component(env, config, "split_config")
    
    # Setup split handler yang menggunakan komponen modular
    from smartcash.ui.dataset.split_config_handler import setup_split_config_handlers
    ui_components = setup_split_config_handlers(ui_components, env, config)
    
    # Tampilkan UI
    display_ui(ui_components)
    
except ImportError as e: err_alert(e)