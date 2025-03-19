"""
File: smartcash/ui/dataset/cell_2_3_split_config.py
Deskripsi: Cell untuk konfigurasi pembagian dataset SmartCash dengan penekanan pada konfigurasi
"""

# Import dasar
from IPython.display import display, HTML
import sys
if '.' not in sys.path: sys.path.append('.')

try:
    # Setup environment dan komponen UI
    from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui
    
    # Setup environment dan load config
    env, config = setup_notebook_environment("split_config", "configs/colab_config.yaml")
    
    # Setup komponen UI
    ui_components = setup_ui_component(env, config, "split_config")
    
    # Setup split handler
    from smartcash.ui.dataset.split_config_handler import setup_split_config_handlers
    ui_components = setup_split_config_handlers(ui_components, env, config)
    
    # Tampilkan UI
    display_ui(ui_components)
    
except ImportError as e:
    from IPython.display import HTML
    display(HTML(f"<div style='padding:10px; background:#f8d7da; color:#721c24; border-radius:5px'><h3>❌ Error Inisialisasi</h3><p>{str(e)}</p></div>"))