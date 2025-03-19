"""
File: cell_2_2_preprocessing.py
Deskripsi: Cell untuk preprocessing dataset SmartCash dengan kode minimal
"""

from IPython.display import display
import sys
if '.' not in sys.path: sys.path.append('.')

try:
    # Import utilitas cell dan komponen UI
    from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui

    # Setup environment dan load config
    env, config = setup_notebook_environment("preprocessing", "configs/preprocessing_config.yaml")

    # Setup komponen UI
    ui_components = setup_ui_component(env, config, "preprocessing")

    # Setup handler
    from smartcash.ui.dataset.preprocessing_handler import setup_preprocessing_handlers
    ui_components = setup_preprocessing_handlers(ui_components, env, config)

    # Tampilkan UI
    display_ui(ui_components)

except ImportError as e:
    from IPython.display import HTML
    display(HTML(f"<div style='padding:10px; background:#f8d7da; color:#721c24; border-radius:5px'><h3>❌ Error Inisialisasi</h3><p>{str(e)}</p></div>"))