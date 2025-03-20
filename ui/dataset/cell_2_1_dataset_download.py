"""
File: cell_2_1_dataset_download.py
Deskripsi: Cell untuk download dataset SmartCash dengan kode minimal
"""

# Import dasar
from IPython.display import display, HTML
import sys
if '.' not in sys.path: sys.path.append('.')

try:
    # Setup environment dan komponen UI
    from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui

    # Setup environment dan load config
    env, config = setup_notebook_environment("dataset_download", "configs/colab_config.yaml")

    # Setup komponen UI
    ui_components = setup_ui_component(env, config, "dataset_download")

    # Setup dataset handler
    from smartcash.ui.dataset.dataset_download_handler import setup_dataset_download_handlers
    ui_components = setup_dataset_download_handlers(ui_components, env, config)

    # Tampilkan UI
    display_ui(ui_components)

except ImportError as e: create_alert(e, 'error')