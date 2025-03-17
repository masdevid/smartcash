"""
File: cell_2_1_dataset_download.py
Deskripsi: Cell untuk download dataset SmartCash dengan pendekatan modular
"""

# Import komponen UI dari smartcash
from IPython.display import display
import sys

# Pastikan smartcash dalam path
if '.' not in sys.path:
    sys.path.append('.')

# Setup environment dan load config
from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui

# Setup environment dan load config
env, config = setup_notebook_environment(
    cell_name="dataset_download",
    config_path="configs/colab_config.yaml"
)

# Setup komponen UI dan handler
ui_components = setup_ui_component(env, config, "dataset_download")

# Tampilkan UI
display_ui(ui_components)

# Pembersihan sumber daya dilakukan dengan:
# cleanup_resources(ui_components)