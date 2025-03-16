"""
File: smartcash/ui/cells/cell_template.py
Deskripsi: Template cell umum untuk notebook SmartCash dengan mekanisme pemisahan UI dan logika
"""

# Cell template untuk semua notebook SmartCash
# Import hanya fungsi yang diperlukan untuk menghemat token
from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui

def run_cell(cell_name, config_path="configs/colab_config.yaml"):
    """
    Runner utama untuk cell dengan komponen UI
    
    Args:
        cell_name: Nama komponen UI/cell
        config_path: Path ke file konfigurasi
    """
    # Setup environment dan load config
    env, config = setup_notebook_environment(cell_name, config_path)
    
    # Setup komponen UI dan handler
    ui_components = setup_ui_component(env, config, cell_name)
    
    # Tampilkan UI
    display_ui(ui_components)
    
    # Return komponen UI untuk penggunaan lanjutan jika diperlukan
    return ui_components

# Contoh penggunaan:
# run_cell("env_config")  # Untuk cell environment config
# run_cell("dataset_download")  # Untuk cell download dataset