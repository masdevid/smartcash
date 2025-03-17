"""
File: smartcash/ui/components/cell_template.py
Deskripsi: Template cell sederhana untuk notebook SmartCash dengan pemisahan UI dan logika
"""

# Cell template untuk semua notebook SmartCash
from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui

def run_cell(cell_name, config_path="configs/colab_config.yaml"):
    """
    Runner utama untuk cell dengan komponen UI
    
    Args:
        cell_name: Nama komponen UI/cell
        config_path: Path ke file konfigurasi
        
    Returns:
        Dictionary berisi komponen UI dan handler yang telah disetup
    """
    # Setup environment dan load config
    env, config = setup_notebook_environment(cell_name, config_path)
    
    # Setup komponen UI dan handler
    ui_components = setup_ui_component(env, config, cell_name)
    
    # Setup logger yang terintegrasi dengan UI
    ui_components['module_name'] = cell_name
    try:
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        logger = setup_ipython_logging(ui_components, logger_name=f"cell_{cell_name}")
        if logger:
            ui_components['logger'] = logger
            logger.info(f"🚀 Cell {cell_name} diinisialisasi")
    except ImportError:
        pass
    
    # Tampilkan UI
    display_ui(ui_components)
    
    # Return komponen UI untuk penggunaan lanjutan jika diperlukan
    return ui_components

# Contoh penggunaan:
# run_cell("env_config")  # Untuk cell environment config
# run_cell("dataset_download")  # Untuk cell download dataset