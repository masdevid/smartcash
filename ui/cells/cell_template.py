"""
File: smartcash/ui/cells/cell_template.py
Author: Refactored
Deskripsi: Template cell umum untuk notebook SmartCash dengan mekanisme pemisahan UI dan logika
"""

# Cell template untuk semua notebook SmartCash
# Import hanya fungsi yang diperlukan untuk menghemat token
from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui
from smartcash.ui.utils.logging_utils import setup_ipython_logging
import ipywidgets as widgets
from IPython.display import display, HTML

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
    logger = setup_ipython_logging(ui_components, logger_name=f"cell_{cell_name}")
    if logger:
        ui_components['logger'] = logger
        logger.info(f"Cell {cell_name} diinisialisasi")
    
    # Pastikan ada progress bar untuk overall tracking
    if 'progress_bar' not in ui_components:
        progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='Overall:',
            bar_style='info',
            orientation='horizontal',
            layout=widgets.Layout(
                width='100%',
                margin='10px 0 5px 0'
            )
        )
        ui_components['progress_bar'] = progress_bar
        
        # Tambahkan progress bar ke UI container jika ada
        if 'ui' in ui_components and isinstance(ui_components['ui'], widgets.Box):
            children = list(ui_components['ui'].children)
            children.append(widgets.HTML("<h4 style='margin-bottom:5px;'>Overall Progress Tracking</h4>"))
            children.append(progress_bar)
            ui_components['ui'].children = tuple(children)
        else:
            # Jika tidak ada UI container, buat container baru
            ui_container = widgets.VBox([
                ui_components.get('ui', widgets.HTML(f"<h2>{cell_name.replace('_', ' ').title()}</h2>")),
                widgets.HTML("<h4 style='margin-bottom:5px;'>Overall Progress Tracking</h4>"),
                progress_bar
            ])
            ui_components['ui'] = ui_container
    
    # Tambahkan utility function untuk update progress
    def update_progress(value, max_value=None, description=None):
        """Update progress bar dengan nilai dan deskripsi baru."""
        if 'progress_bar' in ui_components:
            if max_value is not None:
                ui_components['progress_bar'].max = max_value
            ui_components['progress_bar'].value = value
            if description:
                ui_components['progress_bar'].description = description
    
    ui_components['update_progress'] = update_progress
    
    # Reset progress bar
    update_progress(0, 100, 'Overall:')
    
    # Tampilkan UI
    display_ui(ui_components)
    
    # Return komponen UI untuk penggunaan lanjutan jika diperlukan
    return ui_components

# Contoh penggunaan:
# run_cell("env_config")  # Untuk cell environment config
# run_cell("dataset_download")  # Untuk cell download dataset