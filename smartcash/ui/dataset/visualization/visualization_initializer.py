"""
File: smartcash/ui/dataset/visualization/visualization_initializer.py
Deskripsi: Inisialisasi UI untuk visualisasi dataset dengan pendekatan DRY
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Dict, Any

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.components.status_indicator import create_status_indicator
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.visualization.components.visualization_components import create_visualization_components
from smartcash.ui.dataset.visualization.handlers.visualization_handler import setup_visualization_handlers

logger = get_logger(__name__)

def initialize_visualization_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI untuk visualisasi dataset.
    
    Returns:
        Dictionary berisi komponen UI yang telah diinisialisasi
    """
    try:
        # Buat komponen UI
        ui_components = create_visualization_components()
        
        # Setup handlers
        ui_components = setup_visualization_handlers(ui_components)
        
        # Tampilkan UI
        with ui_components['main_container']:
            clear_output(wait=True)
            display(ui_components['tab'])
        
        # Tampilkan pesan sukses
        with ui_components['status']:
            clear_output(wait=True)
            display(create_status_indicator("success", f"{ICONS.get('success', '✅')} UI Visualisasi Dataset berhasil diinisialisasi"))
        
        return ui_components
    
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat inisialisasi UI visualisasi dataset: {str(e)}")
        
        # Buat container minimal untuk menampilkan error
        error_container = widgets.VBox([
            widgets.HTML(f"<h3>{ICONS.get('error', '❌')} Error saat inisialisasi UI visualisasi dataset</h3>"),
            widgets.HTML(f"<p>{str(e)}</p>")
        ])
        
        display(error_container)
        return {'error_container': error_container}
