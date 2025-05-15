"""
File: smartcash/ui/dataset/visualization/visualization_initializer.py
Deskripsi: Inisialisasi UI untuk visualisasi dataset dengan pendekatan DRY
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
from typing import Dict, Any
import threading

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator
from smartcash.common.logger import get_logger
from smartcash.ui.dataset.visualization.components.dashboard_component import create_dashboard_component
from smartcash.ui.dataset.visualization.handlers.dashboard_handler import setup_dashboard_handlers
from smartcash.ui.dataset.visualization.handlers.visualization_tab_handler import setup_visualization_tab_handlers

logger = get_logger(__name__)

def initialize_visualization_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI untuk visualisasi dataset.
    
    Returns:
        Dictionary berisi komponen UI yang telah diinisialisasi
    """
    try:
        # Buat komponen UI dashboard
        ui_components = create_dashboard_component()
        
        # Setup handlers untuk dashboard dan tab visualisasi
        ui_components = setup_dashboard_handlers(ui_components)
        ui_components = setup_visualization_tab_handlers(ui_components)
        
        # Tampilkan UI
        clear_output(wait=True)
        display(ui_components['main_container'])
        
        # Tampilkan pesan sukses
        ui_components['status'].clear_output(wait=True)
        with ui_components['status']:
            display(create_status_indicator("success", f"{ICONS.get('success', '✅')} Dashboard Visualisasi Dataset berhasil diinisialisasi"))
        
        # Jalankan update dashboard cards di thread terpisah
        threading.Thread(target=lambda: ui_components['refresh_button'].click()).start()
        
        return ui_components
    
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat inisialisasi dashboard visualisasi dataset: {str(e)}")
        
        # Buat container minimal untuk menampilkan error
        error_container = widgets.VBox([
            widgets.HTML(f"<h3>{ICONS.get('error', '❌')} Error saat inisialisasi dashboard visualisasi dataset</h3>"),
            widgets.HTML(f"<p>{str(e)}</p>")
        ])
        
        display(error_container)
        return {'error_container': error_container}
