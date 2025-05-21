"""
File: smartcash/ui/training_config/backbone/handlers/status_handlers.py
Deskripsi: Handler untuk status panel di backbone configuration
"""

from typing import Dict, Any
import ipywidgets as widgets
from IPython.display import clear_output, display
from smartcash.common.logger import get_logger
from smartcash.ui.utils.alert_utils import create_info_alert
from smartcash.ui.utils.constants import ALERT_STYLES, COLORS, ICONS

logger = get_logger(__name__)

def update_status_panel(ui_components: Dict[str, Any], message: str, status: str = 'info') -> None:
    """
    Update panel status dengan pesan terbaru.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
        status: Status pesan (success, error, info, warning)
    """
    if 'status_panel' not in ui_components:
        ui_components = add_status_panel(ui_components)
    
    status_panel = ui_components['status_panel']
    with status_panel:
        clear_output(wait=True)
        display(create_info_alert(message, alert_type=status))

def create_status_panel() -> widgets.Output:
    """
    Buat panel status untuk konfigurasi backbone.
    
    Returns:
        widgets.Output: Panel status
    """
    return widgets.Output(
        layout=widgets.Layout(width='100%', min_height='50px')
    )

def add_status_panel(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tambahkan panel status ke UI backbone.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    # Buat panel status jika belum ada
    if 'status_panel' not in ui_components:
        status_panel = create_status_panel()
        ui_components['status_panel'] = status_panel
        
        # Tambahkan ke UI jika ada container
        if 'main_container' in ui_components and hasattr(ui_components['main_container'], 'children'):
            # Cari posisi yang tepat untuk menambahkan status panel
            # Biasanya di akhir container
            children = list(ui_components['main_container'].children)
            if status_panel not in children:
                children.append(status_panel)
                ui_components['main_container'].children = tuple(children)
    
    return ui_components