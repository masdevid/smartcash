"""
File: smartcash/ui/training_config/hyperparameters/handlers/status_handlers.py
Deskripsi: Handler untuk status panel di hyperparameters configuration
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import clear_output, display
from smartcash.common.logger import get_logger
from smartcash.ui.utils.alert_utils import update_status_panel as utils_update_status_panel
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

def add_status_panel(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tambahkan status panel ke UI components jika belum ada.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    if 'status_panel' not in ui_components:
        ui_components['status_panel'] = widgets.Output()
    
    return ui_components 