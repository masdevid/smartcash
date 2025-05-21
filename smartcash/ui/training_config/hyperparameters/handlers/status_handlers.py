"""
File: smartcash/ui/training_config/hyperparameters/handlers/status_handlers.py
Deskripsi: Handler untuk status panel di hyperparameters configuration
"""

from typing import Dict, Any, Optional
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
    if 'status_panel' not in ui_components and 'status' not in ui_components:
        ui_components = add_status_panel(ui_components)
    
    status_panel = ui_components.get('status_panel', ui_components.get('status'))
    if status_panel:
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
    if 'status_panel' not in ui_components and 'status' not in ui_components:
        status_panel = widgets.Output(
            layout=widgets.Layout(
                width='100%',
                min_height='50px',
                margin='10px 0'
            )
        )
        ui_components['status_panel'] = status_panel
        # Untuk kompatibilitas dengan kode lama
        ui_components['status'] = status_panel
    
    return ui_components

def show_success_status(ui_components: Dict[str, Any], message: str) -> None:
    """
    Tampilkan pesan sukses di status panel.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
    """
    update_status_panel(ui_components, f"{ICONS.get('success', '✅')} {message}", 'success')

def show_error_status(ui_components: Dict[str, Any], message: str) -> None:
    """
    Tampilkan pesan error di status panel.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
    """
    update_status_panel(ui_components, f"{ICONS.get('error', '❌')} {message}", 'error')

def show_warning_status(ui_components: Dict[str, Any], message: str) -> None:
    """
    Tampilkan pesan warning di status panel.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
    """
    update_status_panel(ui_components, f"{ICONS.get('warning', '⚠️')} {message}", 'warning')

def show_info_status(ui_components: Dict[str, Any], message: str) -> None:
    """
    Tampilkan pesan info di status panel.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
    """
    update_status_panel(ui_components, f"{ICONS.get('info', 'ℹ️')} {message}", 'info')

def clear_status_panel(ui_components: Dict[str, Any]) -> None:
    """
    Bersihkan status panel.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    status_panel = ui_components.get('status_panel', ui_components.get('status'))
    if status_panel:
        with status_panel:
            clear_output()