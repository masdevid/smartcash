"""
File: smartcash/ui/dataset/visualization/handlers/status_handlers.py
Deskripsi: Handler untuk status dan pesan dalam visualisasi dataset
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

logger = get_logger(__name__)

def update_status(ui_components: Dict[str, Any], status_type: str, message: str) -> None:
    """
    Update status panel dengan pesan.
    
    Args:
        ui_components: Dictionary komponen UI
        status_type: Tipe status (info, success, warning, error)
        message: Pesan status
    """
    try:
        status = ui_components.get('status')
        if status:
            with status:
                clear_output(wait=True)
                display(create_status_indicator(status_type, message))
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat update status: {str(e)}")

def show_loading_status(output: widgets.Output, message: str = "Memuat data...") -> None:
    """
    Tampilkan status loading pada output widget.
    
    Args:
        output: Widget output untuk menampilkan status
        message: Pesan loading
    """
    try:
        with output:
            clear_output(wait=True)
            display(create_status_indicator("info", f"{ICONS.get('loading', '⏳')} {message}"))
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan status loading: {str(e)}")

def show_success_status(output: widgets.Output, message: str) -> None:
    """
    Tampilkan status sukses pada output widget.
    
    Args:
        output: Widget output untuk menampilkan status
        message: Pesan sukses
    """
    try:
        with output:
            clear_output(wait=True)
            display(create_status_indicator("success", f"{ICONS.get('success', '✅')} {message}"))
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan status sukses: {str(e)}")

def show_warning_status(output: widgets.Output, message: str) -> None:
    """
    Tampilkan status warning pada output widget.
    
    Args:
        output: Widget output untuk menampilkan status
        message: Pesan warning
    """
    try:
        with output:
            clear_output(wait=True)
            display(create_status_indicator("warning", f"{ICONS.get('warning', '⚠️')} {message}"))
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan status warning: {str(e)}")

def show_error_status(output: widgets.Output, message: str) -> None:
    """
    Tampilkan status error pada output widget.
    
    Args:
        output: Widget output untuk menampilkan status
        message: Pesan error
    """
    try:
        with output:
            clear_output(wait=True)
            display(create_status_indicator("error", f"{ICONS.get('error', '❌')} {message}"))
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan status error: {str(e)}")

def show_dummy_data_warning(output: widgets.Output) -> None:
    """
    Tampilkan peringatan penggunaan data dummy pada output widget.
    
    Args:
        output: Widget output untuk menampilkan peringatan
    """
    try:
        display(create_status_indicator("warning", f"{ICONS.get('warning', '⚠️')} Menggunakan data dummy untuk visualisasi karena data aktual tidak tersedia"))
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat menampilkan peringatan data dummy: {str(e)}") 