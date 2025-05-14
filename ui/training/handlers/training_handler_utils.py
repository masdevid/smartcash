"""
File: smartcash/ui/training/handlers/training_handler_utils.py
Deskripsi: Utilitas umum untuk handler UI training
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, HTML
import threading
import time

from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger

# Variabel global untuk status training
training_status = {
    'active': False,
    'thread': None,
    'stop_requested': False
}

def get_training_status():
    """
    Mendapatkan status training saat ini.
    
    Returns:
        Dict berisi status training
    """
    return training_status

def set_training_status(active: bool = False, thread: Optional[threading.Thread] = None, stop_requested: bool = False):
    """
    Mengatur status training.
    
    Args:
        active: Status aktif training
        thread: Thread training
        stop_requested: Flag untuk menghentikan training
    """
    global training_status
    training_status['active'] = active
    
    if thread is not None:
        training_status['thread'] = thread
        
    training_status['stop_requested'] = stop_requested

def update_ui_status(ui_components: Dict[str, Any], status_message: str, is_error: bool = False, progress: Optional[int] = None):
    """
    Memperbarui status UI training.
    
    Args:
        ui_components: Komponen UI
        status_message: Pesan status
        is_error: Apakah status adalah error
        progress: Nilai progress bar (opsional)
    """
    color = "#e74c3c" if is_error else "#2ecc71"
    icon = "❌" if is_error else "✅"
    
    # Update label status
    ui_components['status_label'].value = f'<span style="color:{color}">{icon} {status_message}</span>'
    
    # Update progress bar jika ada
    if progress is not None and 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = progress

def display_status_panel(ui_components: Dict[str, Any], message: str, is_error: bool = False):
    """
    Menampilkan pesan pada panel status.
    
    Args:
        ui_components: Komponen UI
        message: Pesan yang akan ditampilkan
        is_error: Apakah pesan adalah error
    """
    color = "#e74c3c" if is_error else "#2ecc71"
    icon = "❌" if is_error else "✅"
    
    with ui_components['status_panel']:
        ui_components['status_panel'].clear_output()
        display(HTML(f"""
        <div style="color:{color}">
            {icon} {message}
        </div>
        """))

def ensure_ui_persistence(ui_components: Dict[str, Any], module_name: str = 'training'):
    """
    Memastikan persistensi UI components.
    
    Args:
        ui_components: Komponen UI
        module_name: Nama modul untuk persistensi
    """
    try:
        config_manager = get_config_manager()
        config_manager.register_ui_components(module_name, ui_components)
    except Exception as e:
        logger = get_logger("training_ui")
        logger.error(f"❌ Error saat memastikan persistensi UI: {str(e)}")

def update_button_states(ui_components: Dict[str, Any], training_active: bool):
    """
    Memperbarui status tombol berdasarkan status training.
    
    Args:
        ui_components: Komponen UI
        training_active: Status aktif training
    """
    ui_components['start_button'].disabled = training_active
    ui_components['stop_button'].disabled = not training_active
