"""
File: smartcash/ui/setup/dependency_installer/handlers/progress_handlers.py
Deskripsi: Handler untuk progress tracking di dependency installer dengan pendekatan DRY dan one-liner style
"""

from typing import Dict, Any, Optional, Callable, Union
import ipywidgets as widgets
from IPython.display import display

# Import fungsi dari ui_utils.py untuk menghindari duplikasi
from smartcash.ui.setup.dependency_installer.utils.ui_utils import (
    show_for_operation, error_operation, update_status_panel, complete_operation
)

def update_progress(ui_components: Dict[str, Any], progress_type: str, value: int, message: str, color: str = None) -> None:
    """Update progress bar dan label
    
    Args:
        ui_components: Dictionary komponen UI
        progress_type: Jenis progress ('overall', 'step')
        value: Nilai progress (0-100)
        message: Pesan yang akan ditampilkan
        color: Warna progress bar (opsional)
    """
    # Import logger_helper untuk logging yang konsisten
    from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message, get_module_logger
    logger = get_module_logger()
    
    # Update progress bar jika tersedia
    if 'progress_bar' in ui_components: 
        ui_components['progress_bar'].value = value
        if color and hasattr(ui_components['progress_bar'], 'style'): ui_components['progress_bar'].style.bar_color = color
        logger.debug(f"Progress bar updated: {value}% - {message}")
    
    # Update label jika tersedia
    if 'progress_label' in ui_components and hasattr(ui_components['progress_label'], 'value'): 
        ui_components['progress_label'].value = message
    
    # Update status widget jika tersedia
    if 'status_widget' in ui_components and hasattr(ui_components['status_widget'], 'value'): 
        ui_components['status_widget'].value = message
    
    # Log pesan ke UI jika step progress
    if progress_type == 'step': 
        level, icon = ("success", "✅") if value == 100 else ("info", "🔄")
        log_message(ui_components, message, level, icon)

def log_message(ui_components: Dict[str, Any], message: str, level: str = "info", icon: str = None) -> None:
    """Fungsi untuk logging ke UI dengan icon dan level yang sesuai
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
        level: Level log (info, success, warning, error)
        icon: Icon untuk pesan (opsional, akan menggunakan default sesuai level jika None)
    """
    # Gunakan fungsi log_message dari logger_helper
    from smartcash.ui.setup.dependency_installer.utils.logger_helper import log_message as helper_log_message
    helper_log_message(ui_components, message, level, icon)

def setup_progress_tracking(ui_components: Dict[str, Any]) -> None:
    """Setup fungsi-fungsi untuk progress tracking
    
    Args:
        ui_components: Dictionary komponen UI
    """
    ui_components['update_progress'] = lambda progress_type, value, message, color=None: update_progress(ui_components, progress_type, value, message, color)
    ui_components['reset_progress_bar'] = lambda value=0, message="", show_progress=True: reset_progress_bar(ui_components, value, message, show_progress)
    ui_components['show_for_operation'] = lambda operation: show_for_operation(ui_components, operation)
    ui_components['error_operation'] = lambda error_message: error_operation(ui_components, error_message)
    ui_components['complete_operation'] = lambda success_message: complete_operation(ui_components, success_message)
    ui_components['update_status_panel'] = lambda status_type, message: update_status_panel(ui_components, status_type, message)
    ui_components['log_message'] = lambda message, level="info", icon="ℹ️": log_message(ui_components, message, level, icon)

def reset_progress_bar(ui_components: Dict[str, Any], value: int = 0, message: str = "", show_progress: bool = True) -> None:
    """Reset progress bar dan label dengan nilai awal dan kontrol visibilitas
    
    Args:
        ui_components: Dictionary komponen UI
        value: Nilai awal progress bar
        message: Pesan yang akan ditampilkan
        show_progress: True untuk menampilkan progress, False untuk menyembunyikan
    """
    # Import logger_helper untuk logging yang konsisten
    from smartcash.ui.setup.dependency_installer.utils.logger_helper import get_module_logger, reset_progress_bar as helper_reset_progress_bar
    logger = get_module_logger()
    
    # Gunakan fungsi reset_progress_bar dari logger_helper jika tersedia
    try:
        # Coba gunakan fungsi dari logger_helper
        helper_reset_progress_bar(ui_components, value, message, show_progress)
        logger.debug(f"Reset progress bar menggunakan helper: {value}% - {message} - show: {show_progress}")
        return
    except Exception as e:
        # Fallback ke implementasi lokal jika helper gagal
        logger.debug(f"Fallback ke implementasi lokal reset_progress_bar: {str(e)}")
    
    # Implementasi lokal sebagai fallback
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = value
        ui_components['progress_bar'].max = 100
        prev_visibility = ui_components['progress_bar'].layout.visibility if hasattr(ui_components['progress_bar'], 'layout') else 'unknown'
        if hasattr(ui_components['progress_bar'], 'layout'): 
            ui_components['progress_bar'].layout.visibility = 'visible' if show_progress else 'hidden'
            logger.debug(f"Progress bar visibility changed from {prev_visibility} to {'visible' if show_progress else 'hidden'}")
        if hasattr(ui_components['progress_bar'], 'reset') and value == 0: 
            ui_components['progress_bar'].reset()
            logger.debug("Progress bar reset to 0")
    
    # Reset tracker jika tersedia
    if 'reset_all' in ui_components and callable(ui_components['reset_all']) and value == 0:
        ui_components['reset_all']()
        logger.debug("Called reset_all function")
        if 'tracker' in ui_components and hasattr(ui_components['tracker'], 'show') and hasattr(ui_components['tracker'], 'hide'):
            if show_progress:
                ui_components['tracker'].show()
                logger.debug("Tracker shown")
            else:
                ui_components['tracker'].hide()
                logger.debug("Tracker hidden")
    
    # Update progress label
    if 'progress_label' in ui_components and hasattr(ui_components['progress_label'], 'value'):
        ui_components['progress_label'].value = message or "Siap"
        if hasattr(ui_components['progress_label'], 'layout'): 
            ui_components['progress_label'].layout.visibility = 'visible'
            logger.debug(f"Progress label updated: {message or 'Siap'}")
    
    # Update status widget
    if 'status_widget' in ui_components and hasattr(ui_components['status_widget'], 'value'):
        ui_components['status_widget'].value = message or "Siap"
        if hasattr(ui_components['status_widget'], 'layout'): 
            ui_components['status_widget'].layout.visibility = 'visible'
            logger.debug(f"Status widget updated: {message or 'Siap'}")
    
    logger.debug(f"Reset progress bar completed: {value}% - {message} - show: {show_progress}")
    
    if value != 0 and 'update_progress' in ui_components and callable(ui_components['update_progress']): ui_components['update_progress']('overall', value, message)
