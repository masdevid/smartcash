"""
File: smartcash/ui/setup/dependency_installer/utils/ui_utils.py
Deskripsi: Utilitas UI untuk dependency installer dengan pendekatan DRY dan one-liner style
"""

from typing import Dict, Any, Optional, Callable, Union
import ipywidgets as widgets
from IPython.display import display

def show_for_operation(ui_components: Dict[str, Any], operation: str) -> None:
    """Menampilkan komponen UI yang sesuai untuk operasi tertentu
    
    Args:
        ui_components: Dictionary komponen UI
        operation: Jenis operasi ('analyze', 'install', 'validate')
    """
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']): ui_components['reset_progress_bar'](0, "", True)
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'): ui_components['progress_container'].layout.visibility = 'visible'

def reset_ui_logs(ui_components: Dict[str, Any]) -> None:
    """Reset log output dan status panel
    
    Args:
        ui_components: Dictionary komponen UI
    """
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'): ui_components['log_output'].clear_output()
    if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']): ui_components['update_status_panel']("info", "Siap")

def toggle_ui_visibility(ui_components: Dict[str, Any], component_key: str, visible: bool) -> None:
    """Toggle visibilitas komponen UI
    
    Args:
        ui_components: Dictionary komponen UI
        component_key: Kunci komponen yang akan diubah visibilitasnya
        visible: True untuk menampilkan, False untuk menyembunyikan
    """
    if component_key in ui_components and hasattr(ui_components[component_key], 'layout'): ui_components[component_key].layout.visibility = 'visible' if visible else 'hidden'

def error_operation(ui_components: Dict[str, Any], error_message: str) -> None:
    """Menampilkan error pada UI
    
    Args:
        ui_components: Dictionary komponen UI
        error_message: Pesan error yang akan ditampilkan
    """
    if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']): ui_components['update_status_panel']("error", error_message)
    if 'update_progress' in ui_components and callable(ui_components['update_progress']): ui_components['update_progress']('overall', 0, error_message, "#dc3545")
    if 'log_message' in ui_components and callable(ui_components['log_message']): ui_components['log_message'](f"❌ {error_message}", "error")

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info", icon: str = "ℹ️") -> None:
    """Fungsi helper untuk logging ke UI dengan icon dan level yang sesuai
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
        level: Level log (info, success, warning, error)
        icon: Icon untuk pesan
    """
    # Fungsi ini hanya digunakan sebagai implementasi internal, bukan dipanggil langsung dari ui_components['log_message']
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'append_display_data'):
        from IPython.display import display, HTML
        color_map = {"info": "#17a2b8", "success": "#28a745", "warning": "#ffc107", "error": "#dc3545"}
        color = color_map.get(level, color_map["info"])
        html = f"<div style='color: {color};'>{icon} {message}</div>"
        ui_components['log_output'].append_display_data(HTML(html))
    
def update_status_panel(ui_components: Dict[str, Any], status_type: str, message: str) -> None:
    """Update status panel dengan tipe dan pesan tertentu
    
    Args:
        ui_components: Dictionary komponen UI
        status_type: Tipe status (info, success, warning, error)
        message: Pesan yang akan ditampilkan
    """
    if 'status_widget' not in ui_components or not hasattr(ui_components['status_widget'], 'value'): return
    
    colors = {"info": "#17a2b8", "success": "#28a745", "warning": "#ffc107", "error": "#dc3545"}
    bg_color = colors.get(status_type, colors["info"])
    
    if hasattr(ui_components['status_widget'], 'layout'): ui_components['status_widget'].layout.background_color = bg_color
    ui_components['status_widget'].value = message
