"""
File: smartcash/ui/setup/dependency_installer/utils/ui_utils.py
Deskripsi: Utilitas UI untuk dependency installer dengan pendekatan DRY dan one-liner style
"""

from typing import Dict, Any, Optional, Callable, Union, List
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
    """Reset log output, status panel, dan hasil analisis
    
    Args:
        ui_components: Dictionary komponen UI
    """
    # Reset log output
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'): 
        ui_components['log_output'].clear_output()
    
    # Reset status panel
    if 'update_status_panel' in ui_components and callable(ui_components['update_status_panel']): 
        ui_components['update_status_panel']("info", "ℹ️ Siap memulai analisis dan instalasi")
    
    # Reset hasil analisis untuk memastikan pengecekan ulang
    if 'analysis_result' in ui_components:
        # Simpan kategori dan package untuk referensi
        categories = ui_components.get('categories', [])
        # Hapus hasil analisis
        ui_components['analysis_result'] = None
        
        # Reset status semua package menjadi "Checking..."
        for category in categories:
            for pkg in category.get('packages', []):
                if f"{pkg['key']}_status" in ui_components:
                    ui_components[f"{pkg['key']}_status"].value = f"<span style='color:#17a2b8;font-size:11px;white-space:nowrap;'>🔍 Checking...</span>"

def toggle_ui_visibility(ui_components: Dict[str, Any], component_key: str, visible: bool) -> None:
    """Toggle visibilitas komponen UI
    
    Args:
        ui_components: Dictionary komponen UI
        component_key: Kunci komponen yang akan diubah visibilitasnya
        visible: True untuk menampilkan, False untuk menyembunyikan
    """
    if component_key in ui_components and hasattr(ui_components[component_key], 'layout'): ui_components[component_key].layout.visibility = 'visible' if visible else 'hidden'

def error_operation(ui_components: Dict[str, Any], error_message: str) -> None:
    """Menampilkan error pada UI dengan format yang konsisten
    
    Args:
        ui_components: Dictionary komponen UI
        error_message: Pesan error yang akan ditampilkan
    """
    # Pastikan pesan sudah memiliki emoji
    if not error_message.startswith("❌"):
        error_message = f"❌ {error_message}"
    
    # Update status panel dengan pesan error menggunakan fungsi yang telah diperbarui
    update_status_panel("error", error_message, ui_components)
    
    # Update progress bar untuk menunjukkan error
    if 'update_progress' in ui_components and callable(ui_components['update_progress']):
        ui_components['update_progress']('overall', 0, error_message, "#dc3545")
        
        # Update step dan current progress jika tersedia
        if 'step' in ui_components.get('active_bars', []):
            ui_components['update_progress']('step', 0, error_message, "#dc3545")
        if 'current' in ui_components.get('active_bars', []):
            ui_components['update_progress']('current', 0, error_message, "#dc3545")
    
    # Log pesan error dengan format yang konsisten
    if 'log_message' in ui_components and callable(ui_components['log_message']):
        ui_components['log_message'](error_message, "error")

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
    
def update_status_panel(status_type: str, message: str, ui_components: Dict[str, Any] = None) -> None:
    """Update status panel dengan tipe dan pesan tertentu
    
    Args:
        status_type: Tipe status (info, success, warning, error, danger)
        message: Pesan yang akan ditampilkan
        ui_components: Dictionary komponen UI
    """
    if not ui_components or 'status_panel' not in ui_components or not hasattr(ui_components['status_panel'], 'value'):
        return
    
    # Mapping untuk warna dan emoji berdasarkan status_type
    status_config = {
        "info": {"bg": "#d1ecf1", "color": "#0c5460", "border": "#17a2b8", "emoji": "ℹ️"},
        "success": {"bg": "#d4edda", "color": "#155724", "border": "#28a745", "emoji": "✅"},
        "warning": {"bg": "#fff3cd", "color": "#856404", "border": "#ffc107", "emoji": "⚠️"},
        "error": {"bg": "#f8d7da", "color": "#721c24", "border": "#dc3545", "emoji": "❌"},
        "danger": {"bg": "#f8d7da", "color": "#721c24", "border": "#dc3545", "emoji": "❌"}
    }
    
    # Default ke info jika status_type tidak valid
    config = status_config.get(status_type, status_config["info"])
    
    # Pastikan pesan sudah memiliki emoji, jika belum tambahkan
    if not any(emoji in message for emoji in ["✅", "❌", "⚠️", "ℹ️", "🔍", "📦"]):
        message = f"{config['emoji']} {message}"
    
    # Update status panel HTML
    ui_components['status_panel'].value = f"""
    <div style="padding:8px 12px; background-color:{config['bg']}; 
               color:{config['color']}; border-radius:5px; margin:10px 0;
               border-left:4px solid {config['border']};">
        <p style="margin:3px 0">{message}</p>
    </div>
    """

def update_package_status(ui_components: Dict[str, Any], package_key: str, status: str, message: str = None) -> None:
    """Mengupdate status widget untuk package tertentu
    
    Args:
        ui_components: Dictionary komponen UI
        package_key: Kunci package yang akan diupdate
        status: Status package (success, warning, error, info)
        message: Pesan tambahan (opsional)
    """
    from smartcash.ui.utils.constants import COLORS
    
    # Cek apakah status widget tersedia
    status_widget_key = f"{package_key}_status"
    if status_widget_key not in ui_components or not hasattr(ui_components[status_widget_key], 'value'):
        return
    
    # Konfigurasi status
    status_config = {
        "success": {"icon": "✅", "color": COLORS.get('success', "#28a745"), "text": "Terinstall"},
        "warning": {"icon": "⚠️", "color": COLORS.get('warning', "#ffc107"), "text": "Perlu update"},
        "error": {"icon": "❌", "color": COLORS.get('danger', "#dc3545"), "text": "Tidak terinstall"},
        "info": {"icon": "🔍", "color": COLORS.get('info', "#17a2b8"), "text": "Checking..."}
    }
    
    # Default ke info jika status tidak valid
    config = status_config.get(status, status_config["info"])
    
    # Gunakan pesan kustom jika tersedia
    display_text = message if message else config["text"]
    
    # Update status widget
    ui_components[status_widget_key].value = f"<span style='color:{config['color']};font-size:11px;white-space:nowrap;'>{config['icon']} {display_text}</span>"

def complete_operation(ui_components: Dict[str, Any], success_message: str) -> None:
    """Menandai operasi selesai dengan sukses pada progress tracker dan UI
    
    Args:
        ui_components: Dictionary komponen UI
        success_message: Pesan sukses yang akan ditampilkan
    """
    # Update progress bar ke 100%
    if 'update_progress' in ui_components and callable(ui_components['update_progress']):
        ui_components['update_progress']('overall', 100, "Operasi selesai", "#28a745")
        ui_components['update_progress']('step', 100, "Operasi selesai", "#28a745")
        
        # Update current progress jika tersedia
        if 'current' in ui_components.get('active_bars', []):
            ui_components['update_progress']('current', 100, "✅ Operasi selesai", "#28a745")
    
    # Update status panel dengan pesan sukses
    # Pastikan pesan sudah memiliki emoji
    if not success_message.startswith("✅"):
        success_message = f"✅ {success_message}"
        
    # Gunakan fungsi update_status_panel yang telah diperbarui
    update_status_panel("success", success_message, ui_components)
    
    # Log pesan sukses dengan format yang konsisten
    if 'log_message' in ui_components and callable(ui_components['log_message']) and not ui_components.get('suppress_logs', False):
        ui_components['log_message'](success_message, "success")
