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
    # Import logger untuk keperluan logging
    from smartcash.common.logger import get_logger
    logger = get_logger('dependency_installer')
    
    # Log operasi yang sedang dijalankan
    logger.info(f"Showing UI for operation: {operation}")
    
    # Reset progress bar dengan pesan yang sesuai dengan operasi
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
        # Import konstanta terpusat
        from smartcash.ui.setup.dependency_installer.utils.constants import get_status_config
        
        # Dapatkan emoji dari konfigurasi status
        info_config = get_status_config('info')
        success_config = get_status_config('success')
        
        operation_messages = {
            'analyze': f"{info_config['emoji']} Memulai analisis package...",
            'install': "📦 Memulai instalasi package...",
            'validate': f"{success_config['emoji']} Memulai validasi package..."
        }
        message = operation_messages.get(operation, "Memulai operasi...")
        ui_components['reset_progress_bar'](0, message, True)
    
    # Pastikan progress container terlihat
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        # Simpan status visibilitas sebelumnya untuk debugging
        prev_visibility = ui_components['progress_container'].layout.visibility
        ui_components['progress_container'].layout.visibility = 'visible'
        
        # Log perubahan visibilitas
        if prev_visibility != 'visible':
            logger.info(f"Progress container visibility changed from {prev_visibility} to visible")
    else:
        logger.warning("Progress container not found or doesn't have layout attribute")

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
        ui_components['update_status_panel'](ui_components, "info", "ℹ️ Siap memulai analisis dan instalasi")
    
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
                    # Dapatkan konfigurasi untuk level info
                    info_config = get_status_config('info')
                    ui_components[f"{pkg['key']}_status"].value = f"<span style='color:{info_config['border']};font-size:11px;white-space:nowrap;'>{info_config['emoji']} Checking...</span>"

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
    # Import konstanta terpusat
    from smartcash.ui.setup.dependency_installer.utils.constants import get_status_config
    
    # Dapatkan konfigurasi untuk level error
    error_config = get_status_config('error')
    
    # Pastikan pesan sudah memiliki emoji
    if not error_message.startswith(error_config['emoji']):
        error_message = f"{error_config['emoji']} {error_message}"
    
    # Update status panel dengan pesan error menggunakan fungsi yang telah diperbarui
    update_status_panel(ui_components, "error", error_message)
    
    # Update progress bar untuk menunjukkan error
    if 'update_progress' in ui_components and callable(ui_components['update_progress']):
        # Update overall progress
        ui_components['update_progress']('overall', 0, error_message, error_config['border'])
        
        # Update step dan current progress jika tersedia
        if 'step' in ui_components.get('active_bars', []):
            ui_components['update_progress']('step', 0, error_message, error_config['border'])
        if 'current' in ui_components.get('active_bars', []):
            ui_components['update_progress']('current', 0, error_message, error_config['border'])
    
    # Log pesan error dengan format yang konsisten
    if 'log_message' in ui_components and callable(ui_components['log_message']):
        ui_components['log_message'](error_message, "error")

def log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info", icon: str = None) -> None:
    """Fungsi helper untuk logging ke UI dengan icon dan level yang sesuai
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
        level: Level log (info, success, warning, error)
        icon: Icon untuk pesan (opsional, akan menggunakan default sesuai level jika None)
    """
    # Import logger untuk keperluan logging
    from smartcash.common.logger import get_logger
    from smartcash.ui.setup.dependency_installer.utils.constants import get_status_config
    logger = get_logger('dependency_installer')
    
    # Dapatkan konfigurasi status berdasarkan level
    config = get_status_config(level)
    
    # Tentukan icon default berdasarkan level jika tidak disediakan
    if icon is None:
        icon = config['emoji']
    
    # Dapatkan warna berdasarkan level dari konfigurasi terpusat
    color = config['border']
    
    # Format pesan dengan style yang konsisten
    formatted_message = f"{icon} {message}"
    
    # Log ke console untuk debugging
    logger.debug(f"UI Log ({level}): {formatted_message}")
    
    # Tampilkan ke UI jika log_output tersedia
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'append_display_data'):
        try:
            from IPython.display import display, HTML
            
            # Format HTML dengan style yang konsisten dengan modul UI lainnya
            html = f"""<div style='margin:4px 0;padding:4px 8px;color:{color};font-family:sans-serif;'>
                <span style='font-weight:bold;margin-right:6px;'>{icon}</span>
                <span>{message}</span>
            </div>"""
            
            # Tampilkan ke log output
            ui_components['log_output'].append_display_data(HTML(html))
        except Exception as e:
            logger.error(f"Error displaying log to UI: {str(e)}")
    else:
        logger.warning("Log output component not available or doesn't have append_display_data method")
    
def update_status_panel(ui_components: Dict[str, Any], level: str = "info", message: str = "") -> None:
    """Update status panel dengan pesan dan level yang konsisten
    
    Args:
        ui_components: Dictionary komponen UI
        level: Level status (info, success, warning, error, danger)
        message: Pesan yang akan ditampilkan
    """
    if not ui_components or 'status_panel' not in ui_components or not hasattr(ui_components['status_panel'], 'value'):
        return
    
    # Gunakan konfigurasi dari constants.py
    from smartcash.ui.setup.dependency_installer.utils.constants import get_status_config
    
    # Dapatkan konfigurasi status berdasarkan level
    config = get_status_config(level)
    
    # Config sudah didapatkan dari get_status_config di atas
    
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
    from smartcash.ui.setup.dependency_installer.utils.constants import get_package_status
    
    # Cek apakah status widget tersedia
    status_widget_key = f"{package_key}_status"
    if status_widget_key not in ui_components or not hasattr(ui_components[status_widget_key], 'value'):
        return
    
    # Gunakan konfigurasi dari constants.py
    config = get_package_status(status)
    
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
        # Dapatkan konfigurasi untuk level success
        success_config = get_status_config('success')
        
        ui_components['update_progress']('overall', 100, "Operasi selesai", success_config['border'])
        ui_components['update_progress']('step', 100, "Operasi selesai", success_config['border'])
        
        # Update current progress jika tersedia
        if 'current' in ui_components.get('active_bars', []):
            ui_components['update_progress']('current', 100, f"{success_config['emoji']} Operasi selesai", success_config['border'])
    
    # Update status panel dengan pesan sukses
    # Pastikan pesan sudah memiliki emoji
    if not success_message.startswith("✅"):
        success_message = f"✅ {success_message}"
        
    # Gunakan fungsi update_status_panel dengan parameter yang benar
    update_status_panel(ui_components, "success", success_message)
    
    # Log pesan sukses dengan format yang konsisten
    if 'log_message' in ui_components and callable(ui_components['log_message']) and not ui_components.get('suppress_logs', False):
        ui_components['log_message'](success_message, "success")
