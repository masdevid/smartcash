"""
File: smartcash/ui/dataset/split/handlers/sync_logger.py
Deskripsi: Utilitas untuk mencatat status sinkronisasi konfigurasi di UI
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS, COLORS

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
        return
    
    # Dapatkan icon berdasarkan status
    icon = ICONS.get(status, ICONS.get('info', 'ℹ️'))
    
    # Dapatkan warna berdasarkan status
    bg_color = COLORS.get(f'alert_{status}_bg', COLORS.get('alert_info_bg', '#d1ecf1'))
    text_color = COLORS.get(f'alert_{status}_text', COLORS.get('alert_info_text', '#0c5460'))
    
    # Update panel status
    ui_components['status_panel'].value = f"""<div style="padding:10px; background-color:{bg_color}; 
                 color:{text_color}; border-radius:4px; margin:5px 0;
                 border-left:4px solid {text_color}">
            <p style="margin:5px 0">{icon} {message}</p>
        </div>"""

def log_sync_status(ui_components: Dict[str, Any], message: str, status: str = 'info') -> None:
    """
    Log status sinkronisasi ke output UI.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
        status: Status pesan (success, error, info, warning)
    """
    if 'logger' not in ui_components:
        return
    
    # Dapatkan icon berdasarkan status
    icon = ICONS.get(status, ICONS.get('info', 'ℹ️'))
    
    # Format pesan dengan icon
    formatted_message = f"{icon} {message}"
    
    # Update status panel
    update_status_panel(ui_components, message, status)
    
    # Log ke UI logger berdasarkan status
    if status == 'error':
        ui_components['logger'].error(formatted_message)
    elif status == 'warning':
        ui_components['logger'].warning(formatted_message)
    elif status == 'success':
        ui_components['logger'].info(formatted_message)
    else:
        ui_components['logger'].info(formatted_message)
    
    # Log juga ke console logger untuk keperluan debug
    if status == 'error':
        logger.error(formatted_message)
    elif status == 'warning':
        logger.warning(formatted_message)
    elif status == 'success':
        logger.info(formatted_message)
    else:
        logger.info(formatted_message)

def log_sync_success(ui_components: Dict[str, Any], message: str) -> None:
    """
    Log status sinkronisasi sukses ke output UI.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
    """
    log_sync_status(ui_components, message, 'success')

def log_sync_error(ui_components: Dict[str, Any], message: str) -> None:
    """
    Log status sinkronisasi error ke output UI.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
    """
    log_sync_status(ui_components, message, 'error')

def log_sync_warning(ui_components: Dict[str, Any], message: str) -> None:
    """
    Log status sinkronisasi warning ke output UI.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
    """
    log_sync_status(ui_components, message, 'warning')

def log_sync_info(ui_components: Dict[str, Any], message: str) -> None:
    """
    Log status sinkronisasi info ke output UI.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
    """
    log_sync_status(ui_components, message, 'info')

def update_sync_status_only(ui_components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
    """
    Update panel status sinkronisasi di UI tanpa menambahkan log baru.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
        status_type: Tipe status (info, success, warning, error)
    """
    try:
        # Periksa apakah status_panel tersedia
        if 'status_panel' not in ui_components:
            ui_components = add_sync_status_panel(ui_components)
            
        # Dapatkan warna berdasarkan status
        color_map = {
            'info': '#0dcaf0',    # Biru muda
            'success': '#198754', # Hijau
            'warning': '#ffc107', # Kuning
            'error': '#dc3545'    # Merah
        }
        color = color_map.get(status_type, '#6c757d')  # Default: abu-abu
        
        # Dapatkan ikon berdasarkan status
        icon_map = {
            'info': '🔄',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        }
        icon = icon_map.get(status_type, 'ℹ️')
        
        # Update panel status
        ui_components['status_panel'].value = f'<div style="padding: 5px; border-radius: 4px; background-color: #f8f9fa; margin-top: 5px;">Status: <span style="color: {color};">{icon} {message}</span></div>'
    except Exception as e:
        # Fallback ke logger jika ada error
        if 'logger' in ui_components:
            ui_components['logger'].error(f"Error saat update status panel: {str(e)}")

def create_sync_status_panel() -> widgets.HTML:
    """
    Buat panel status untuk sinkronisasi.
    
    Returns:
        widgets.HTML: Panel status
    """
    return widgets.HTML(
        value='<span style="color: blue;">Status sinkronisasi akan ditampilkan di sini</span>',
        layout=widgets.Layout(width='100%', padding='5px')
    )

def add_sync_status_panel(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tambahkan panel status sinkronisasi ke UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    # Buat panel status jika belum ada
    if 'status_panel' not in ui_components:
        status_panel = widgets.HTML(
            value='<div style="padding: 5px; border-radius: 4px; background-color: #f8f9fa; margin-top: 5px;">Status: <span style="color: #6c757d;">Siap</span></div>',
            layout=widgets.Layout(width='100%')
        )
        ui_components['status_panel'] = status_panel
        
        # Tambahkan ke UI jika ada container
        if 'footer_container' in ui_components:
            ui_components['footer_container'].children = (*ui_components['footer_container'].children, status_panel)
        elif 'ui' in ui_components and hasattr(ui_components['ui'], 'children'):
            # Cari container yang tepat untuk menambahkan status panel
            found = False
            for i, child in enumerate(ui_components['ui'].children):
                if hasattr(child, 'description') and child.description == 'Tombol':
                    # Tambahkan status panel setelah container tombol
                    container = widgets.VBox([status_panel])
                    ui_components['ui'].children = (*ui_components['ui'].children[:i+1], container, *ui_components['ui'].children[i+1:])
                    found = True
                    break
            
            # Jika tidak ada container tombol, tambahkan di akhir
            if not found:
                container = widgets.VBox([status_panel])
                ui_components['ui'].children = (*ui_components['ui'].children, container)
    
    return ui_components 