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
            return
            
        # Dapatkan warna berdasarkan status_type
        color_map = {
            'info': 'blue',
            'success': 'green',
            'warning': 'orange',
            'error': 'red'
        }
        color = color_map.get(status_type, 'black')
        
        # Update panel status
        ui_components['status_panel'].value = f'<span style="color: {color};">{message}</span>'
    except Exception as e:
        print(f"❌ Error saat update status panel: {str(e)}")

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
    try:
        # Buat panel status
        status_panel = create_sync_status_panel()
        
        # Tambahkan ke komponen UI
        ui_components['status_panel'] = status_panel
        
        # Tambahkan ke layout jika ada
        if 'ui' in ui_components and hasattr(ui_components['ui'], 'children'):
            # Cari posisi yang tepat untuk menyisipkan status panel
            children = list(ui_components['ui'].children)
            
            # Cari posisi tombol untuk memasukkan status panel setelahnya
            button_pos = -1
            for i, child in enumerate(children):
                if isinstance(child, widgets.Button) or (hasattr(child, 'children') and 
                                                       any(isinstance(c, widgets.Button) for c in child.children)):
                    button_pos = i
                    break
            
            # Sisipkan setelah posisi tombol atau di akhir jika tidak ditemukan
            if button_pos >= 0:
                children.insert(button_pos + 1, status_panel)
            else:
                children.append(status_panel)
            
            # Update children
            ui_components['ui'].children = tuple(children)
        
        return ui_components
    except Exception as e:
        print(f"❌ Error saat menambahkan panel status: {str(e)}")
        return ui_components 