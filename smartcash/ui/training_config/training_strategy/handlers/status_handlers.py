"""
File: smartcash/ui/training_config/training_strategy/handlers/status_handlers.py
Deskripsi: Handler untuk status panel di UI konfigurasi strategi pelatihan
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS, COLORS

logger = get_logger(__name__)

def add_status_panel(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tambahkan status panel ke komponen UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary komponen UI yang telah diupdate
    """
    if 'status_panel' not in ui_components:
        logger.info(f"{ICONS.get('info', 'ℹ️')} Menambahkan status panel")
        
        ui_components['status_panel'] = widgets.Output(
            layout=widgets.Layout(
                width='100%',
                min_height='50px',
                margin='10px 0'
            )
        )
        
        # Tambahkan status panel ke main_container jika ada
        if 'main_container' in ui_components:
            if isinstance(ui_components['main_container'], widgets.VBox):
                # Tambahkan status panel ke posisi sebelum footer (jika ada)
                children = list(ui_components['main_container'].children)
                if len(children) > 2:  # Hanya jika ada lebih dari 2 children (header, tabs, dan mungkin footer)
                    # Tambahkan status panel sebelum elemen terakhir (footer)
                    children.insert(len(children) - 1, ui_components['status_panel'])
                    ui_components['main_container'].children = tuple(children)
                else:
                    # Tambahkan status panel ke akhir
                    ui_components['main_container'].children = (*ui_components['main_container'].children, ui_components['status_panel'])
            elif isinstance(ui_components['main_container'], widgets.Box):
                # Tambahkan status panel ke main_container
                ui_components['main_container'].children = (*ui_components['main_container'].children, ui_components['status_panel'])
                
        logger.info(f"{ICONS.get('success', '✅')} Status panel berhasil ditambahkan")
    
    # Untuk kompatibilitas dengan kode lama
    if 'status' not in ui_components:
        ui_components['status'] = ui_components['status_panel']
    
    return ui_components

def update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
    """
    Update status panel dengan pesan.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
        status_type: Tipe status (info, success, warning, error)
    """
    try:
        status_panel = ui_components.get('status_panel')
        if not status_panel:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Status panel tidak ditemukan")
            return
            
        # Tentukan warna berdasarkan tipe status
        bg_color = {
            'info': '#d1ecf1',
            'success': '#d4edda',
            'warning': '#fff3cd',
            'error': '#f8d7da'
        }.get(status_type, '#d1ecf1')
        
        text_color = {
            'info': '#0c5460',
            'success': '#155724',
            'warning': '#856404',
            'error': '#721c24'
        }.get(status_type, '#0c5460')
        
        icon = {
            'info': ICONS.get('info', 'ℹ️'),
            'success': ICONS.get('success', '✅'),
            'warning': ICONS.get('warning', '⚠️'),
            'error': ICONS.get('error', '❌')
        }.get(status_type, ICONS.get('info', 'ℹ️'))
        
        # Buat HTML untuk status
        status_html = f"""
        <div style="padding: 10px; background-color: {bg_color}; color: {text_color}; border-left: 4px solid {text_color}; border-radius: 5px; margin: 4px 0;">
            <div style="display: flex; align-items: flex-start;">
                <div style="margin-right: 10px; font-size: 1.2em;">{icon}</div>
                <div>{message}</div>
            </div>
        </div>
        """
        
        # Update status panel
        with status_panel:
            clear_output(wait=True)
            display(widgets.HTML(status_html))
        
        # Log pesan
        log_methods = {
            'info': logger.info,
            'success': logger.info,
            'warning': logger.warning,
            'error': logger.error
        }
        log_method = log_methods.get(status_type, logger.info)
        log_method(f"{icon} {message}")
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat update status panel: {str(e)}")
        
def clear_status_panel(ui_components: Dict[str, Any]) -> None:
    """
    Bersihkan status panel.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        status_panel = ui_components.get('status_panel')
        if not status_panel:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Status panel tidak ditemukan")
            return
            
        # Bersihkan status panel
        with status_panel:
            clear_output()
            
        logger.info(f"{ICONS.get('success', '✅')} Status panel berhasil dibersihkan")
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat membersihkan status panel: {str(e)}")