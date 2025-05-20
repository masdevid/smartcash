"""
File: smartcash/ui/training_config/training_strategy/handlers/sync_logger.py
Deskripsi: Logger untuk sinkronisasi konfigurasi strategi pelatihan
"""

from typing import Dict, Any, Optional, Tuple
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

logger = get_logger(__name__)

def update_sync_status(ui_components: Dict[str, Any], message: str, status_type: str = 'info', 
                       clear: bool = True) -> None:
    """
    Update status sinkronisasi di panel status.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
        status_type: Tipe status (info, success, warning, error)
        clear: Apakah akan menghapus output sebelumnya
    """
    try:
        # Dapatkan status panel
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
            if clear:
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
        logger.error(f"{ICONS.get('error', '❌')} Error saat update status sinkronisasi: {str(e)}")

def update_sync_status_only(ui_components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
    """
    Update status sinkronisasi di panel status tanpa menampilkan di info panel.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
        status_type: Tipe status (info, success, warning, error)
    """
    from smartcash.ui.training_config.training_strategy.handlers.status_handlers import update_status_panel
    update_status_panel(ui_components, message, status_type)

def log_sync_status(message: str, status_type: str = 'info') -> None:
    """
    Log status sinkronisasi ke logger.
    
    Args:
        message: Pesan yang akan di-log
        status_type: Tipe status (info, success, warning, error)
    """
    icon = {
        'info': ICONS.get('info', 'ℹ️'),
        'success': ICONS.get('success', '✅'),
        'warning': ICONS.get('warning', '⚠️'),
        'error': ICONS.get('error', '❌')
    }.get(status_type, ICONS.get('info', 'ℹ️'))
    
    log_methods = {
        'info': logger.info,
        'success': logger.info,
        'warning': logger.warning,
        'error': logger.error
    }
    log_method = log_methods.get(status_type, logger.info)
    log_method(f"{icon} {message}") 