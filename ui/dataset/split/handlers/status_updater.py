"""
File: smartcash/ui/dataset/split/handlers/status_updater.py
Deskripsi: Update status panel di UI split dataset
"""

from typing import Dict, Any

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ALERT_STYLES, COLORS, ICONS

logger = get_logger(__name__)


def update_status(ui_components: Dict[str, Any], message: str, status_type: str = 'info') -> None:
    """
    Update status panel dengan pesan dan tipe status.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan yang akan ditampilkan
        status_type: Tipe status (info, success, warning, error)
    """
    try:
        if 'status_panel' not in ui_components:
            logger.warning("⚠️ Status panel tidak ditemukan")
            return
        
        # Get style berdasarkan status type
        style_info = ALERT_STYLES.get(status_type, ALERT_STYLES.get('info', {}))
        
        bg_color = style_info.get('bg_color', COLORS.get('alert_info_bg', '#d1ecf1'))
        text_color = style_info.get('text_color', COLORS.get('alert_info_text', '#0c5460'))
        icon = style_info.get('icon', ICONS.get(status_type, ICONS.get('info', 'ℹ️')))
        
        # Update status panel HTML
        ui_components['status_panel'].value = f"""
        <div style="padding:10px; background-color:{bg_color}; 
                   color:{text_color}; border-radius:4px; margin:5px 0;
                   border-left:4px solid {text_color};">
            <p style="margin:5px 0">{icon} {message}</p>
        </div>
        """
        
        # Log ke UI logger jika tersedia
        if 'logger' in ui_components:
            log_method = getattr(ui_components['logger'], status_type, ui_components['logger'].info)
            log_method(message)
        
    except Exception as e:
        logger.error(f"💥 Error updating status: {str(e)}")


def clear_status(ui_components: Dict[str, Any]) -> None:
    """Clear status panel."""
    try:
        update_status(ui_components, "Status siap", 'info')
    except Exception as e:
        logger.error(f"💥 Error clearing status: {str(e)}")


def show_loading_status(ui_components: Dict[str, Any], message: str = "Memproses...") -> None:
    """Tampilkan status loading."""
    update_status(ui_components, f"🔄 {message}", 'info')


def show_success_status(ui_components: Dict[str, Any], message: str) -> None:
    """Tampilkan status sukses."""
    update_status(ui_components, message, 'success')


def show_error_status(ui_components: Dict[str, Any], message: str) -> None:
    """Tampilkan status error."""
    update_status(ui_components, message, 'error')


def show_warning_status(ui_components: Dict[str, Any], message: str) -> None:
    """Tampilkan status warning."""
    update_status(ui_components, message, 'warning')


def create_status_panel() -> Any:
    """
    Buat status panel widget.
    
    Returns:
        HTML widget untuk status panel
    """
    import ipywidgets as widgets
    
    bg_color = COLORS.get('alert_info_bg', '#d1ecf1')
    text_color = COLORS.get('alert_info_text', '#0c5460')
    icon = ICONS.get('info', 'ℹ️')
    
    return widgets.HTML(
        value=f"""
        <div style="padding:10px; background-color:{bg_color}; 
                   color:{text_color}; border-radius:4px; margin:5px 0;
                   border-left:4px solid {text_color};">
            <p style="margin:5px 0">{icon} Status konfigurasi akan ditampilkan di sini</p>
        </div>
        """,
        layout=widgets.Layout(width='100%')
    )