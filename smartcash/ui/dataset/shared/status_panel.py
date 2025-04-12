"""
File: smartcash/ui/dataset/shared/status_panel.py
Deskripsi: Utilitas bersama untuk mengelola panel status dengan styling yang konsisten
untuk preprocessing dan augmentasi
"""

from typing import Dict, Any, Optional
from smartcash.ui.utils.constants import ALERT_STYLES, ICONS

def update_status_panel(ui_components: Dict[str, Any], status_type: str = "info", message: str = "") -> None:
    """
    Update status panel UI dengan pesan dan tipe status, dengan fallback yang ditingkatkan.
    
    Args:
        ui_components: Dictionary komponen UI
        status_type: Tipe status ('info', 'success', 'warning', 'error')
        message: Pesan yang akan ditampilkan
    """
    try:
        # Gunakan alert_utils standar jika tersedia
        from smartcash.ui.utils.alert_utils import create_info_alert
        
        if 'status_panel' in ui_components:
            ui_components['status_panel'].value = create_info_alert(message, status_type).value
    except ImportError:
        # Fallback dengan styling inline jika tidak bisa import alert_utils
        _update_status_panel_fallback(ui_components, status_type, message)

def _update_status_panel_fallback(ui_components: Dict[str, Any], status_type: str, message: str) -> None:
    """
    Fallback untuk update status panel tanpa ketergantungan pada alert_utils.
    
    Args:
        ui_components: Dictionary komponen UI
        status_type: Tipe status ('info', 'success', 'warning', 'error')
        message: Pesan yang akan ditampilkan
    """
    if 'status_panel' not in ui_components:
        return
        
    # Generate style berdasarkan status_type dari ALERT_STYLES
    style = ALERT_STYLES.get(status_type, ALERT_STYLES['info'])
    bg_color = style.get('bg_color', '#d1ecf1')
    text_color = style.get('text_color', '#0c5460')
    border_color = style.get('border_color', '#0c5460')
    icon = style.get('icon', ICONS.get(status_type, 'ℹ️'))
    
    # Generate HTML dengan styling inline
    html_content = f"""
    <div style="padding: 10px; background-color: {bg_color}; 
                color: {text_color}; margin: 10px 0; border-radius: 4px; 
                border-left: 4px solid {border_color};">
        <p style="margin:5px 0">{icon} {message}</p>
    </div>
    """
    
    # Update widget value
    ui_components['status_panel'].value = html_content

def create_status_panel(initial_message: str = "", status_type: str = "info", 
                      style_override: Optional[Dict[str, str]] = None) -> Any:
    """
    Buat panel status dengan dukungan styling kustom dan konten awal.
    
    Args:
        initial_message: Pesan awal
        status_type: Tipe status pesan awal ('info', 'success', 'warning', 'error')
        style_override: Override styling default
        
    Returns:
        Widget HTML panel status
    """
    try:
        import ipywidgets as widgets
        
        # Buat panel kosong jika tidak ada pesan
        if not initial_message:
            return widgets.HTML(
                layout=widgets.Layout(width='100%', margin='10px 0')
            )
        
        # Gunakan alert_utils jika tersedia
        try:
            from smartcash.ui.utils.alert_utils import create_info_alert
            html_content = create_info_alert(initial_message, status_type).value
        except ImportError:
            # Fallback dengan styling inline
            style = ALERT_STYLES.get(status_type, ALERT_STYLES['info'])
            bg_color = style.get('bg_color', '#d1ecf1')
            text_color = style.get('text_color', '#0c5460')
            border_color = style.get('border_color', '#0c5460')
            icon = style.get('icon', ICONS.get(status_type, 'ℹ️'))
            
            # Apply style override jika ada
            if style_override:
                bg_color = style_override.get('bg_color', bg_color)
                text_color = style_override.get('text_color', text_color)
                border_color = style_override.get('border_color', border_color)
            
            html_content = f"""
            <div style="padding: 10px; background-color: {bg_color}; 
                        color: {text_color}; margin: 10px 0; border-radius: 4px; 
                        border-left: 4px solid {border_color};">
                <p style="margin:5px 0">{icon} {initial_message}</p>
            </div>
            """
        
        # Buat dan kembalikan panel
        return widgets.HTML(
            value=html_content,
            layout=widgets.Layout(width='100%', margin='10px 0')
        )
    except ImportError:
        # Jika ipywidgets tidak tersedia, return None
        return None