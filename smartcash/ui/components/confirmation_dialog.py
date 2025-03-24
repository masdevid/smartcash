"""
File: smartcash/ui/components/confirmation_dialog.py
Deskripsi: Dialog konfirmasi untuk operasi yang mungkin menimpa data yang sudah ada
"""

import ipywidgets as widgets
from typing import Dict, Any, Callable, Optional

def create_confirmation_dialog(
    message: str,
    on_confirm: Callable,
    on_cancel: Optional[Callable] = None,
    title: str = "Konfirmasi",
    confirm_label: str = "Lanjutkan",
    cancel_label: str = "Batal"
) -> widgets.VBox:
    """
    Buat dialog konfirmasi untuk operasi yang mungkin menimpa data.
    
    Args:
        message: Pesan konfirmasi
        on_confirm: Callback saat user mengkonfirmasi
        on_cancel: Callback saat user membatalkan (opsional)
        title: Judul dialog
        confirm_label: Label tombol konfirmasi
        cancel_label: Label tombol batal
        
    Returns:
        Widget dialog konfirmasi
    """
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Dialog content
    content = widgets.VBox([
        widgets.HTML(f"""
        <div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; 
                     color:{COLORS['alert_warning_text']}; 
                     border-left:4px solid {COLORS['alert_warning_text']}; 
                     border-radius:4px; margin:10px 0;">
            <h4 style="margin-top:0; color: inherit;">{ICONS['warning']} {title}</h4>
            <p style="margin-bottom:0;">{message}</p>
        </div>
        """),
        widgets.HBox([
            widgets.Button(
                description=cancel_label,
                button_style="warning",
                icon='times',
                layout=widgets.Layout(margin='5px'),
                tooltip="Batalkan operasi"
            ),
            widgets.Button(
                description=confirm_label,
                button_style="danger",
                icon='check',
                layout=widgets.Layout(margin='5px'),
                tooltip="Konfirmasi operasi"
            )
        ], layout=widgets.Layout(display='flex', justify_content='flex-end'))
    ], layout=widgets.Layout(padding='15px', border='1px solid #ddd', border_radius='4px'))
    
    # Set callbacks
    cancel_button = content.children[1].children[0]
    confirm_button = content.children[1].children[1]
    
    def handle_cancel(b):
        if on_cancel:
            on_cancel()
        content.layout.display = 'none'
    
    def handle_confirm(b):
        on_confirm()
        content.layout.display = 'none'
    
    cancel_button.on_click(handle_cancel)
    confirm_button.on_click(handle_confirm)
    
    return content