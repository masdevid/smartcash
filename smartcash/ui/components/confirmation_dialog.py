"""
File: smartcash/ui/components/confirmation_dialog.py
Deskripsi: Enhanced confirmation dialog component untuk dataset operations
"""

import ipywidgets as widgets
from typing import Callable, Optional

def create_confirmation_dialog(
    title: str,
    message: str,
    on_confirm: Callable,
    on_cancel: Callable,
    confirm_text: str = "Ya, Lanjutkan",
    cancel_text: str = "Batal",
    dialog_width: str = "600px",
    danger_mode: bool = False
) -> widgets.VBox:
    """
    Create enhanced confirmation dialog dengan styling yang lebih baik.
    
    Args:
        title: Judul dialog
        message: Pesan konfirmasi (bisa multiline)
        on_confirm: Callback saat confirm button diklik
        on_cancel: Callback saat cancel button diklik
        confirm_text: Text untuk confirm button
        cancel_text: Text untuk cancel button
        dialog_width: Lebar dialog
        danger_mode: Jika True, gunakan red styling untuk confirm button
        
    Returns:
        VBox widget berisi dialog
    """
    
    # 🎨 Styling
    title_style = "font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 15px;"
    message_style = "font-size: 14px; line-height: 1.6; color: #34495e; white-space: pre-wrap;"
    dialog_style = """
        padding: 25px;
        background-color: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        max-width: 100%;
        margin: 10px auto;
    """
    
    # 📝 Header
    title_html = widgets.HTML(
        value=f'<div style="{title_style}">{title}</div>',
        layout=widgets.Layout(margin='0 0 10px 0')
    )
    
    # 📄 Message content
    message_html = widgets.HTML(
        value=f'<div style="{message_style}">{message}</div>',
        layout=widgets.Layout(margin='0 0 20px 0')
    )
    
    # 🔘 Buttons
    button_style = danger_mode and 'danger' or 'primary'
    
    confirm_button = widgets.Button(
        description=confirm_text,
        button_style=button_style,
        icon='check' if not danger_mode else 'warning',
        layout=widgets.Layout(width='auto', margin='0 10px 0 0')
    )
    
    cancel_button = widgets.Button(
        description=cancel_text,
        button_style='',
        icon='times',
        layout=widgets.Layout(width='auto')
    )
    
    # 📦 Button container
    button_container = widgets.HBox(
        [confirm_button, cancel_button],
        layout=widgets.Layout(
            justify_content='flex-end',
            margin='15px 0 0 0'
        )
    )
    
    # 🏠 Main container
    dialog_container = widgets.VBox(
        [title_html, message_html, button_container],
        layout=widgets.Layout(
            width=dialog_width,
            padding='0px',
        )
    )
    
    # 🎁 Wrapper dengan styling
    styled_wrapper = widgets.HTML(
        value=f'<div style="{dialog_style}"></div>',
        layout=widgets.Layout(width=dialog_width)
    )
    
    # 📋 Final dialog
    final_dialog = widgets.VBox(
        [dialog_container],
        layout=widgets.Layout(
            width=dialog_width,
            padding='25px',
            background_color='white',
            border='1px solid #ddd',
            border_radius='8px',
            box_shadow='0 4px 12px rgba(0,0,0,0.15)',
            margin='10px auto'
        )
    )
    
    # 🔗 Event handlers
    confirm_button.on_click(on_confirm)
    cancel_button.on_click(on_cancel)
    
    return final_dialog

def create_simple_confirmation(
    message: str,
    on_confirm: Callable,
    on_cancel: Optional[Callable] = None,
    confirm_text: str = "Ya",
    cancel_text: str = "Tidak"
) -> widgets.VBox:
    """
    Create simple confirmation dialog untuk quick confirmations.
    
    Args:
        message: Pesan konfirmasi
        on_confirm: Callback saat confirm
        on_cancel: Callback saat cancel (optional)
        confirm_text: Text confirm button
        cancel_text: Text cancel button
        
    Returns:
        VBox widget berisi simple dialog
    """
    
    def default_cancel(b):
        pass
    
    return create_confirmation_dialog(
        title="Konfirmasi",
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel or default_cancel,
        confirm_text=confirm_text,
        cancel_text=cancel_text,
        dialog_width="400px"
    )

def create_destructive_confirmation(
    title: str,
    message: str,
    on_confirm: Callable,
    on_cancel: Callable,
    item_name: str = "item",
    confirm_text: str = None,
    cancel_text: str = "Batal"
) -> widgets.VBox:
    """
    Create destructive action confirmation dengan red styling.
    
    Args:
        title: Judul dialog
        message: Pesan warning
        on_confirm: Callback saat confirm destructive action
        on_cancel: Callback saat cancel
        item_name: Nama item yang akan dihapus
        confirm_text: Custom confirm text (default: "Ya, Hapus {item_name}")
        cancel_text: Cancel button text
        
    Returns:
        VBox widget berisi destructive confirmation dialog
    """
    
    if confirm_text is None:
        confirm_text = f"Ya, Hapus {item_name}"
    
    return create_confirmation_dialog(
        title=title,
        message=message,
        on_confirm=on_confirm,
        on_cancel=on_cancel,
        confirm_text=confirm_text,
        cancel_text=cancel_text,
        dialog_width="500px",
        danger_mode=True
    )