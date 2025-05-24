"""
File: smartcash/ui/components/confirmation_dialog.py
Deskripsi: Enhanced confirmation dialog dengan full width, light colors, dan better UX
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
    dialog_width: str = "100%",
    danger_mode: bool = False
) -> widgets.VBox:
    """
    Create enhanced full-width confirmation dialog dengan light colors dan modern styling.
    
    Args:
        title: Judul dialog
        message: Pesan konfirmasi (bisa multiline)
        on_confirm: Callback saat confirm button diklik
        on_cancel: Callback saat cancel button diklik
        confirm_text: Text untuk confirm button
        cancel_text: Text untuk cancel button
        dialog_width: Lebar dialog (default: 100%)
        danger_mode: Jika True, gunakan red styling untuk confirm button
        
    Returns:
        VBox widget berisi dialog
    """
    
    # 🎨 Light color palette
    bg_color = "#f8f9fa" if not danger_mode else "#fff5f5"
    border_color = "#e9ecef" if not danger_mode else "#fed7d7"
    title_color = "#495057" if not danger_mode else "#c53030"
    message_color = "#6c757d"
    
    # 📝 Title dengan icon dan styling yang modern
    title_icon = "⚠️" if danger_mode else "ℹ️"
    title_html = widgets.HTML(
        value=f"""
        <div style="
            display: flex; 
            align-items: center; 
            padding: 20px 24px 16px 24px;
            margin: 0;
            border-bottom: 1px solid {border_color};
            background: linear-gradient(135deg, {bg_color} 0%, #ffffff 100%);
        ">
            <span style="font-size: 24px; margin-right: 12px;">{title_icon}</span>
            <h3 style="
                margin: 0; 
                font-size: 18px; 
                font-weight: 600; 
                color: {title_color};
                line-height: 1.4;
            ">{title}</h3>
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='0')
    )
    
    # 📄 Message content dengan better typography
    message_html = widgets.HTML(
        value=f"""
        <div style="
            padding: 24px;
            line-height: 1.6;
            color: {message_color};
            font-size: 14px;
            white-space: pre-wrap;
            background-color: white;
        ">{message}</div>
        """,
        layout=widgets.Layout(width='100%', margin='0')
    )
    
    # 🔘 Modern button styling
    confirm_style = 'danger' if danger_mode else 'primary'
    confirm_icon = 'exclamation-triangle' if danger_mode else 'check'
    
    confirm_button = widgets.Button(
        description=confirm_text,
        button_style=confirm_style,
        icon=confirm_icon,
        layout=widgets.Layout(
            width='auto', 
            min_width='120px',
            height='38px',
            margin='0 8px 0 0'
        )
    )
    
    cancel_button = widgets.Button(
        description=cancel_text,
        button_style='',
        icon='times',
        layout=widgets.Layout(
            width='auto',
            min_width='120px', 
            height='38px'
        )
    )
    
    # 📦 Button container dengan modern spacing
    button_container = widgets.HBox(
        [confirm_button, cancel_button],
        layout=widgets.Layout(
            justify_content='flex-end',
            align_items='center',
            padding='20px 24px',
            background_color=bg_color,
            border_top=f'1px solid {border_color}'
        )
    )
    
    # 🏠 Main dialog container dengan modern card design
    dialog_container = widgets.VBox(
        [title_html, message_html, button_container],
        layout=widgets.Layout(
            width=dialog_width,
            margin='16px 0',
            border=f'1px solid {border_color}',
            border_radius='12px',
            box_shadow='0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            background_color='white',
            overflow='hidden'
        )
    )
    
    # 🔗 Event handlers
    confirm_button.on_click(on_confirm)
    cancel_button.on_click(on_cancel)
    
    return dialog_container

def create_simple_confirmation(
    message: str,
    on_confirm: Callable,
    on_cancel: Optional[Callable] = None,
    confirm_text: str = "Ya",
    cancel_text: str = "Tidak"
) -> widgets.VBox:
    """
    Create simple full-width confirmation dialog untuk quick confirmations.
    
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
        dialog_width="100%"
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
    Create destructive action confirmation dengan modern danger styling.
    
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
        dialog_width="100%",
        danger_mode=True
    )

def create_info_dialog(
    title: str,
    message: str,
    on_close: Callable,
    close_text: str = "Tutup",
    dialog_width: str = "100%"
) -> widgets.VBox:
    """
    Create info-only dialog dengan single close button.
    
    Args:
        title: Judul dialog
        message: Pesan informasi
        on_close: Callback saat close button diklik
        close_text: Text untuk close button
        dialog_width: Lebar dialog
        
    Returns:
        VBox widget berisi info dialog
    """
    
    # 📝 Title dengan info icon
    title_html = widgets.HTML(
        value=f"""
        <div style="
            display: flex; 
            align-items: center; 
            padding: 20px 24px 16px 24px;
            margin: 0;
            border-bottom: 1px solid #e9ecef;
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        ">
            <span style="font-size: 24px; margin-right: 12px;">ℹ️</span>
            <h3 style="
                margin: 0; 
                font-size: 18px; 
                font-weight: 600; 
                color: #495057;
                line-height: 1.4;
            ">{title}</h3>
        </div>
        """,
        layout=widgets.Layout(width='100%', margin='0')
    )
    
    # 📄 Message content
    message_html = widgets.HTML(
        value=f"""
        <div style="
            padding: 24px;
            line-height: 1.6;
            color: #6c757d;
            font-size: 14px;
            white-space: pre-wrap;
            background-color: white;
        ">{message}</div>
        """,
        layout=widgets.Layout(width='100%', margin='0')
    )
    
    # 🔘 Close button
    close_button = widgets.Button(
        description=close_text,
        button_style='primary',
        icon='check',
        layout=widgets.Layout(
            width='auto',
            min_width='120px',
            height='38px'
        )
    )
    
    # 📦 Button container
    button_container = widgets.HBox(
        [close_button],
        layout=widgets.Layout(
            justify_content='flex-end',
            align_items='center',
            padding='20px 24px',
            background_color='#f8f9fa',
            border_top='1px solid #e9ecef'
        )
    )
    
    # 🏠 Main dialog container
    dialog_container = widgets.VBox(
        [title_html, message_html, button_container],
        layout=widgets.Layout(
            width=dialog_width,
            margin='16px 0',
            border='1px solid #e9ecef',
            border_radius='12px',
            box_shadow='0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            background_color='white',
            overflow='hidden'
        )
    )
    
    # 🔗 Event handler
    close_button.on_click(on_close)
    
    return dialog_container