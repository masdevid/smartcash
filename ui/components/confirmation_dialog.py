"""
File: smartcash/ui/components/confirmation_dialog.py
Deskripsi: Fixed dialog components dengan proper cleanup mechanism dan persistent state management
"""

import ipywidgets as widgets
from typing import Callable, Optional, Dict, Any
from IPython.display import clear_output
import uuid

# Global registry untuk track active dialogs
_ACTIVE_DIALOGS = {}

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
    """Create enhanced confirmation dialog dengan proper cleanup mechanism."""
    
    # Generate unique dialog ID
    dialog_id = str(uuid.uuid4())
    
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
    
    # 🔘 Buttons dengan proper cleanup handlers
    button_style = 'danger' if danger_mode else 'primary'
    
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
            padding='25px',
            background_color='white',
            border='1px solid #ddd',
            border_radius='8px',
            box_shadow='0 4px 12px rgba(0,0,0,0.15)',
            margin='10px auto'
        )
    )
    
    # Enhanced event handlers dengan proper cleanup
    def handle_confirm(b):
        """Handle confirm dengan cleanup registry."""
        try:
            # Remove dari registry
            _cleanup_dialog_from_registry(dialog_id)
            
            # Execute callback
            if on_confirm:
                on_confirm(b)
        except Exception as e:
            print(f"⚠️ Confirm callback error: {str(e)}")
    
    def handle_cancel(b):
        """Handle cancel dengan cleanup registry."""
        try:
            # Remove dari registry
            _cleanup_dialog_from_registry(dialog_id)
            
            # Execute callback
            if on_cancel:
                on_cancel(b)
        except Exception as e:
            print(f"⚠️ Cancel callback error: {str(e)}")
    
    # 🔗 Register event handlers
    confirm_button.on_click(handle_confirm)
    cancel_button.on_click(handle_cancel)
    
    # Register dialog di global registry
    _ACTIVE_DIALOGS[dialog_id] = {
        'dialog': dialog_container,
        'confirm_button': confirm_button,
        'cancel_button': cancel_button
    }
    
    # Add cleanup method ke dialog
    dialog_container._dialog_id = dialog_id
    dialog_container._cleanup = lambda: _cleanup_dialog_from_registry(dialog_id)
    
    return dialog_container

def create_destructive_confirmation(
    title: str,
    message: str,
    on_confirm: Callable,
    on_cancel: Callable,
    item_name: str = "item",
    confirm_text: str = None,
    cancel_text: str = "Batal"
) -> widgets.VBox:
    """Create destructive action confirmation dengan enhanced cleanup."""
    
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

def cleanup_all_dialogs():
    """Cleanup semua active dialogs (untuk cell re-execution)."""
    global _ACTIVE_DIALOGS
    
    cleanup_count = len(_ACTIVE_DIALOGS)
    
    for dialog_id in list(_ACTIVE_DIALOGS.keys()):
        _cleanup_dialog_from_registry(dialog_id)
    
    if cleanup_count > 0:
        print(f"🧹 Cleaned up {cleanup_count} active dialogs")

def _cleanup_dialog_from_registry(dialog_id: str):
    """Internal cleanup function untuk remove dialog dari registry."""
    global _ACTIVE_DIALOGS
    
    if dialog_id in _ACTIVE_DIALOGS:
        dialog_info = _ACTIVE_DIALOGS[dialog_id]
        
        try:
            # Disable buttons untuk prevent further clicks
            dialog_info['confirm_button'].disabled = True
            dialog_info['cancel_button'].disabled = True
            
            # Remove event handlers
            dialog_info['confirm_button'].on_click(lambda b: None, remove=True)
            dialog_info['cancel_button'].on_click(lambda b: None, remove=True)
            
        except Exception:
            pass  # Silent cleanup errors
        
        # Remove dari registry
        del _ACTIVE_DIALOGS[dialog_id]

# Auto cleanup saat module import (untuk fresh start)
cleanup_all_dialogs()