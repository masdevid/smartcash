"""
File: smartcash/ui/components/confirmation_dialog.py
Deskripsi: Fixed dialog dengan IPython session cleanup dan auto-clear saat cell restart
"""

import ipywidgets as widgets
from typing import Callable, Optional, Dict, Any
from IPython.display import clear_output
import uuid
import atexit

# Global registry dengan auto-cleanup
_ACTIVE_DIALOGS = {}
_CLEANUP_REGISTERED = False

def _register_cleanup_handler():
    """Register cleanup handler untuk session end."""
    global _CLEANUP_REGISTERED
    if not _CLEANUP_REGISTERED:
        try:
            # Register untuk IPython session cleanup
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython:
                # Clear saat cell baru dieksekusi
                ipython.events.register('pre_run_cell', lambda: cleanup_all_dialogs())
                _CLEANUP_REGISTERED = True
        except Exception:
            pass

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
    """Create confirmation dialog dengan auto-cleanup session."""
    
    # Register cleanup handler
    _register_cleanup_handler()
    
    # Generate unique dialog ID
    dialog_id = str(uuid.uuid4())
    
    # Components dengan responsive styling
    title_html = widgets.HTML(
        value=f'<div style="font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 15px;">{title}</div>',
        layout=widgets.Layout(margin='0 0 10px 0')
    )
    
    message_html = widgets.HTML(
        value=f'<div style="font-size: 14px; line-height: 1.6; color: #34495e; white-space: pre-wrap;">{message}</div>',
        layout=widgets.Layout(margin='0 0 20px 0')
    )
    
    # Buttons dengan enhanced cleanup
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
    
    button_container = widgets.HBox(
        [confirm_button, cancel_button],
        layout=widgets.Layout(justify_content='flex-end', margin='15px 0 0 0')
    )
    
    dialog_container = widgets.VBox(
        [title_html, message_html, button_container],
        layout=widgets.Layout(
            width=dialog_width,
            max_width='100%',
            padding='25px',
            background_color='white',
            border='1px solid #ddd',
            border_radius='8px',
            box_shadow='0 4px 12px rgba(0,0,0,0.15)',
            margin='10px auto',
            overflow='hidden'
        )
    )
    
    # Enhanced event handlers dengan immediate cleanup
    def handle_confirm(b):
        """Handle confirm dengan immediate cleanup."""
        try:
            _cleanup_dialog_immediate(dialog_id)
            if on_confirm:
                on_confirm(b)
        except Exception as e:
            print(f"⚠️ Confirm callback error: {str(e)}")
    
    def handle_cancel(b):
        """Handle cancel dengan immediate cleanup."""
        try:
            _cleanup_dialog_immediate(dialog_id)
            if on_cancel:
                on_cancel(b)
        except Exception as e:
            print(f"⚠️ Cancel callback error: {str(e)}")
    
    # Register handlers
    confirm_button.on_click(handle_confirm)
    cancel_button.on_click(handle_cancel)
    
    # Store dalam registry dengan metadata
    _ACTIVE_DIALOGS[dialog_id] = {
        'dialog': dialog_container,
        'confirm_button': confirm_button,
        'cancel_button': cancel_button,
        'cleanup_handlers': [handle_confirm, handle_cancel]
    }
    
    # Add metadata ke dialog
    dialog_container._dialog_id = dialog_id
    dialog_container._auto_cleanup = lambda: _cleanup_dialog_immediate(dialog_id)
    
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
    """Create destructive confirmation dengan auto-cleanup."""
    
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
    """Force cleanup semua active dialogs - dipanggil otomatis saat cell restart."""
    global _ACTIVE_DIALOGS
    
    cleanup_count = len(_ACTIVE_DIALOGS)
    
    # Batch cleanup
    dialog_ids = list(_ACTIVE_DIALOGS.keys())
    for dialog_id in dialog_ids:
        _cleanup_dialog_immediate(dialog_id)
    
    # Force clear registry
    _ACTIVE_DIALOGS.clear()
    
    if cleanup_count > 0:
        print(f"🧹 Auto-cleaned {cleanup_count} persistent dialogs")

def _cleanup_dialog_immediate(dialog_id: str):
    """Immediate cleanup untuk single dialog."""
    global _ACTIVE_DIALOGS
    
    if dialog_id in _ACTIVE_DIALOGS:
        dialog_info = _ACTIVE_DIALOGS[dialog_id]
        
        try:
            # Disable buttons immediately
            for button_key in ['confirm_button', 'cancel_button']:
                button = dialog_info.get(button_key)
                if button:
                    button.disabled = True
                    button.description = "Selesai"
                    # Clear click handlers
                    button._click_handlers.callbacks.clear()
            
            # Hide dialog container
            dialog = dialog_info.get('dialog')
            if dialog:
                dialog.layout.display = 'none'
                dialog.layout.visibility = 'hidden'
                
        except Exception:
            pass  # Silent cleanup errors
        
        # Remove dari registry
        del _ACTIVE_DIALOGS[dialog_id]

def get_active_dialog_count() -> int:
    """Get jumlah active dialogs untuk debugging."""
    return len(_ACTIVE_DIALOGS)

# Auto cleanup saat module import
cleanup_all_dialogs()