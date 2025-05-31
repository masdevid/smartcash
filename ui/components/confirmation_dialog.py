"""
File: smartcash/ui/components/confirmation_dialog.py
Deskripsi: Fixed dialog dengan one-liner style dan IPython session cleanup
"""

import ipywidgets as widgets
from typing import Callable, Optional
from IPython.display import clear_output
import uuid
import atexit

_ACTIVE_DIALOGS = {}
_CLEANUP_REGISTERED = False

def _register_cleanup_handler():
    """Register cleanup handler untuk session end dengan one-liner."""
    global _CLEANUP_REGISTERED
    if not _CLEANUP_REGISTERED:
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            ipython and ipython.events.register('pre_run_cell', lambda: cleanup_all_dialogs()) and setattr(globals(), '_CLEANUP_REGISTERED', True)
        except Exception:
            pass

def create_confirmation_dialog(title: str, message: str, on_confirm: Callable, on_cancel: Callable,
                              confirm_text: str = "Ya, Lanjutkan", cancel_text: str = "Batal",
                              dialog_width: str = "600px", danger_mode: bool = False) -> widgets.VBox:
    """Create confirmation dialog dengan one-liner auto-cleanup session."""
    _register_cleanup_handler()
    dialog_id = str(uuid.uuid4())
    
    title_html = widgets.HTML(f'<div style="font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 15px;">{title}</div>',
                             layout=widgets.Layout(margin='0 0 10px 0'))
    message_html = widgets.HTML(f'<div style="font-size: 14px; line-height: 1.6; color: #34495e; white-space: pre-wrap;">{message}</div>',
                               layout=widgets.Layout(margin='0 0 20px 0'))
    
    button_style = 'danger' if danger_mode else 'primary'
    confirm_button = widgets.Button(description=confirm_text, button_style=button_style, 
                                   icon='check' if not danger_mode else 'warning',
                                   layout=widgets.Layout(width='auto', margin='0 10px 0 0'))
    cancel_button = widgets.Button(description=cancel_text, button_style='', icon='times',
                                  layout=widgets.Layout(width='auto'))
    
    button_container = widgets.HBox([confirm_button, cancel_button], 
                                   layout=widgets.Layout(justify_content='flex-end', margin='15px 0 0 0'))
    dialog_container = widgets.VBox([title_html, message_html, button_container],
                                   layout=widgets.Layout(width=dialog_width, max_width='100%', padding='25px',
                                                        background_color='white', border='1px solid #ddd', border_radius='8px',
                                                        box_shadow='0 4px 12px rgba(0,0,0,0.15)', margin='10px auto', overflow='hidden'))
    
    def handle_confirm(b): (_cleanup_dialog_immediate(dialog_id), on_confirm and on_confirm(b)) if True else None
    def handle_cancel(b): (_cleanup_dialog_immediate(dialog_id), on_cancel and on_cancel(b)) if True else None
    
    confirm_button.on_click(handle_confirm), cancel_button.on_click(handle_cancel)
    
    _ACTIVE_DIALOGS[dialog_id] = {'dialog': dialog_container, 'confirm_button': confirm_button, 
                                  'cancel_button': cancel_button, 'cleanup_handlers': [handle_confirm, handle_cancel]}
    setattr(dialog_container, '_dialog_id', dialog_id), setattr(dialog_container, '_auto_cleanup', lambda: _cleanup_dialog_immediate(dialog_id))
    return dialog_container

def create_destructive_confirmation(title: str, message: str, on_confirm: Callable, on_cancel: Callable,
                                   item_name: str = "item", confirm_text: str = None, cancel_text: str = "Batal") -> widgets.VBox:
    """Create destructive confirmation dengan one-liner auto-cleanup."""
    return create_confirmation_dialog(title, message, on_confirm, on_cancel,
                                    confirm_text or f"Ya, Hapus {item_name}", cancel_text, "500px", True)

def cleanup_all_dialogs():
    """Force cleanup semua active dialogs dengan one-liner batch processing."""
    global _ACTIVE_DIALOGS
    cleanup_count = len(_ACTIVE_DIALOGS)
    [_cleanup_dialog_immediate(dialog_id) for dialog_id in list(_ACTIVE_DIALOGS.keys())]
    _ACTIVE_DIALOGS.clear()
    cleanup_count > 0 and print(f"🧹 Auto-cleaned {cleanup_count} persistent dialogs")

def _cleanup_dialog_immediate(dialog_id: str):
    """Immediate cleanup untuk single dialog dengan one-liner."""
    global _ACTIVE_DIALOGS
    if dialog_id in _ACTIVE_DIALOGS:
        dialog_info = _ACTIVE_DIALOGS[dialog_id]
        try:
            [button and (setattr(button, 'disabled', True), setattr(button, 'description', "Selesai"), button._click_handlers.callbacks.clear()) 
             for button in [dialog_info.get('confirm_button'), dialog_info.get('cancel_button')] if button]
            dialog = dialog_info.get('dialog')
            dialog and (setattr(dialog.layout, 'display', 'none'), setattr(dialog.layout, 'visibility', 'hidden'))
        except Exception:
            pass
        del _ACTIVE_DIALOGS[dialog_id]

def get_active_dialog_count() -> int:
    """Get jumlah active dialogs untuk debugging."""
    return len(_ACTIVE_DIALOGS)

cleanup_all_dialogs()  # Auto cleanup saat module import