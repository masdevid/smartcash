"""
File: smartcash/ui/components/confirmation_dialog.py
Deskripsi: Fixed confirmation dialog dengan safe callback handling dan proper cleanup
"""

import ipywidgets as widgets
from typing import Callable, Optional
from IPython.display import clear_output
import uuid
import atexit
from typing import Dict, Any, Optional, List
from smartcash.common.utils.one_liner_fixes import safe_operation_or_none, safe_widget_operation

_ACTIVE_DIALOGS = {}
_CLEANUP_REGISTERED = False

def _register_cleanup_handler():
    """Register cleanup handler untuk session end dengan safe operations"""
    global _CLEANUP_REGISTERED
    if not _CLEANUP_REGISTERED:
        def register_operation():
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython and hasattr(ipython, 'events'):
                ipython.events.register('pre_run_cell', lambda: cleanup_all_dialogs())
                return True
            return False
        
        if safe_operation_or_none(register_operation):
            _CLEANUP_REGISTERED = True

def create_confirmation_dialog(title: str, message: str, on_confirm: Callable, on_cancel: Callable,
                              confirm_text: str = "Ya, Lanjutkan", cancel_text: str = "Batal",
                              dialog_width: str = "600px", danger_mode: bool = False) -> widgets.VBox:
    """Create confirmation dialog dengan safe callback handling"""
    _register_cleanup_handler()
    dialog_id = str(uuid.uuid4())
    
    # Create title widget
    title_html = widgets.HTML(
        f'<div style="font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 15px;">{title}</div>',
        layout=widgets.Layout(margin='0 0 10px 0')
    )
    
    # Create message widget
    message_html = widgets.HTML(
        f'<div style="font-size: 14px; line-height: 1.6; color: #34495e; white-space: pre-wrap;">{message}</div>',
        layout=widgets.Layout(margin='0 0 20px 0')
    )
    
    # Create buttons
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
    
    # Create button container
    button_container = widgets.HBox(
        [confirm_button, cancel_button],
        layout=widgets.Layout(justify_content='flex-end', margin='15px 0 0 0')
    )
    
    # Create dialog container
    dialog_container = widgets.VBox(
        [title_html, message_html, button_container],
        layout=widgets.Layout(
            width=dialog_width, max_width='100%', padding='25px',
            background_color='white', border='1px solid #ddd', border_radius='8px',
            box_shadow='0 4px 12px rgba(0,0,0,0.15)', margin='10px auto', overflow='hidden'
        )
    )
    
    # Safe button handlers
    def handle_confirm(button):
        """Safe confirm handler"""
        def confirm_operation():
            _cleanup_dialog_immediate(dialog_id)
            if on_confirm and callable(on_confirm):
                on_confirm(button)
        
        safe_operation_or_none(confirm_operation)
    
    def handle_cancel(button):
        """Safe cancel handler"""
        def cancel_operation():
            _cleanup_dialog_immediate(dialog_id)
            if on_cancel and callable(on_cancel):
                on_cancel(button)
        
        safe_operation_or_none(cancel_operation)
    
    # Bind handlers safely
    safe_widget_operation(confirm_button, 'on_click', handle_confirm)
    safe_widget_operation(cancel_button, 'on_click', handle_cancel)
    
    # Register dialog safely
    _ACTIVE_DIALOGS[dialog_id] = {
        'dialog': dialog_container,
        'confirm_button': confirm_button,
        'cancel_button': cancel_button,
        'cleanup_handlers': [handle_confirm, handle_cancel]
    }
    
    # Add cleanup attributes
    setattr(dialog_container, '_dialog_id', dialog_id)
    setattr(dialog_container, '_auto_cleanup', lambda: _cleanup_dialog_immediate(dialog_id))
    
    return dialog_container

def create_destructive_confirmation(title: str, message: str, on_confirm: Callable, on_cancel: Callable,
                                   item_name: str = "item", confirm_text: str = None, cancel_text: str = "Batal") -> widgets.VBox:
    """Create destructive confirmation dengan safe operations"""
    confirm_text = confirm_text or f"Ya, Hapus {item_name}"
    
    return create_confirmation_dialog(
        title, message, on_confirm, on_cancel,
        confirm_text, cancel_text, "500px", True
    )

def cleanup_all_dialogs():
    """Force cleanup semua active dialogs dengan safe operations"""
    global _ACTIVE_DIALOGS
    
    def cleanup_operation():
        cleanup_count = len(_ACTIVE_DIALOGS)
        dialog_ids = list(_ACTIVE_DIALOGS.keys())
        
        for dialog_id in dialog_ids:
            _cleanup_dialog_immediate(dialog_id)
        
        _ACTIVE_DIALOGS.clear()
        
        if cleanup_count > 0:
            print(f"🧹 Auto-cleaned {cleanup_count} persistent dialogs")
        
        return cleanup_count
    
    safe_operation_or_none(cleanup_operation)

def _cleanup_dialog_immediate(dialog_id: str):
    """Immediate cleanup untuk single dialog dengan safe operations"""
    global _ACTIVE_DIALOGS
    
    def cleanup_operation():
        if dialog_id not in _ACTIVE_DIALOGS:
            return False
        
        dialog_info = _ACTIVE_DIALOGS[dialog_id]
        
        # Disable buttons safely
        for button_key in ['confirm_button', 'cancel_button']:
            button = dialog_info.get(button_key)
            if button:
                safe_widget_operation(button, 'disabled', True)
                safe_widget_operation(button, 'description', "Selesai")
                
                # Clear click handlers safely
                if hasattr(button, '_click_handlers') and hasattr(button._click_handlers, 'callbacks'):
                    try:
                        button._click_handlers.callbacks.clear()
                    except Exception:
                        pass
        
        # Hide dialog safely
        dialog = dialog_info.get('dialog')
        if dialog and hasattr(dialog, 'layout'):
            safe_widget_operation(dialog.layout, 'display', 'none')
            safe_widget_operation(dialog.layout, 'visibility', 'hidden')
        
        # Remove from active dialogs
        del _ACTIVE_DIALOGS[dialog_id]
        return True
    
    safe_operation_or_none(cleanup_operation)

def get_active_dialog_count() -> int:
    """Get jumlah active dialogs untuk debugging"""
    def count_operation():
        return len(_ACTIVE_DIALOGS)
    
    return safe_operation_or_none(count_operation) or 0

def force_cleanup_dialog_by_id(dialog_id: str) -> bool:
    """Force cleanup specific dialog by ID"""
    def cleanup_operation():
        if dialog_id in _ACTIVE_DIALOGS:
            _cleanup_dialog_immediate(dialog_id)
            return True
        return False
    
    return bool(safe_operation_or_none(cleanup_operation))

def get_dialog_status() -> Dict[str, Any]:
    """Get status of dialog system"""
    def status_operation():
        return {
            'active_dialogs': len(_ACTIVE_DIALOGS),
            'cleanup_registered': _CLEANUP_REGISTERED,
            'dialog_ids': list(_ACTIVE_DIALOGS.keys())
        }
    
    return safe_operation_or_none(status_operation) or {
        'active_dialogs': 0, 'cleanup_registered': False, 'dialog_ids': []
    }

def create_safe_confirmation_with_validation(title: str, message: str, 
                                           validation_func: Callable = None,
                                           on_confirm: Callable = None, on_cancel: Callable = None,
                                           **kwargs) -> widgets.VBox:
    """Create confirmation dialog dengan validation function"""
    
    def safe_confirm_handler(button):
        """Safe confirm handler dengan validation"""
        def validate_and_confirm():
            # Run validation if provided
            if validation_func and callable(validation_func):
                try:
                    validation_result = validation_func()
                    if not validation_result:
                        return False
                except Exception:
                    return False
            
            # Execute original confirm handler
            if on_confirm and callable(on_confirm):
                on_confirm(button)
            
            return True
        
        safe_operation_or_none(validate_and_confirm)
    
    def safe_cancel_handler(button):
        """Safe cancel handler"""
        if on_cancel and callable(on_cancel):
            safe_operation_or_none(lambda: on_cancel(button))
    
    return create_confirmation_dialog(
        title, message, safe_confirm_handler, safe_cancel_handler, **kwargs
    )

def create_info_dialog(title: str, message: str, on_close: Callable = None,
                      close_text: str = "OK", dialog_width: str = "500px") -> widgets.VBox:
    """Create info dialog dengan single close button"""
    dialog_id = str(uuid.uuid4())
    
    # Create title widget
    title_html = widgets.HTML(
        f'<div style="font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 15px;">ℹ️ {title}</div>',
        layout=widgets.Layout(margin='0 0 10px 0')
    )
    
    # Create message widget
    message_html = widgets.HTML(
        f'<div style="font-size: 14px; line-height: 1.6; color: #34495e; white-space: pre-wrap;">{message}</div>',
        layout=widgets.Layout(margin='0 0 20px 0')
    )
    
    # Create close button
    close_button = widgets.Button(
        description=close_text,
        button_style='primary',
        icon='check',
        layout=widgets.Layout(width='auto')
    )
    
    # Create button container
    button_container = widgets.HBox(
        [close_button],
        layout=widgets.Layout(justify_content='center', margin='15px 0 0 0')
    )
    
    # Create dialog container
    dialog_container = widgets.VBox(
        [title_html, message_html, button_container],
        layout=widgets.Layout(
            width=dialog_width, max_width='100%', padding='25px',
            background_color='white', border='1px solid #007bff', border_radius='8px',
            box_shadow='0 4px 12px rgba(0,123,255,0.15)', margin='10px auto', overflow='hidden'
        )
    )
    
    # Safe close handler
    def handle_close(button):
        """Safe close handler"""
        def close_operation():
            _cleanup_dialog_immediate(dialog_id)
            if on_close and callable(on_close):
                on_close(button)
        
        safe_operation_or_none(close_operation)
    
    # Bind handler safely
    safe_widget_operation(close_button, 'on_click', handle_close)
    
    # Register dialog
    _ACTIVE_DIALOGS[dialog_id] = {
        'dialog': dialog_container,
        'close_button': close_button,
        'cleanup_handlers': [handle_close]
    }
    
    return dialog_container

def create_progress_dialog(title: str, message: str, progress_callback: Callable = None,
                          cancel_callback: Callable = None, dialog_width: str = "600px") -> Dict[str, Any]:
    """Create progress dialog dengan cancel support"""
    dialog_id = str(uuid.uuid4())
    
    # Create title widget
    title_html = widgets.HTML(
        f'<div style="font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 15px;">⏳ {title}</div>',
        layout=widgets.Layout(margin='0 0 10px 0')
    )
    
    # Create message widget
    message_html = widgets.HTML(
        f'<div style="font-size: 14px; line-height: 1.6; color: #34495e;">{message}</div>',
        layout=widgets.Layout(margin='0 0 15px 0')
    )
    
    # Create progress bar
    progress_bar = widgets.IntProgress(
        value=0, min=0, max=100,
        bar_style='info',
        layout=widgets.Layout(width='100%', margin='10px 0')
    )
    
    # Create status label
    status_label = widgets.HTML(
        '<div style="text-align: center; color: #666; font-size: 12px;">Starting...</div>',
        layout=widgets.Layout(margin='5px 0 15px 0')
    )
    
    # Create cancel button (optional)
    components = [title_html, message_html, progress_bar, status_label]
    
    if cancel_callback and callable(cancel_callback):
        cancel_button = widgets.Button(
            description="Cancel",
            button_style='',
            icon='times',
            layout=widgets.Layout(width='auto')
        )
        
        button_container = widgets.HBox(
            [cancel_button],
            layout=widgets.Layout(justify_content='center', margin='10px 0 0 0')
        )
        components.append(button_container)
        
        # Safe cancel handler
        def handle_cancel(button):
            """Safe cancel handler"""
            def cancel_operation():
                _cleanup_dialog_immediate(dialog_id)
                cancel_callback(button)
            
            safe_operation_or_none(cancel_operation)
        
        safe_widget_operation(cancel_button, 'on_click', handle_cancel)
    else:
        cancel_button = None
    
    # Create dialog container
    dialog_container = widgets.VBox(
        components,
        layout=widgets.Layout(
            width=dialog_width, max_width='100%', padding='25px',
            background_color='white', border='1px solid #17a2b8', border_radius='8px',
            box_shadow='0 4px 12px rgba(23,162,184,0.15)', margin='10px auto', overflow='hidden'
        )
    )
    
    # Progress update function
    def update_progress(value: int, status: str = None):
        """Update progress safely"""
        def update_operation():
            if 0 <= value <= 100:
                progress_bar.value = value
            
            if status:
                status_label.value = f'<div style="text-align: center; color: #666; font-size: 12px;">{status}</div>'
        
        safe_operation_or_none(update_operation)
    
    # Complete function
    def complete_progress(success: bool = True, final_message: str = None):
        """Complete progress safely"""
        def complete_operation():
            progress_bar.value = 100
            progress_bar.bar_style = 'success' if success else 'danger'
            
            final_msg = final_message or ('Completed successfully!' if success else 'Operation failed!')
            status_label.value = f'<div style="text-align: center; color: {"#28a745" if success else "#dc3545"}; font-size: 12px; font-weight: bold;">{final_msg}</div>'
            
            # Auto cleanup after delay
            import threading
            def delayed_cleanup():
                import time
                time.sleep(3)
                _cleanup_dialog_immediate(dialog_id)
            
            threading.Thread(target=delayed_cleanup, daemon=True).start()
        
        safe_operation_or_none(complete_operation)
    
    # Register dialog
    dialog_info = {
        'dialog': dialog_container,
        'progress_bar': progress_bar,
        'status_label': status_label,
        'update_progress': update_progress,
        'complete_progress': complete_progress
    }
    
    if cancel_button:
        dialog_info['cancel_button'] = cancel_button
    
    _ACTIVE_DIALOGS[dialog_id] = dialog_info
    
    return {
        'dialog': dialog_container,
        'update_progress': update_progress,
        'complete_progress': complete_progress,
        'dialog_id': dialog_id
    }

# Auto cleanup saat module import
safe_operation_or_none(cleanup_all_dialogs)