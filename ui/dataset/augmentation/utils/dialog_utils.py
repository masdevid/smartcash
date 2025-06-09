"""
File: smartcash/ui/dataset/augmentation/utils/dialog_utils.py
Deskripsi: Dialog utilities yang menggunakan confirmation area dengan fallback ke ipywidgets dialog
"""

from typing import Dict, Any, Callable, Optional
import ipywidgets as widgets
from IPython.display import display, HTML

def show_confirmation_in_area(ui_components: Dict[str, Any], title: str, message: str, 
                             on_confirm: Callable = None, on_cancel: Callable = None,
                             confirm_text: str = "Ya", cancel_text: str = "Batal",
                             danger_mode: bool = False) -> bool:
    """Show confirmation dialog di confirmation area dengan inline buttons"""
    confirmation_area = ui_components.get('confirmation_area')
    
    if not confirmation_area or not hasattr(confirmation_area, 'clear_output'):
        # Fallback ke standard dialog jika confirmation area tidak tersedia
        return _show_fallback_confirmation(title, message, on_confirm, on_cancel, confirm_text, cancel_text, danger_mode)
    
    # Clear existing content
    confirmation_area.clear_output()
    
    # Create confirmation UI
    confirmation_ui = _create_confirmation_ui(
        title, message, on_confirm, on_cancel, 
        confirm_text, cancel_text, danger_mode, confirmation_area
    )
    
    # Display di confirmation area
    with confirmation_area:
        display(confirmation_ui)
    
    return True

def show_info_in_area(ui_components: Dict[str, Any], title: str, message: str, 
                     on_close: Callable = None, close_text: str = "OK") -> bool:
    """Show info message di confirmation area"""
    confirmation_area = ui_components.get('confirmation_area')
    
    if not confirmation_area or not hasattr(confirmation_area, 'clear_output'):
        _log_to_ui_safe(ui_components, f"ℹ️ {title}: {message}", "info")
        return False
    
    # Clear existing content
    confirmation_area.clear_output()
    
    # Create info UI
    info_ui = _create_info_ui(title, message, on_close, close_text, confirmation_area)
    
    # Display di confirmation area
    with confirmation_area:
        display(info_ui)
    
    return True

def show_warning_in_area(ui_components: Dict[str, Any], title: str, message: str,
                        on_close: Callable = None, close_text: str = "Mengerti") -> bool:
    """Show warning message di confirmation area"""
    confirmation_area = ui_components.get('confirmation_area')
    
    if not confirmation_area or not hasattr(confirmation_area, 'clear_output'):
        _log_to_ui_safe(ui_components, f"⚠️ {title}: {message}", "warning")
        return False
    
    # Clear existing content
    confirmation_area.clear_output()
    
    # Create warning UI
    warning_ui = _create_warning_ui(title, message, on_close, close_text, confirmation_area)
    
    # Display di confirmation area
    with confirmation_area:
        display(warning_ui)
    
    return True

def clear_confirmation_area(ui_components: Dict[str, Any]) -> bool:
    """Clear confirmation area"""
    confirmation_area = ui_components.get('confirmation_area')
    
    if confirmation_area and hasattr(confirmation_area, 'clear_output'):
        confirmation_area.clear_output()
        return True
    
    return False

def _create_confirmation_ui(title: str, message: str, on_confirm: Callable, on_cancel: Callable,
                           confirm_text: str, cancel_text: str, danger_mode: bool, 
                           confirmation_area) -> widgets.Widget:
    """Create confirmation UI dengan inline buttons"""
    
    # Style berdasarkan danger mode
    if danger_mode:
        title_color = "#dc3545"
        border_color = "#dc3545"
        confirm_style = "danger"
        icon = "⚠️"
    else:
        title_color = "#007bff"
        border_color = "#007bff"
        confirm_style = "primary"
        icon = "❓"
    
    # Header
    header = widgets.HTML(f"""
    <div style="padding: 10px; border-left: 4px solid {border_color}; 
                background-color: rgba(0,123,255,0.1); margin-bottom: 10px;">
        <h5 style="color: {title_color}; margin: 0; font-size: 16px;">
            {icon} {title}
        </h5>
    </div>
    """)
    
    # Message
    message_widget = widgets.HTML(f"""
    <div style="padding: 10px 0; font-size: 14px; line-height: 1.4;">
        {message}
    </div>
    """)
    
    # Buttons
    confirm_button = widgets.Button(
        description=confirm_text,
        button_style=confirm_style,
        layout=widgets.Layout(width='auto', margin='0 5px 0 0')
    )
    
    cancel_button = widgets.Button(
        description=cancel_text,
        button_style='',
        layout=widgets.Layout(width='auto', margin='0')
    )
    
    # Button handlers
    def handle_confirm(button):
        confirmation_area.clear_output()
        if on_confirm and callable(on_confirm):
            try:
                on_confirm(button)
            except Exception:
                pass
    
    def handle_cancel(button):
        confirmation_area.clear_output()
        if on_cancel and callable(on_cancel):
            try:
                on_cancel(button)
            except Exception:
                pass
    
    confirm_button.on_click(handle_confirm)
    cancel_button.on_click(handle_cancel)
    
    # Button container
    button_container = widgets.HBox([confirm_button, cancel_button], 
        layout=widgets.Layout(
            justify_content='flex-end',
            margin='10px 0 0 0'
        ))
    
    # Main container
    container = widgets.VBox([header, message_widget, button_container],
        layout=widgets.Layout(
            width='100%',
            padding='10px',
            border='1px solid #e0e0e0',
            border_radius='8px',
            background_color='#ffffff'
        ))
    
    return container

def _create_info_ui(title: str, message: str, on_close: Callable, close_text: str,
                   confirmation_area) -> widgets.Widget:
    """Create info UI dengan close button"""
    
    # Header
    header = widgets.HTML(f"""
    <div style="padding: 10px; border-left: 4px solid #17a2b8; 
                background-color: rgba(23,162,184,0.1); margin-bottom: 10px;">
        <h5 style="color: #17a2b8; margin: 0; font-size: 16px;">
            ℹ️ {title}
        </h5>
    </div>
    """)
    
    # Message
    message_widget = widgets.HTML(f"""
    <div style="padding: 10px 0; font-size: 14px; line-height: 1.4;">
        {message}
    </div>
    """)
    
    # Close button
    close_button = widgets.Button(
        description=close_text,
        button_style='info',
        layout=widgets.Layout(width='auto')
    )
    
    def handle_close(button):
        confirmation_area.clear_output()
        if on_close and callable(on_close):
            try:
                on_close(button)
            except Exception:
                pass
    
    close_button.on_click(handle_close)
    
    # Button container
    button_container = widgets.HBox([close_button], 
        layout=widgets.Layout(
            justify_content='flex-end',
            margin='10px 0 0 0'
        ))
    
    # Main container
    container = widgets.VBox([header, message_widget, button_container],
        layout=widgets.Layout(
            width='100%',
            padding='10px',
            border='1px solid #e0e0e0',
            border_radius='8px',
            background_color='#ffffff'
        ))
    
    return container

def _create_warning_ui(title: str, message: str, on_close: Callable, close_text: str,
                      confirmation_area) -> widgets.Widget:
    """Create warning UI dengan close button"""
    
    # Header
    header = widgets.HTML(f"""
    <div style="padding: 10px; border-left: 4px solid #ffc107; 
                background-color: rgba(255,193,7,0.1); margin-bottom: 10px;">
        <h5 style="color: #856404; margin: 0; font-size: 16px;">
            ⚠️ {title}
        </h5>
    </div>
    """)
    
    # Message
    message_widget = widgets.HTML(f"""
    <div style="padding: 10px 0; font-size: 14px; line-height: 1.4;">
        {message}
    </div>
    """)
    
    # Close button
    close_button = widgets.Button(
        description=close_text,
        button_style='warning',
        layout=widgets.Layout(width='auto')
    )
    
    def handle_close(button):
        confirmation_area.clear_output()
        if on_close and callable(on_close):
            try:
                on_close(button)
            except Exception:
                pass
    
    close_button.on_click(handle_close)
    
    # Button container
    button_container = widgets.HBox([close_button], 
        layout=widgets.Layout(
            justify_content='flex-end',
            margin='10px 0 0 0'
        ))
    
    # Main container
    container = widgets.VBox([header, message_widget, button_container],
        layout=widgets.Layout(
            width='100%',
            padding='10px',
            border='1px solid #e0e0e0',
            border_radius='8px',
            background_color='#ffffff'
        ))
    
    return container

def _show_fallback_confirmation(title: str, message: str, on_confirm: Callable, on_cancel: Callable,
                               confirm_text: str, cancel_text: str, danger_mode: bool) -> bool:
    """Fallback confirmation menggunakan standard dialog"""
    try:
        from smartcash.ui.components.dialogs import show_confirmation, show_destructive_confirmation
        
        if danger_mode:
            show_destructive_confirmation(title, message, "item", on_confirm, on_cancel)
        else:
            show_confirmation(title, message, on_confirm, on_cancel)
        
        return True
    except Exception:
        print(f"{'⚠️' if danger_mode else '❓'} {title}: {message}")
        return False

def _log_to_ui_safe(ui_components: Dict[str, Any], message: str, level: str):
    """Safe logging ke UI components"""
    try:
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, message, level)
    except Exception:
        print(f"[{level.upper()}] {message}")

# Convenience functions
confirm_in_area = lambda ui_components, title, msg, on_yes=None, on_no=None: show_confirmation_in_area(ui_components, title, msg, on_yes, on_no)
info_in_area = lambda ui_components, title, msg, on_close=None: show_info_in_area(ui_components, title, msg, on_close)
warn_in_area = lambda ui_components, title, msg, on_close=None: show_warning_in_area(ui_components, title, msg, on_close)
clear_area = lambda ui_components: clear_confirmation_area(ui_components)