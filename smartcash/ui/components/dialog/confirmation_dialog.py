"""
File: smartcash/ui/components/dialog/confirmation_dialog.py
Deskripsi: Confirmation dialog component yang reusable dengan callback support
"""

from typing import Dict, Any, Callable, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output

def show_confirmation_dialog(
    ui_components: Dict[str, Any],
    title: str,
    message: str,
    on_confirm: Callable = None,
    on_cancel: Callable = None,
    confirm_text: str = "Ya",
    cancel_text: str = "Batal",
    danger_mode: bool = False
) -> None:
    """
    Show confirmation dialog dengan callback support.
    
    Args:
        ui_components: Dictionary UI components
        title: Dialog title
        message: Dialog message (dapat berisi HTML)
        on_confirm: Callback untuk konfirmasi
        on_cancel: Callback untuk pembatalan
        confirm_text: Text tombol konfirmasi
        cancel_text: Text tombol batal
        danger_mode: Apakah menggunakan style danger (merah)
    """
    dialog_area = ui_components.get('confirmation_area') or ui_components.get('dialog_area')
    if not dialog_area:
        print(f"⚠️ Dialog area tidak tersedia. {title}: {message}")
        return
    
    # Clear existing dialog
    clear_dialog_area(ui_components)
    
    # Set dialog visibility flag
    ui_components['_dialog_visible'] = True
    
    # Create dialog
    dialog = _create_dialog_widget(
        title=title,
        message=message,
        confirm_text=confirm_text,
        cancel_text=cancel_text,
        danger_mode=danger_mode,
        on_confirm=lambda: _handle_confirm(ui_components, on_confirm),
        on_cancel=lambda: _handle_cancel(ui_components, on_cancel)
    )
    
    # Display dialog
    with dialog_area:
        clear_output(wait=True)
        display(dialog)

def show_info_dialog(
    ui_components: Dict[str, Any],
    title: str,
    message: str,
    on_close: Callable = None,
    close_text: str = "Tutup",
    dialog_type: str = "info"
) -> None:
    """
    Show info dialog dengan close callback.
    
    Args:
        ui_components: Dictionary UI components
        title: Dialog title
        message: Dialog message
        on_close: Callback untuk close
        close_text: Text tombol close
        dialog_type: Type dialog ('info', 'success', 'warning', 'error')
    """
    dialog_area = ui_components.get('confirmation_area') or ui_components.get('dialog_area')
    if not dialog_area:
        print(f"ℹ️ {title}: {message}")
        return
    
    # Set dialog visibility flag
    ui_components['_dialog_visible'] = True
    
    # Create info dialog
    dialog = _create_info_dialog_widget(
        title=title,
        message=message,
        close_text=close_text,
        dialog_type=dialog_type,
        on_close=lambda: _handle_close(ui_components, on_close)
    )
    
    # Display dialog
    with dialog_area:
        clear_output(wait=True)
        display(dialog)

def clear_dialog_area(ui_components: Dict[str, Any]) -> None:
    """Clear dialog area dan reset visibility flag"""
    dialog_area = ui_components.get('confirmation_area') or ui_components.get('dialog_area')
    if dialog_area:
        with dialog_area:
            clear_output(wait=True)
    
    # Clear visibility flag
    ui_components.pop('_dialog_visible', None)
    
    # Force widget refresh
    if dialog_area and hasattr(dialog_area, 'layout'):
        dialog_area.layout.height = '0px'
        dialog_area.layout.height = 'auto'

def is_dialog_visible(ui_components: Dict[str, Any]) -> bool:
    """Check apakah dialog sedang visible"""
    return ui_components.get('_dialog_visible', False)

# === INTERNAL FUNCTIONS ===

def _create_dialog_widget(
    title: str,
    message: str,
    confirm_text: str,
    cancel_text: str,
    danger_mode: bool,
    on_confirm: Callable,
    on_cancel: Callable
) -> widgets.VBox:
    """Create confirmation dialog widget"""
    
    # Create buttons
    confirm_style = 'danger' if danger_mode else 'primary'
    confirm_icon = 'trash' if danger_mode else 'check'
    
    confirm_btn = widgets.Button(
        description=confirm_text,
        button_style=confirm_style,
        icon=confirm_icon,
        layout=widgets.Layout(width='120px', height='32px', margin='4px')
    )
    
    cancel_btn = widgets.Button(
        description=cancel_text,
        button_style='',
        icon='times',
        layout=widgets.Layout(width='100px', height='32px', margin='4px')
    )
    
    # Bind events
    confirm_btn.on_click(lambda btn: on_confirm())
    cancel_btn.on_click(lambda btn: on_cancel())
    
    # Dialog styling
    border_color = '#dc3545' if danger_mode else '#007bff'
    bg_color = '#fff5f5' if danger_mode else '#f8f9fa'
    
    # Create dialog structure
    header = widgets.HTML(f"""
        <div style='text-align: center; margin-bottom: 15px;'>
            <h4 style='color: {border_color}; margin: 0; font-size: 16px;'>{title}</h4>
        </div>
    """)
    
    content = widgets.HTML(f"""
        <div style='margin-bottom: 20px; text-align: center; line-height: 1.4; font-size: 14px;'>
            {message}
        </div>
    """)
    
    buttons = widgets.HBox([confirm_btn, cancel_btn], layout=widgets.Layout(
        justify_content='center',
        align_items='center',
        margin='10px 0 0 0'
    ))
    
    dialog = widgets.VBox([header, content, buttons], layout=widgets.Layout(
        padding='20px',
        border=f'2px solid {border_color}',
        border_radius='8px',
        background_color=bg_color,
        width='100%',
        max_width='500px',
        margin='10px auto',
        box_shadow='0 4px 8px rgba(0,0,0,0.1)'
    ))
    
    return dialog

def _create_info_dialog_widget(
    title: str,
    message: str,
    close_text: str,
    dialog_type: str,
    on_close: Callable
) -> widgets.VBox:
    """Create info dialog widget"""
    
    # Type-specific styling
    type_config = {
        'info': {'color': '#17a2b8', 'bg': '#d1ecf1', 'icon': 'info'},
        'success': {'color': '#28a745', 'bg': '#d4edda', 'icon': 'check'},
        'warning': {'color': '#ffc107', 'bg': '#fff3cd', 'icon': 'exclamation-triangle'},
        'error': {'color': '#dc3545', 'bg': '#f8d7da', 'icon': 'times'}
    }
    
    config = type_config.get(dialog_type, type_config['info'])
    
    # Create close button
    close_btn = widgets.Button(
        description=close_text,
        button_style='',
        icon='times',
        layout=widgets.Layout(width='100px', height='32px', margin='4px')
    )
    
    close_btn.on_click(lambda btn: on_close())
    
    # Create dialog structure
    header = widgets.HTML(f"""
        <div style='text-align: center; margin-bottom: 15px;'>
            <h4 style='color: {config["color"]}; margin: 0; font-size: 16px;'>{title}</h4>
        </div>
    """)
    
    content = widgets.HTML(f"""
        <div style='margin-bottom: 20px; text-align: center; line-height: 1.4; font-size: 14px;'>
            {message}
        </div>
    """)
    
    buttons = widgets.HBox([close_btn], layout=widgets.Layout(
        justify_content='center',
        align_items='center',
        margin='10px 0 0 0'
    ))
    
    dialog = widgets.VBox([header, content, buttons], layout=widgets.Layout(
        padding='20px',
        border=f'2px solid {config["color"]}',
        border_radius='8px',
        background_color=config['bg'],
        width='100%',
        max_width='500px',
        margin='10px auto',
        box_shadow='0 4px 8px rgba(0,0,0,0.1)'
    ))
    
    return dialog

def _handle_confirm(ui_components: Dict[str, Any], callback: Callable):
    """Handle confirm action"""
    try:
        # Clear dialog immediately
        clear_dialog_area(ui_components)
        
        # Execute callback after clearing
        if callback:
            callback()
    except Exception as e:
        print(f"❌ Error in confirm callback: {str(e)}")

def _handle_cancel(ui_components: Dict[str, Any], callback: Callable):
    """Handle cancel action"""
    try:
        # Clear dialog immediately
        clear_dialog_area(ui_components)
        
        # Execute callback after clearing
        if callback:
            callback()
    except Exception as e:
        print(f"❌ Error in cancel callback: {str(e)}")

def _handle_close(ui_components: Dict[str, Any], callback: Callable):
    """Handle close action untuk info dialog"""
    try:
        # Clear dialog immediately
        clear_dialog_area(ui_components)
        
        # Execute callback after clearing
        if callback:
            callback()
    except Exception as e:
        print(f"❌ Error in close callback: {str(e)}")