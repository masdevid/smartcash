"""
File: smartcash/ui/components/dialog/confirmation_dialog.py
Deskripsi: Simple dialog dengan proper cleanup (REPLACE existing file)
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
    """Show confirmation dialog dengan auto-cleanup"""
    dialog_area = ui_components.get('confirmation_area') or ui_components.get('dialog_area')
    if not dialog_area:
        print(f"⚠️ {title}: {message}")
        if on_confirm:
            on_confirm()
        return
    
    # Clear existing content
    with dialog_area:
        clear_output(wait=True)
    
    # Create buttons dengan auto-destroy callbacks
    def handle_confirm(btn):
        with dialog_area:
            clear_output(wait=True)
        if on_confirm:
            on_confirm()
    
    def handle_cancel(btn):
        with dialog_area:
            clear_output(wait=True)
        if on_cancel:
            on_cancel()
    
    confirm_style = 'danger' if danger_mode else 'primary'
    confirm_btn = widgets.Button(
        description=confirm_text,
        button_style=confirm_style,
        layout=widgets.Layout(width='120px', margin='4px')
    )
    confirm_btn.on_click(handle_confirm)
    
    cancel_btn = widgets.Button(
        description=cancel_text,
        button_style='',
        layout=widgets.Layout(width='100px', margin='4px')
    )
    cancel_btn.on_click(handle_cancel)
    
    # Simple dialog layout
    border_color = '#dc3545' if danger_mode else '#007bff'
    bg_color = '#fff5f5' if danger_mode else '#f8f9fa'
    
    dialog = widgets.VBox([
        widgets.HTML(f"<h4 style='color:{border_color};text-align:center;margin:0 0 10px 0;'>{title}</h4>"),
        widgets.HTML(f"<div style='text-align:center;margin-bottom:15px;'>{message}</div>"),
        widgets.HBox([confirm_btn, cancel_btn], layout=widgets.Layout(justify_content='center'))
    ], layout=widgets.Layout(
        padding='15px',
        border=f'2px solid {border_color}',
        border_radius='8px',
        background_color=bg_color,
        margin='10px 0'
    ))
    
    # Display dialog
    with dialog_area:
        display(dialog)

def show_info_dialog(
    ui_components: Dict[str, Any],
    title: str,
    message: str,
    on_close: Callable = None,
    close_text: str = "Tutup",
    dialog_type: str = "info"
) -> None:
    """Show info dialog"""
    dialog_area = ui_components.get('confirmation_area') or ui_components.get('dialog_area')
    if not dialog_area:
        print(f"ℹ️ {title}: {message}")
        if on_close:
            on_close()
        return
    
    def handle_close(btn):
        with dialog_area:
            clear_output(wait=True)
        if on_close:
            on_close()
    
    colors = {'info': '#17a2b8', 'success': '#28a745', 'warning': '#ffc107', 'error': '#dc3545'}
    color = colors.get(dialog_type, '#17a2b8')
    
    close_btn = widgets.Button(description=close_text, layout=widgets.Layout(width='100px'))
    close_btn.on_click(handle_close)
    
    dialog = widgets.VBox([
        widgets.HTML(f"<h4 style='color:{color};text-align:center;margin:0 0 10px 0;'>{title}</h4>"),
        widgets.HTML(f"<div style='text-align:center;margin-bottom:15px;'>{message}</div>"),
        widgets.HBox([close_btn], layout=widgets.Layout(justify_content='center'))
    ], layout=widgets.Layout(
        padding='15px',
        border=f'2px solid {color}',
        border_radius='8px',
        margin='10px 0'
    ))
    
    with dialog_area:
        clear_output(wait=True)
        display(dialog)

def clear_dialog_area(ui_components: Dict[str, Any]) -> None:
    """Clear dialog area"""
    dialog_area = ui_components.get('confirmation_area') or ui_components.get('dialog_area')
    if dialog_area:
        with dialog_area:
            clear_output(wait=True)

def is_dialog_visible(ui_components: Dict[str, Any]) -> bool:
    """Check if dialog visible"""
    return False  # Simplified - tidak perlu state tracking