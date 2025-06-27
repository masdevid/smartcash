"""
File: smartcash/ui/components/dialog/confirmation_dialog.py
Deskripsi: Komponen dialog untuk konfirmasi dan interaksi pengguna
"""

from typing import Dict, Any, Callable, Optional, Tuple
import ipywidgets as widgets
from IPython.display import display, clear_output

def create_confirmation_area(
    width: str = '100%',
    min_height: str = '0px',
    max_height: str = '800px',
    margin: str = '10px 0',
    padding: str = '5px',
    border: str = '1px solid #e0e0e0',
    border_radius: str = '4px',
    background_color: str = '#fafafa',
    overflow: str = 'auto',
    visibility: str = 'hidden'
) -> Tuple[widgets.Output, Dict[str, str]]:
    """
    Membuat area konfirmasi yang dapat digunakan untuk menampilkan dialog.
    
    Args:
        width: Lebar area konfirmasi
        min_height: Tinggi minimum area
        max_height: Tinggi maksimum area
        margin: Margin area
        padding: Padding area
        border: Gaya border
        border_radius: Sudut border
        background_color: Warna latar belakang
        overflow: Pengaturan overflow
        visibility: Visibilitas awal ('hidden' atau 'visible')
        
    Returns:
        Tuple berisi:
        - Widget Output yang dapat digunakan untuk menampilkan konten
        - Dictionary berisi layout yang digunakan (untuk referensi)
    """
    layout = {
        'width': width,
        'min_height': min_height,
        'max_height': max_height,
        'margin': margin,
        'padding': padding,
        'border': border,
        'border_radius': border_radius,
        'background_color': background_color,
        'overflow': overflow,
        'visibility': visibility
    }
    
    confirmation_area = widgets.Output(
        layout=widgets.Layout(**{
            k: v for k, v in layout.items() 
            if k not in ['visibility']  # visibility diatur melalui display property
        })
    )
    
    # Set initial visibility
    confirmation_area.layout.display = 'none' if visibility == 'hidden' else 'flex'
    
    return confirmation_area, layout

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
    
    # Ensure dialog area is visible
    if hasattr(dialog_area, 'layout') and hasattr(dialog_area.layout, 'display'):
        dialog_area.layout.display = 'block'
    
    # Clear existing content
    with dialog_area:
        clear_output(wait=True)
    
    # Create buttons dengan auto-destroy callbacks
    def handle_confirm(btn):
        with dialog_area:
            clear_output(wait=True)
            if hasattr(dialog_area, 'layout') and hasattr(dialog_area.layout, 'display'):
                dialog_area.layout.display = 'none'
        if on_confirm:
            on_confirm()
    
    def handle_cancel(btn):
        with dialog_area:
            clear_output(wait=True)
            if hasattr(dialog_area, 'layout') and hasattr(dialog_area.layout, 'display'):
                dialog_area.layout.display = 'none'
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