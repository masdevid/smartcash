"""
File: smartcash/ui/dataset/augmentation/utils/dialog_utils.py
Deskripsi: Fixed dialog utilities dengan confirmation_area visibility management dan proper API compliance
"""

import ipywidgets as widgets
from IPython.display import display, HTML
from typing import Dict, Any, Callable, Optional

def show_confirmation_in_area(
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
    Show confirmation dialog dengan visibility management dan API compliance.
    
    Args:
        ui_components: Dictionary UI components dengan 'confirmation_area'
        title: Judul dialog
        message: Pesan dialog (mendukung HTML)
        on_confirm: Callback untuk tombol konfirmasi
        on_cancel: Callback untuk tombol batal
        confirm_text: Text tombol konfirmasi
        cancel_text: Text tombol batal
        danger_mode: Menggunakan style merah untuk operasi berbahaya
    """
    confirmation_area = _get_dialog_area(ui_components)
    if not confirmation_area:
        _fallback_console_dialog(title, message, confirm_text, cancel_text)
        return
    
    # Show confirmation area dan set dialog visibility flag
    show_confirmation_area(ui_components)
    ui_components['_dialog_visible'] = True
    
    # Clear area first
    confirmation_area.clear_output(wait=True)
    
    # Create styled dialog
    dialog_style = _get_dialog_style(danger_mode)
    button_style = _get_button_style(danger_mode)
    
    # Create buttons dengan enhanced callbacks
    confirm_btn = widgets.Button(
        description=confirm_text,
        button_style=button_style['confirm'],
        layout=widgets.Layout(width='120px', height='32px', margin='0 5px 0 0')
    )
    
    cancel_btn = widgets.Button(
        description=cancel_text,
        button_style=button_style['cancel'],
        layout=widgets.Layout(width='120px', height='32px', margin='0')
    )
    
    # Enhanced callback wrappers dengan visibility management
    def _enhanced_confirm_handler(btn):
        try:
            hide_confirmation_area(ui_components)  # Hide area first
            ui_components['_dialog_visible'] = False  # Reset flag
            if on_confirm:
                on_confirm(btn)
        except Exception as e:
            _handle_callback_error(ui_components, f"Confirm callback error: {str(e)}")
    
    def _enhanced_cancel_handler(btn):
        try:
            hide_confirmation_area(ui_components)  # Hide area first
            ui_components['_dialog_visible'] = False  # Reset flag
            if on_cancel:
                on_cancel(btn)
        except Exception as e:
            _handle_callback_error(ui_components, f"Cancel callback error: {str(e)}")
    
    # Bind callbacks
    confirm_btn.on_click(_enhanced_confirm_handler)
    cancel_btn.on_click(_enhanced_cancel_handler)
    
    # Create dialog HTML dengan responsive design
    dialog_html = f"""
    <div style="{dialog_style['container']}">
        <div style="{dialog_style['header']}">
            <h5 style="margin: 0; color: {dialog_style['title_color']};">{title}</h5>
        </div>
        <div style="{dialog_style['body']}">
            {message}
        </div>
    </div>
    """
    
    # Display dialog
    button_container = widgets.HBox(
        [confirm_btn, cancel_btn],
        layout=widgets.Layout(
            justify_content='center',
            margin='10px 0 5px 0',
            width='100%'
        )
    )
    
    with confirmation_area:
        display(HTML(dialog_html))
        display(button_container)

def show_info_in_area(
    ui_components: Dict[str, Any],
    title: str,
    message: str,
    on_close: Callable = None,
    close_text: str = "Tutup",
    dialog_type: str = "info"
) -> None:
    """
    Show info dialog dengan single close button dan visibility management.
    
    Args:
        ui_components: Dictionary UI components
        title: Judul dialog
        message: Pesan dialog
        on_close: Callback untuk tombol close
        close_text: Text tombol close
        dialog_type: Type dialog ('info', 'success', 'warning', 'error')
    """
    confirmation_area = _get_dialog_area(ui_components)
    if not confirmation_area:
        _fallback_console_info(title, message, dialog_type)
        return
    
    # Show confirmation area dan set dialog visibility flag
    show_confirmation_area(ui_components)
    ui_components['_dialog_visible'] = True
    
    # Clear area first
    confirmation_area.clear_output(wait=True)
    
    # Get type-specific styling
    type_config = _get_dialog_type_config(dialog_type)
    
    # Create close button
    close_btn = widgets.Button(
        description=close_text,
        button_style='',
        layout=widgets.Layout(width='120px', height='32px', margin='0')
    )
    
    # Enhanced close callback
    def _enhanced_close_handler(btn):
        try:
            hide_confirmation_area(ui_components)  # Hide area first
            ui_components['_dialog_visible'] = False  # Reset flag
            if on_close:
                on_close(btn)
        except Exception as e:
            _handle_callback_error(ui_components, f"Close callback error: {str(e)}")
    
    close_btn.on_click(_enhanced_close_handler)
    
    # Create info dialog HTML
    dialog_html = f"""
    <div style="background: white; border: 2px solid {type_config['border_color']}; 
                border-radius: 8px; padding: 0; margin: 10px auto; max-width: 500px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1); overflow: hidden;">
        <div style="background: {type_config['bg_color']}; padding: 12px 15px; 
                    border-bottom: 1px solid {type_config['border_color']};">
            <h5 style="margin: 0; color: {type_config['text_color']};">{type_config['icon']} {title}</h5>
        </div>
        <div style="padding: 15px; line-height: 1.5; color: #333;">
            {message}
        </div>
    </div>
    """
    
    # Display info dialog
    button_container = widgets.HBox(
        [close_btn],
        layout=widgets.Layout(
            justify_content='center',
            margin='10px 0 5px 0',
            width='100%'
        )
    )
    
    with confirmation_area:
        display(HTML(dialog_html))
        display(button_container)

def show_warning_in_area(
    ui_components: Dict[str, Any],
    title: str,
    message: str,
    on_close: Callable = None
) -> None:
    """Shortcut untuk warning dialog."""
    show_info_in_area(ui_components, title, message, on_close, "Tutup", "warning")

def clear_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """
    Clear confirmation area dan reset visibility flag.
    
    Args:
        ui_components: Dictionary UI components
    """
    confirmation_area = _get_dialog_area(ui_components)
    if confirmation_area and hasattr(confirmation_area, 'clear_output'):
        confirmation_area.clear_output(wait=True)
    
    # Hide confirmation area setelah clear
    hide_confirmation_area(ui_components)
    
    # Reset dialog visibility flag
    ui_components['_dialog_visible'] = False

def hide_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Hide confirmation area dengan visibility control"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'layout'):
        confirmation_area.layout.visibility = 'hidden'
        confirmation_area.layout.height = '0px'
        confirmation_area.layout.margin = '0px'

def show_confirmation_area(ui_components: Dict[str, Any]) -> None:
    """Show confirmation area dengan restore layout"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'layout'):
        confirmation_area.layout.visibility = 'visible'
        confirmation_area.layout.height = 'auto'
        confirmation_area.layout.margin = '10px 0'

def is_dialog_visible(ui_components: Dict[str, Any]) -> bool:
    """
    Check apakah dialog sedang ditampilkan.
    
    Args:
        ui_components: Dictionary UI components
        
    Returns:
        Boolean status visibility dialog
    """
    return ui_components.get('_dialog_visible', False)

def _get_dialog_area(ui_components: Dict[str, Any]) -> Optional[widgets.Output]:
    """Get dialog area dengan fallback priority."""
    return (ui_components.get('confirmation_area') or 
            ui_components.get('dialog_area') or 
            ui_components.get('status'))

def _get_dialog_style(danger_mode: bool = False) -> Dict[str, str]:
    """Get dialog styling berdasarkan mode."""
    if danger_mode:
        return {
            'container': "background: white; border: 2px solid #dc3545; border-radius: 8px; padding: 0; margin: 10px auto; max-width: 500px; box-shadow: 0 4px 6px rgba(220,53,69,0.2); overflow: hidden;",
            'header': "background: #f8d7da; padding: 12px 15px; border-bottom: 1px solid #dc3545;",
            'body': "padding: 15px; line-height: 1.5; color: #333;",
            'title_color': "#721c24"
        }
    else:
        return {
            'container': "background: white; border: 2px solid #17a2b8; border-radius: 8px; padding: 0; margin: 10px auto; max-width: 500px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); overflow: hidden;",
            'header': "background: #d1ecf1; padding: 12px 15px; border-bottom: 1px solid #17a2b8;",
            'body': "padding: 15px; line-height: 1.5; color: #333;",
            'title_color': "#0c5460"
        }

def _get_button_style(danger_mode: bool = False) -> Dict[str, str]:
    """Get button styling berdasarkan mode."""
    if danger_mode:
        return {'confirm': 'danger', 'cancel': ''}
    else:
        return {'confirm': 'primary', 'cancel': ''}

def _get_dialog_type_config(dialog_type: str) -> Dict[str, str]:
    """Get configuration untuk berbagai dialog types."""
    configs = {
        'info': {
            'icon': 'ℹ️',
            'bg_color': '#d1ecf1',
            'border_color': '#17a2b8',
            'text_color': '#0c5460'
        },
        'success': {
            'icon': '✅',
            'bg_color': '#d4edda',
            'border_color': '#28a745',
            'text_color': '#155724'
        },
        'warning': {
            'icon': '⚠️',
            'bg_color': '#fff3cd',
            'border_color': '#ffc107',
            'text_color': '#856404'
        },
        'error': {
            'icon': '❌',
            'bg_color': '#f8d7da',
            'border_color': '#dc3545',
            'text_color': '#721c24'
        }
    }
    return configs.get(dialog_type, configs['info'])

def _handle_callback_error(ui_components: Dict[str, Any], error_message: str) -> None:
    """Handle callback errors dengan logging."""
    try:
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, f"❌ Dialog callback error: {error_message}", "error")
    except Exception:
        print(f"❌ Dialog callback error: {error_message}")

def _fallback_console_dialog(title: str, message: str, confirm_text: str, cancel_text: str) -> None:
    """Fallback ke console jika dialog area tidak tersedia."""
    print(f"🔄 DIALOG: {title}")
    print(f"📋 {message}")
    print(f"⚠️ Dialog area tidak tersedia - operation requires manual confirmation")

def _fallback_console_info(title: str, message: str, dialog_type: str) -> None:
    """Fallback console untuk info dialog."""
    icons = {'info': 'ℹ️', 'success': '✅', 'warning': '⚠️', 'error': '❌'}
    icon = icons.get(dialog_type, 'ℹ️')
    print(f"{icon} {title}: {message}")

# Convenience functions untuk backward compatibility
show_confirmation_dialog = show_confirmation_in_area
show_info_dialog = show_info_in_area
clear_dialog_area = clear_confirmation_area