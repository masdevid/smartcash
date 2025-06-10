"""
File: smartcash/ui/dataset/preprocessing/utils/confirmation_utils.py
Deskripsi: Enhanced confirmation utilities untuk cleanup operations dengan DRY principles
"""

from typing import Dict, Any, Callable, Optional
from ipywidgets import VBox, HBox, Button, HTML, Output
from IPython.display import display, clear_output
import threading
import time

def show_cleanup_confirmation(ui_components: Dict[str, Any], files_count: int, target_split: str = "semua split") -> Optional[bool]:
    """🤔 Show cleanup confirmation dengan enhanced UI integration"""
    if 'confirmation_area' not in ui_components:
        return None  # No UI confirmation area available
    
    # Use thread-safe confirmation
    result_container = {'confirmed': None, 'completed': False}
    
    def on_confirm():
        result_container['confirmed'] = True
        result_container['completed'] = True
        _clear_confirmation_area(ui_components)
        _show_confirmation_feedback(ui_components, "✅ Cleanup dikonfirmasi", "success")
    
    def on_cancel():
        result_container['confirmed'] = False
        result_container['completed'] = True
        _clear_confirmation_area(ui_components)
        _show_confirmation_feedback(ui_components, "🚫 Cleanup dibatalkan", "info")
    
    # Show confirmation dialog
    _show_cleanup_confirmation_dialog(
        ui_components=ui_components,
        files_count=files_count,
        target_split=target_split,
        on_confirm=on_confirm,
        on_cancel=on_cancel
    )
    
    # Wait for user response dengan timeout
    timeout = 300  # 5 minutes
    start_time = time.time()
    
    while not result_container['completed'] and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    # Return result atau None jika timeout
    return result_container['confirmed'] if result_container['completed'] else None

def _show_cleanup_confirmation_dialog(ui_components: Dict[str, Any], files_count: int, 
                                    target_split: str, on_confirm: Callable, on_cancel: Callable):
    """Show cleanup confirmation dialog"""
    title = "⚠️ Konfirmasi Cleanup Dataset"
    message = f"""
    Anda akan menghapus <strong>{files_count:,} file preprocessed</strong> untuk {target_split}.
    
    <div style="color: #dc3545; font-weight: bold; margin: 10px 0;">
        ⚠️ Tindakan ini tidak dapat dibatalkan!
    </div>
    
    Apakah Anda yakin ingin melanjutkan?
    """
    
    confirm_btn = Button(
        description="Ya, Hapus",
        button_style='danger',
        icon='trash',
        layout={'width': '120px', 'margin': '5px'}
    )
    
    cancel_btn = Button(
        description="Batal",
        button_style='',
        icon='times',
        layout={'width': '120px', 'margin': '5px'}
    )
    
    confirm_btn.on_click(lambda btn: on_confirm())
    cancel_btn.on_click(lambda btn: on_cancel())
    
    dialog = VBox([
        HTML(f"<h4 style='color: #dc3545; margin-bottom: 15px;'>{title}</h4>"),
        HTML(f'<div style="margin-bottom: 20px;">{message}</div>'),
        HBox([confirm_btn, cancel_btn], layout={
            'justify_content': 'center',
            'align_items': 'center'
        })
    ], layout={
        'padding': '20px',
        'border': '2px solid #dc3545',
        'border_radius': '8px',
        'background_color': '#fff5f5',
        'width': '100%',
        'max_width': '500px'
    })
    
    # Display di confirmation area
    with ui_components['confirmation_area']:
        clear_output(wait=True)
        display(dialog)

def show_preprocessing_confirmation(ui_components: Dict[str, Any], action: str = "preprocessing") -> Optional[bool]:
    """🤔 Show general preprocessing confirmation"""
    if 'confirmation_area' not in ui_components:
        return None
    
    result_container = {'confirmed': None, 'completed': False}
    
    def on_confirm():
        result_container['confirmed'] = True
        result_container['completed'] = True
        _clear_confirmation_area(ui_components)
    
    def on_cancel():
        result_container['confirmed'] = False
        result_container['completed'] = True
        _clear_confirmation_area(ui_components)
    
    _show_general_confirmation_dialog(
        ui_components=ui_components,
        action=action,
        on_confirm=on_confirm,
        on_cancel=on_cancel
    )
    
    # Wait for response
    timeout = 180  # 3 minutes
    start_time = time.time()
    
    while not result_container['completed'] and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    return result_container['confirmed'] if result_container['completed'] else None

def _show_general_confirmation_dialog(ui_components: Dict[str, Any], action: str,
                                    on_confirm: Callable, on_cancel: Callable):
    """Show general confirmation dialog"""
    action_configs = {
        'preprocessing': {
            'title': '🔄 Konfirmasi Preprocessing',
            'message': 'Apakah Anda yakin ingin memulai preprocessing dataset?',
            'confirm_text': 'Ya, Mulai',
            'icon': 'play'
        },
        'reset': {
            'title': '🔄 Konfirmasi Reset',
            'message': 'Apakah Anda yakin ingin mereset konfigurasi ke nilai default?',
            'confirm_text': 'Ya, Reset',
            'icon': 'refresh'
        }
    }
    
    config = action_configs.get(action, action_configs['preprocessing'])
    
    confirm_btn = Button(
        description=config['confirm_text'],
        button_style='primary',
        icon=config['icon'],
        layout={'width': '120px', 'margin': '5px'}
    )
    
    cancel_btn = Button(
        description="Batal",
        button_style='',
        icon='times',
        layout={'width': '120px', 'margin': '5px'}
    )
    
    confirm_btn.on_click(lambda btn: on_confirm())
    cancel_btn.on_click(lambda btn: on_cancel())
    
    dialog = VBox([
        HTML(f"<h4 style='color: #007bff; margin-bottom: 15px;'>{config['title']}</h4>"),
        HTML(f'<p style="margin-bottom: 20px;">{config["message"]}</p>'),
        HBox([confirm_btn, cancel_btn], layout={
            'justify_content': 'center',
            'align_items': 'center'
        })
    ], layout={
        'padding': '20px',
        'border': '2px solid #007bff',
        'border_radius': '8px',
        'background_color': '#f8f9fa',
        'width': '100%',
        'max_width': '400px'
    })
    
    with ui_components['confirmation_area']:
        clear_output(wait=True)
        display(dialog)

def _clear_confirmation_area(ui_components: Dict[str, Any]):
    """Clear confirmation area"""
    try:
        if 'confirmation_area' in ui_components:
            with ui_components['confirmation_area']:
                clear_output(wait=True)
    except Exception:
        pass

def _show_confirmation_feedback(ui_components: Dict[str, Any], message: str, level: str = "info"):
    """Show brief confirmation feedback"""
    try:
        from smartcash.ui.dataset.preprocessing.utils.ui_utils import log_to_accordion
        log_to_accordion(ui_components, message, level)
    except Exception:
        pass

# Backward compatibility functions
def create_cleanup_confirmation_dialog(ui_components: Dict[str, Any], files_count: int, 
                                     on_confirm: Callable, on_cancel: Callable,
                                     target_split: str = "semua split"):
    """Backward compatibility wrapper"""
    _show_cleanup_confirmation_dialog(ui_components, files_count, target_split, on_confirm, on_cancel)

def clear_confirmation_area(ui_components: Dict[str, Any]):
    """Backward compatibility wrapper"""
    _clear_confirmation_area(ui_components)