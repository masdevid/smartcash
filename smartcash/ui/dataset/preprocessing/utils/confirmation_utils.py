"""
File: smartcash/ui/dataset/preprocessing/utils/confirmation_utils.py
Deskripsi: Fixed confirmation utilities dengan cancel/continue yang berfungsi dan auto hide
"""

from typing import Dict, Any, Callable, Optional
from ipywidgets import VBox, HBox, Button, HTML, Output
from IPython.display import display, clear_output
import time

def show_cleanup_confirmation(ui_components: Dict[str, Any], detailed_stats: str) -> Optional[bool]:
    """Fixed cleanup confirmation dengan detailed stats"""
    if 'confirmation_area' not in ui_components:
        return True  # Default proceed jika no confirmation area
    
    # Clear area first
    _clear_confirmation_area(ui_components)
    
    # Show confirmation
    return _show_interactive_confirmation(
        ui_components=ui_components,
        title="⚠️ Konfirmasi Cleanup Dataset",
        message=f"Anda akan menghapus data preprocessed:<br><strong>{detailed_stats}</strong><br><span style='color: #dc3545;'>⚠️ Tindakan ini tidak dapat dibatalkan!</span>",
        confirm_text="Ya, Hapus",
        cancel_text="Batal",
        danger_mode=True
    )

def show_preprocessing_confirmation(ui_components: Dict[str, Any], message: str = None) -> Optional[bool]:
    """Fixed preprocessing confirmation"""
    if 'confirmation_area' not in ui_components:
        return True
    
    _clear_confirmation_area(ui_components)
    
    default_message = "Apakah Anda yakin ingin memulai preprocessing dataset?"
    return _show_interactive_confirmation(
        ui_components=ui_components,
        title="🔄 Konfirmasi Preprocessing",
        message=message or default_message,
        confirm_text="Ya, Mulai",
        cancel_text="Batal",
        danger_mode=False
    )

def _show_interactive_confirmation(ui_components: Dict[str, Any], title: str, message: str, 
                                 confirm_text: str, cancel_text: str, danger_mode: bool = False) -> Optional[bool]:
    """Fixed interactive confirmation dengan proper state handling"""
    confirmation_area = ui_components['confirmation_area']
    
    # State container
    result = {'value': None, 'completed': False}
    
    # Buttons
    confirm_style = 'danger' if danger_mode else 'primary'
    confirm_btn = Button(
        description=confirm_text,
        button_style=confirm_style,
        icon='trash' if danger_mode else 'check',
        layout={'width': '120px', 'margin': '5px'}
    )
    
    cancel_btn = Button(
        description=cancel_text,
        button_style='',
        icon='times',
        layout={'width': '120px', 'margin': '5px'}
    )
    
    # Event handlers
    def on_confirm(btn):
        result['value'] = True
        result['completed'] = True
        _show_brief_feedback(ui_components, f"✅ {confirm_text} dikonfirmasi", "success")
        _auto_clear_after_delay(ui_components, 2)
    
    def on_cancel(btn):
        result['value'] = False
        result['completed'] = True
        _show_brief_feedback(ui_components, f"🚫 {cancel_text}", "info")
        _auto_clear_after_delay(ui_components, 2)
    
    confirm_btn.on_click(on_confirm)
    cancel_btn.on_click(on_cancel)
    
    # Dialog
    border_color = '#dc3545' if danger_mode else '#007bff'
    bg_color = '#fff5f5' if danger_mode else '#f8f9fa'
    
    dialog = VBox([
        HTML(f"<h4 style='color: {border_color}; margin-bottom: 15px;'>{title}</h4>"),
        HTML(f'<div style="margin-bottom: 20px;">{message}</div>'),
        HBox([confirm_btn, cancel_btn], layout={
            'justify_content': 'center',
            'align_items': 'center'
        })
    ], layout={
        'padding': '20px',
        'border': f'2px solid {border_color}',
        'border_radius': '8px',
        'background_color': bg_color,
        'width': '100%',
        'max_width': '500px',
        'margin': '4px auto'
    })
    
    # Display
    with confirmation_area:
        clear_output(wait=True)
        display(dialog)
    
    # Wait for user response dengan timeout
    timeout = 60  # 1 minute
    start_time = time.time()
    
    while not result['completed'] and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    return result['value'] if result['completed'] else None

def _show_brief_feedback(ui_components: Dict[str, Any], message: str, level: str):
    """Show brief feedback message"""
    confirmation_area = ui_components.get('confirmation_area')
    if not confirmation_area:
        return
    
    colors = {
        'success': '#28a745',
        'info': '#17a2b8',
        'warning': '#ffc107',
        'error': '#dc3545'
    }
    
    color = colors.get(level, '#17a2b8')
    
    feedback_html = HTML(f"""
    <div style='padding: 15px; background-color: rgba(248,249,250,0.9); margin: 4px auto;
               border: 1px solid {color}; border-radius: 6px; text-align: center;'>
        <span style='color: {color}; font-weight: 500;'>{message}</span>
    </div>
    """)
    
    with confirmation_area:
        clear_output(wait=True)
        display(feedback_html)

def _auto_clear_after_delay(ui_components: Dict[str, Any], delay_seconds: int):
    """Auto clear confirmation area after delay"""
    import threading
    
    def clear_delayed():
        time.sleep(delay_seconds)
        _clear_confirmation_area(ui_components)
    
    # Use thread untuk delay clearing tanpa blocking UI
    thread = threading.Thread(target=clear_delayed, daemon=True)
    thread.start()

def _clear_confirmation_area(ui_components: Dict[str, Any]):
    """Clear confirmation area"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        with confirmation_area:
            clear_output(wait=True)

# Public utilities
def clear_confirmation_area(ui_components: Dict[str, Any]):
    """Public function to clear confirmation area"""
    _clear_confirmation_area(ui_components)

def show_info_message(ui_components: Dict[str, Any], message: str, auto_clear: bool = True):
    """Show info message dalam confirmation area"""
    if 'confirmation_area' not in ui_components:
        return
    
    _show_brief_feedback(ui_components, message, "info")
    
    if auto_clear:
        _auto_clear_after_delay(ui_components, 3)

def show_success_message(ui_components: Dict[str, Any], message: str, auto_clear: bool = True):
    """Show success message dalam confirmation area"""
    if 'confirmation_area' not in ui_components:
        return
    
    _show_brief_feedback(ui_components, message, "success")
    
    if auto_clear:
        _auto_clear_after_delay(ui_components, 3)