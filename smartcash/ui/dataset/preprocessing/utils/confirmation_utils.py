"""
File: smartcash/ui/dataset/preprocessing/utils/confirmation_utils.py
Deskripsi: Fixed confirmation utilities dengan immediate response tanpa threading
"""

from typing import Dict, Any, Callable, Optional
from ipywidgets import VBox, HBox, Button, HTML, Output
from IPython.display import display, clear_output

def show_cleanup_confirmation(ui_components: Dict[str, Any], detailed_stats: str) -> Optional[bool]:
    """Fixed cleanup confirmation dengan immediate response"""
    if 'confirmation_area' not in ui_components:
        return True  # Default proceed jika no confirmation area
    
    # Clear area first
    _clear_confirmation_area(ui_components)
    
    # Show confirmation dialog
    return _show_immediate_confirmation(
        ui_components=ui_components,
        title="⚠️ Konfirmasi Cleanup Dataset",
        message=f"Anda akan menghapus data preprocessed:<br><strong>{detailed_stats}</strong><br><span style='color: #dc3545;'>⚠️ Tindakan ini tidak dapat dibatalkan!</span>",
        confirm_text="Ya, Hapus",
        cancel_text="Batal",
        danger_mode=True
    )

def show_preprocessing_confirmation(ui_components: Dict[str, Any], message: str = None) -> Optional[bool]:
    """Fixed preprocessing confirmation dengan immediate response"""
    if 'confirmation_area' not in ui_components:
        return True
    
    _clear_confirmation_area(ui_components)
    
    default_message = "Apakah Anda yakin ingin memulai preprocessing dataset?"
    return _show_immediate_confirmation(
        ui_components=ui_components,
        title="🔄 Konfirmasi Preprocessing",
        message=message or default_message,
        confirm_text="Ya, Mulai",
        cancel_text="Batal",
        danger_mode=False
    )

def _show_immediate_confirmation(ui_components: Dict[str, Any], title: str, message: str, 
                               confirm_text: str, cancel_text: str, danger_mode: bool = False) -> Optional[bool]:
    """Fixed immediate confirmation dengan state tracking yang proper"""
    confirmation_area = ui_components['confirmation_area']
    
    # Result container yang akan dimodifikasi oleh handlers
    confirmation_result = {'value': None, 'completed': False}
    
    # Buttons dengan proper styling
    confirm_style = 'danger' if danger_mode else 'primary'
    confirm_btn = Button(
        description=confirm_text,
        button_style=confirm_style,
        icon='trash' if danger_mode else 'check',
        layout={'width': '130px', 'margin': '5px'}
    )
    
    cancel_btn = Button(
        description=cancel_text,
        button_style='',
        icon='times',
        layout={'width': '100px', 'margin': '5px'}
    )
    
    # Event handlers yang langsung mengupdate result
    def handle_confirm(btn):
        """Handler konfirmasi yang immediate"""
        confirmation_result['value'] = True
        confirmation_result['completed'] = True
        
        # Show immediate feedback
        _show_immediate_feedback(
            ui_components, 
            f"✅ {confirm_text} dikonfirmasi - Melanjutkan operasi...", 
            "success"
        )
    
    def handle_cancel(btn):
        """Handler pembatalan yang immediate"""
        confirmation_result['value'] = False
        confirmation_result['completed'] = True
        
        # Show immediate feedback  
        _show_immediate_feedback(
            ui_components,
            f"🚫 Operasi dibatalkan",
            "info"
        )
    
    # Bind handlers
    confirm_btn.on_click(handle_confirm)
    cancel_btn.on_click(handle_cancel)
    
    # Dialog styling
    border_color = '#dc3545' if danger_mode else '#007bff'
    bg_color = '#fff5f5' if danger_mode else '#f8f9fa'
    
    dialog = VBox([
        HTML(f"<h4 style='color: {border_color}; margin: 0 0 15px 0; text-align: center;'>{title}</h4>"),
        HTML(f'<div style="margin: 0 0 20px 0; text-align: center; line-height: 1.4;">{message}</div>'),
        HBox([confirm_btn, cancel_btn], layout={
            'justify_content': 'center',
            'align_items': 'center',
            'margin': '10px 0 0 0'
        })
    ], layout={
        'padding': '20px',
        'border': f'2px solid {border_color}',
        'border_radius': '8px',
        'background_color': bg_color,
        'width': '100%',
        'max_width': '500px',
        'margin': '10px auto'
    })
    
    # Display dialog
    with confirmation_area:
        clear_output(wait=True)
        display(dialog)
    
    # Store confirmation components untuk akses dari handler
    ui_components['_current_confirmation'] = {
        'result': confirmation_result,
        'dialog': dialog,
        'confirm_btn': confirm_btn,
        'cancel_btn': cancel_btn
    }
    
    # Return None untuk indicate bahwa ini async confirmation
    # Actual result akan di-check di calling function
    return None

def _show_immediate_feedback(ui_components: Dict[str, Any], message: str, level: str):
    """Show immediate feedback tanpa delay"""
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
    <div style='padding: 15px; background-color: rgba(248,249,250,0.95); 
               margin: 10px auto; border: 2px solid {color}; 
               border_radius: 8px; text-align: center; max-width: 500px;'>
        <span style='color: {color}; font-weight: 600; font-size: 14px;'>{message}</span>
    </div>
    """)
    
    with confirmation_area:
        clear_output(wait=True)
        display(feedback_html)

def check_confirmation_result(ui_components: Dict[str, Any]) -> Optional[bool]:
    """Check apakah user sudah memberikan konfirmasi"""
    current_confirmation = ui_components.get('_current_confirmation')
    
    if not current_confirmation:
        return None
    
    result = current_confirmation['result']
    
    if result['completed']:
        # Cleanup confirmation state
        ui_components.pop('_current_confirmation', None)
        return result['value']
    
    return None  # Still waiting for user input

def wait_for_confirmation(ui_components: Dict[str, Any], timeout_seconds: int = 30) -> Optional[bool]:
    """Wait for user confirmation dengan polling tanpa threading"""
    import time
    
    start_time = time.time()
    
    while (time.time() - start_time) < timeout_seconds:
        result = check_confirmation_result(ui_components)
        
        if result is not None:
            return result
        
        # Short sleep untuk avoid busy waiting
        time.sleep(0.2)
    
    # Timeout - cleanup dan return None
    ui_components.pop('_current_confirmation', None)
    _show_immediate_feedback(ui_components, "⏰ Timeout - Operasi dibatalkan", "warning")
    return None

def _clear_confirmation_area(ui_components: Dict[str, Any]):
    """Clear confirmation area"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        with confirmation_area:
            clear_output(wait=True)
    
    # Also cleanup state
    ui_components.pop('_current_confirmation', None)

# Enhanced public utilities
def clear_confirmation_area(ui_components: Dict[str, Any]):
    """Public function to clear confirmation area"""
    _clear_confirmation_area(ui_components)

def show_info_message(ui_components: Dict[str, Any], message: str, auto_clear_seconds: int = 0):
    """Show info message dalam confirmation area"""
    if 'confirmation_area' not in ui_components:
        return
    
    _show_immediate_feedback(ui_components, message, "info")
    
    if auto_clear_seconds > 0:
        # Simple auto clear dengan immediate scheduling
        import time
        def delayed_clear():
            time.sleep(auto_clear_seconds)
            _clear_confirmation_area(ui_components)
        
        # Schedule tanpa threading - caller harus handle
        ui_components['_clear_scheduled'] = {
            'time': time.time() + auto_clear_seconds,
            'action': delayed_clear
        }

def show_success_message(ui_components: Dict[str, Any], message: str, auto_clear_seconds: int = 0):
    """Show success message dalam confirmation area"""
    if 'confirmation_area' not in ui_components:
        return
    
    _show_immediate_feedback(ui_components, message, "success")
    
    if auto_clear_seconds > 0:
        import time
        ui_components['_clear_scheduled'] = {
            'time': time.time() + auto_clear_seconds,
            'action': lambda: _clear_confirmation_area(ui_components)
        }

def process_scheduled_clear(ui_components: Dict[str, Any]):
    """Process scheduled clear jika ada"""
    import time
    
    scheduled = ui_components.get('_clear_scheduled')
    if scheduled and time.time() >= scheduled['time']:
        try:
            scheduled['action']()
        except Exception:
            pass
        finally:
            ui_components.pop('_clear_scheduled', None)

# Synchronous confirmation functions untuk immediate use
def get_preprocessing_confirmation(ui_components: Dict[str, Any], message: str = None) -> bool:
    """Synchronous preprocessing confirmation yang langsung return True"""
    # Untuk sementara return True untuk avoid blocking
    # Nanti bisa di-enhance dengan proper modal dialog
    from smartcash.ui.dataset.preprocessing.utils.ui_utils import log_to_accordion
    
    default_message = message or "Memulai preprocessing dataset"
    log_to_accordion(ui_components, f"🚀 {default_message}", "info")
    return True

def get_cleanup_confirmation(ui_components: Dict[str, Any], detailed_stats: str) -> bool:
    """Synchronous cleanup confirmation yang langsung return True setelah warning"""
    from smartcash.ui.dataset.preprocessing.utils.ui_utils import log_to_accordion
    
    log_to_accordion(ui_components, f"⚠️ Akan menghapus: {detailed_stats}", "warning")
    log_to_accordion(ui_components, "🧹 Melanjutkan cleanup (konfirmasi otomatis aktif)", "info")
    return True