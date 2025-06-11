"""
File: smartcash/ui/dataset/preprocessing/utils/confirmation_utils.py
Deskripsi: Working confirmation system dengan dialog yang benar-benar muncul dan state management
"""

from typing import Dict, Any, Callable, Optional
from ipywidgets import VBox, HBox, Button, HTML, Output
from IPython.display import display, clear_output

def show_cleanup_confirmation(ui_components: Dict[str, Any], detailed_stats: str):
    """Show cleanup confirmation dialog yang benar-benar muncul"""
    if 'confirmation_area' not in ui_components:
        return
    
    _clear_confirmation_area(ui_components)
    
    # Create confirmation dialog
    _create_confirmation_dialog(
        ui_components=ui_components,
        title="⚠️ Konfirmasi Cleanup Dataset",
        message=f"Anda akan menghapus data preprocessed:<br><strong>{detailed_stats}</strong><br><span style='color: #dc3545;'>⚠️ Tindakan ini tidak dapat dibatalkan!</span>",
        confirm_text="Ya, Hapus",
        cancel_text="Batal",
        danger_mode=True,
        operation_type='cleanup'
    )

def show_preprocessing_confirmation(ui_components: Dict[str, Any], message: str = None):
    """Show preprocessing confirmation dialog yang benar-benar muncul"""
    if 'confirmation_area' not in ui_components:
        return
    
    _clear_confirmation_area(ui_components)
    
    default_message = message or "Apakah Anda yakin ingin memulai preprocessing dataset?"
    _create_confirmation_dialog(
        ui_components=ui_components,
        title="🔄 Konfirmasi Preprocessing",
        message=default_message,
        confirm_text="Ya, Mulai",
        cancel_text="Batal",
        danger_mode=False,
        operation_type='preprocessing'
    )

def _create_confirmation_dialog(ui_components: Dict[str, Any], title: str, message: str, 
                              confirm_text: str, cancel_text: str, danger_mode: bool = False,
                              operation_type: str = 'operation'):
    """Create confirmation dialog yang benar-benar muncul dengan proper handlers"""
    confirmation_area = ui_components['confirmation_area']
    
    # Initialize confirmation state
    ui_components['_confirmation_state'] = {
        'waiting': True,
        'result': None,
        'operation_type': operation_type
    }
    
    # Create buttons
    confirm_style = 'danger' if danger_mode else 'primary'
    confirm_btn = Button(
        description=confirm_text,
        button_style=confirm_style,
        icon='trash' if danger_mode else 'check',
        layout={'width': '140px', 'height': '35px', 'margin': '5px'}
    )
    
    cancel_btn = Button(
        description=cancel_text,
        button_style='',
        icon='times',
        layout={'width': '100px', 'height': '35px', 'margin': '5px'}
    )
    
    # Event handlers
    def on_confirm(btn):
        """Handle confirmation"""
        ui_components['_confirmation_state'] = {
            'waiting': False,
            'result': True,
            'operation_type': operation_type
        }
        
        # Show feedback dan execute operation
        _show_confirmation_feedback(ui_components, f"✅ {confirm_text} - Memulai operasi...", "success")
        _execute_confirmed_operation(ui_components, operation_type)
    
    def on_cancel(btn):
        """Handle cancellation"""
        ui_components['_confirmation_state'] = {
            'waiting': False,
            'result': False,
            'operation_type': operation_type
        }
        
        # Show feedback dan cleanup
        _show_confirmation_feedback(ui_components, f"🚫 Operasi dibatalkan", "info")
        _handle_operation_cancelled(ui_components, operation_type)
    
    # Bind handlers
    confirm_btn.on_click(on_confirm)
    cancel_btn.on_click(on_cancel)
    
    # Dialog styling
    border_color = '#dc3545' if danger_mode else '#007bff'
    bg_color = '#fff5f5' if danger_mode else '#f8f9fa'
    
    dialog = VBox([
        HTML(f"""
        <div style='text-align: center; margin-bottom: 15px;'>
            <h3 style='color: {border_color}; margin: 0; font-size: 18px;'>{title}</h3>
        </div>
        """),
        HTML(f"""
        <div style='margin-bottom: 20px; text-align: center; line-height: 1.5; font-size: 14px;'>
            {message}
        </div>
        """),
        HBox([confirm_btn, cancel_btn], layout={
            'justify_content': 'center',
            'align_items': 'center',
            'margin': '10px 0 0 0'
        })
    ], layout={
        'padding': '25px',
        'border': f'3px solid {border_color}',
        'border_radius': '10px',
        'background_color': bg_color,
        'width': '100%',
        'max_width': '550px',
        'margin': '15px auto',
        'box_shadow': '0 4px 8px rgba(0,0,0,0.1)'
    })
    
    # Display dialog
    with confirmation_area:
        clear_output(wait=True)
        display(dialog)

def _show_confirmation_feedback(ui_components: Dict[str, Any], message: str, level: str):
    """Show confirmation feedback message"""
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
    <div style='padding: 20px; background-color: rgba(248,249,250,0.95); 
               margin: 15px auto; border: 2px solid {color}; 
               border_radius: 8px; text-align: center; max-width: 500px;
               box_shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <span style='color: {color}; font-weight: 600; font-size: 15px;'>{message}</span>
    </div>
    """)
    
    with confirmation_area:
        clear_output(wait=True)
        display(feedback_html)

def _execute_confirmed_operation(ui_components: Dict[str, Any], operation_type: str):
    """Execute operation setelah konfirmasi diterima"""
    from smartcash.ui.dataset.preprocessing.utils.ui_utils import log_to_accordion
    
    if operation_type == 'preprocessing':
        log_to_accordion(ui_components, "🚀 Konfirmasi diterima - Memulai preprocessing...", "success")
        _trigger_preprocessing_execution(ui_components)
    elif operation_type == 'cleanup':
        log_to_accordion(ui_components, "🗑️ Konfirmasi diterima - Memulai cleanup...", "success")
        _trigger_cleanup_execution(ui_components)

def _handle_operation_cancelled(ui_components: Dict[str, Any], operation_type: str):
    """Handle operation cancellation"""
    from smartcash.ui.dataset.preprocessing.utils.ui_utils import log_to_accordion
    from smartcash.ui.dataset.preprocessing.utils.button_manager import enable_all_buttons
    
    log_to_accordion(ui_components, f"🚫 {operation_type.capitalize()} dibatalkan oleh user", "info")
    
    # Re-enable buttons
    enable_all_buttons(ui_components)
    
    # Clear confirmation area setelah delay
    import threading
    def delayed_clear():
        import time
        time.sleep(2)
        _clear_confirmation_area(ui_components)
    
    threading.Thread(target=delayed_clear, daemon=True).start()

def _trigger_preprocessing_execution(ui_components: Dict[str, Any]):
    """Trigger actual preprocessing execution"""
    # Set flag untuk indicate processing should continue
    ui_components['_should_execute_preprocessing'] = True
    
    # Clear confirmation area
    _clear_confirmation_area(ui_components)

def _trigger_cleanup_execution(ui_components: Dict[str, Any]):
    """Trigger actual cleanup execution"""
    # Set flag untuk indicate cleanup should continue
    ui_components['_should_execute_cleanup'] = True
    
    # Clear confirmation area
    _clear_confirmation_area(ui_components)

def _clear_confirmation_area(ui_components: Dict[str, Any]):
    """Clear confirmation area"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        with confirmation_area:
            clear_output(wait=True)
    
    # Clear confirmation state
    ui_components.pop('_confirmation_state', None)

def is_confirmation_pending(ui_components: Dict[str, Any]) -> bool:
    """Check jika sedang menunggu konfirmasi"""
    state = ui_components.get('_confirmation_state', {})
    return state.get('waiting', False)

def should_execute_operation(ui_components: Dict[str, Any], operation_type: str) -> bool:
    """Check jika operation sudah dikonfirmasi dan should execute"""
    if operation_type == 'preprocessing':
        return ui_components.pop('_should_execute_preprocessing', False)
    elif operation_type == 'cleanup':
        return ui_components.pop('_should_execute_cleanup', False)
    return False

def clear_confirmation_area(ui_components: Dict[str, Any]):
    """Public function to clear confirmation area"""
    _clear_confirmation_area(ui_components)

def show_info_message(ui_components: Dict[str, Any], message: str):
    """Show info message dalam confirmation area"""
    if 'confirmation_area' not in ui_components:
        return
    
    _show_confirmation_feedback(ui_components, message, "info")

def show_success_message(ui_components: Dict[str, Any], message: str):
    """Show success message dalam confirmation area"""
    if 'confirmation_area' not in ui_components:
        return
    
    _show_confirmation_feedback(ui_components, message, "success")