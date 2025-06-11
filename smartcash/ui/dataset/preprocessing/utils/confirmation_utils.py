"""
File: smartcash/ui/dataset/preprocessing/utils/confirmation_utils.py
Deskripsi: FIXED confirmation system dengan proper dialog clearing dan execution
"""

from typing import Dict, Any, Callable, Optional
from ipywidgets import VBox, HBox, Button, HTML, Output
from IPython.display import display, clear_output

def show_cleanup_confirmation(ui_components: Dict[str, Any], detailed_stats: str):
    """Show cleanup confirmation dialog"""
    if 'confirmation_area' not in ui_components:
        return
    
    _clear_confirmation_area(ui_components)
    
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
    """Show preprocessing confirmation dialog"""
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
    """Create confirmation dialog dengan proper handlers"""
    confirmation_area = ui_components['confirmation_area']
    
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
    
    # FIXED: Event handlers yang clear dialog dan execute operations
    def on_confirm(btn):
        """Handle confirmation - FIXED"""
        _clear_confirmation_area(ui_components)
        _execute_confirmed_operation(ui_components, operation_type)
    
    def on_cancel(btn):
        """Handle cancellation - FIXED"""
        _clear_confirmation_area(ui_components)  
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

def _execute_confirmed_operation(ui_components: Dict[str, Any], operation_type: str):
    """FIXED: Execute operation setelah konfirmasi dengan immediate execution"""
    from smartcash.ui.dataset.preprocessing.utils.ui_utils import log_to_accordion
    from smartcash.ui.dataset.preprocessing.utils.button_manager import disable_all_buttons
    
    # Disable buttons immediately
    disable_all_buttons(ui_components)
    
    if operation_type == 'preprocessing':
        log_to_accordion(ui_components, "🚀 Memulai preprocessing...", "success")
        _trigger_actual_preprocessing(ui_components)
    elif operation_type == 'cleanup':
        log_to_accordion(ui_components, "🗑️ Memulai cleanup...", "success")
        _trigger_actual_cleanup(ui_components)

def _trigger_actual_preprocessing(ui_components: Dict[str, Any]):
    """Execute actual preprocessing operation"""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handlers import _execute_preprocessing_with_api
        from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
        
        config = extract_preprocessing_config(ui_components)
        _execute_preprocessing_with_api(ui_components, config)
        
    except Exception as e:
        from smartcash.ui.dataset.preprocessing.utils.ui_utils import handle_ui_error
        handle_ui_error(ui_components, f"Error executing preprocessing: {str(e)}")

def _trigger_actual_cleanup(ui_components: Dict[str, Any]):
    """Execute actual cleanup operation"""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handlers import _execute_cleanup_with_api
        from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
        
        config = extract_preprocessing_config(ui_components)
        _execute_cleanup_with_api(ui_components, config)
        
    except Exception as e:
        from smartcash.ui.dataset.preprocessing.utils.ui_utils import handle_ui_error
        handle_ui_error(ui_components, f"Error executing cleanup: {str(e)}")

def _handle_operation_cancelled(ui_components: Dict[str, Any], operation_type: str):
    """Handle operation cancellation"""
    from smartcash.ui.dataset.preprocessing.utils.ui_utils import log_to_accordion
    from smartcash.ui.dataset.preprocessing.utils.button_manager import enable_all_buttons
    
    log_to_accordion(ui_components, f"🚫 {operation_type.capitalize()} dibatalkan oleh user", "info")
    enable_all_buttons(ui_components)

def _clear_confirmation_area(ui_components: Dict[str, Any]):
    """Clear confirmation area"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        with confirmation_area:
            clear_output(wait=True)
    
    # Clear any state flags
    ui_components.pop('_confirmation_state', None)
    ui_components.pop('_should_execute_preprocessing', None)
    ui_components.pop('_should_execute_cleanup', None)

# Public utility functions
def clear_confirmation_area(ui_components: Dict[str, Any]):
    """Public function to clear confirmation area"""
    _clear_confirmation_area(ui_components)

def show_info_message(ui_components: Dict[str, Any], message: str):
    """Show info message dalam confirmation area"""
    if 'confirmation_area' not in ui_components:
        return
    
    info_html = HTML(f"""
    <div style='padding: 20px; background-color: #d1ecf1; 
               margin: 15px auto; border: 2px solid #17a2b8; 
               border-radius: 8px; text-align: center; max-width: 500px;'>
        <span style='color: #0c5460; font-weight: 600; font-size: 15px;'>ℹ️ {message}</span>
    </div>
    """)
    
    with ui_components['confirmation_area']:
        clear_output(wait=True)
        display(info_html)

def show_success_message(ui_components: Dict[str, Any], message: str):
    """Show success message dalam confirmation area"""
    if 'confirmation_area' not in ui_components:
        return
        
    success_html = HTML(f"""
    <div style='padding: 20px; background-color: #d4edda; 
               margin: 15px auto; border: 2px solid #28a745; 
               border-radius: 8px; text-align: center; max-width: 500px;'>
        <span style='color: #155724; font-weight: 600; font-size: 15px;'>✅ {message}</span>
    </div>
    """)
    
    with ui_components['confirmation_area']:
        clear_output(wait=True)
        display(success_html)