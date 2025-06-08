"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handlers.py
Deskripsi: Fixed main handlers dengan dialog integration dan config handler yang diperbaiki
"""

from typing import Dict, Any

def setup_augmentation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handlers dengan config handler UI integration dan dialog support"""
    
    # CRITICAL: Setup config handler dengan UI logger
    config_handler = ui_components.get('config_handler')
    if config_handler and hasattr(config_handler, 'set_ui_components'):
        config_handler.set_ui_components(ui_components)
    
    # Setup operation handlers
    _setup_operation_handlers(ui_components, config, env)
    
    # Setup config handlers dengan proper UI logging
    _setup_config_handlers_fixed(ui_components, config)
    
    return ui_components

def _setup_operation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None):
    """Setup operation handlers dengan progress tracking"""
    
    def augment_handler(button):
        """Augmentation handler dengan progress tracking"""
        _clear_outputs(ui_components)
        from smartcash.ui.dataset.augmentation.utils.operation_utils import execute_augmentation
        execute_augmentation(ui_components)
    
    def check_handler(button):
        """Check dataset handler dengan detailed analysis"""
        _clear_outputs(ui_components)
        from smartcash.ui.dataset.augmentation.utils.operation_utils import execute_check
        execute_check(ui_components)
    
    def cleanup_handler(button):
        """Cleanup handler dengan dialog confirmation"""
        _clear_outputs(ui_components)
        _show_cleanup_confirmation_dialog(ui_components)
    
    # Bind operation handlers
    operation_buttons = {
        'augment_button': augment_handler,
        'check_button': check_handler,
        'cleanup_button': cleanup_handler
    }
    
    for button_name, handler in operation_buttons.items():
        button = ui_components.get(button_name)
        if button and hasattr(button, 'on_click'):
            if hasattr(button, '_click_handlers'):
                button._click_handlers.callbacks.clear()
            button.on_click(handler)

def _setup_config_handlers_fixed(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """CRITICAL: Config handlers dengan proper UI logging"""
    
    def save_config(button=None):
        _clear_outputs(ui_components)
        
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                _handle_ui_error(ui_components, "❌ Config handler tidak tersedia")
                return
            
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            config_handler.save_config(ui_components)
            
        except Exception as e:
            _handle_ui_error(ui_components, f"❌ Error save: {str(e)}")
    
    def reset_config(button=None):
        _clear_outputs(ui_components)
        
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                _handle_ui_error(ui_components, "❌ Config handler tidak tersedia")
                return
            
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            config_handler.reset_config(ui_components)
            
        except Exception as e:
            _handle_ui_error(ui_components, f"❌ Error reset: {str(e)}")
    
    # Bind config handlers
    config_buttons = {'save_button': save_config, 'reset_button': reset_config}
    
    for button_name, handler in config_buttons.items():
        button = ui_components.get(button_name)
        if button and hasattr(button, 'on_click'):
            if hasattr(button, '_click_handlers'):
                button._click_handlers.callbacks.clear()
            button.on_click(handler)

def _show_cleanup_confirmation_dialog(ui_components: Dict[str, Any]):
    """Show cleanup confirmation menggunakan existing dialog_utils"""
    from smartcash.ui.dataset.augmentation.utils.dialog_utils import show_cleanup_confirmation
    from smartcash.ui.dataset.augmentation.utils.operation_utils import execute_cleanup_with_progress
    
    def confirm_cleanup(button):
        execute_cleanup_with_progress(ui_components)
    
    show_cleanup_confirmation(ui_components, confirm_cleanup)

def _clear_outputs(ui_components: Dict[str, Any]):
    """Clear output areas dan reset progress"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker and hasattr(progress_tracker, 'reset'):
        progress_tracker.reset()
    elif 'reset_all' in ui_components:
        ui_components['reset_all']()
    
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area and hasattr(confirmation_area, 'clear_output'):
        confirmation_area.clear_output(wait=True)

def _handle_ui_error(ui_components: Dict[str, Any], message: str):
    """Handle UI error dengan fallback logging"""
    _log_to_ui(ui_components, message, 'error')

def _log_to_ui(ui_components: Dict[str, Any], message: str, level: str = 'info'):
    """Log ke UI dengan fallback chain"""
    try:
        logger = ui_components.get('logger')
        if logger and hasattr(logger, level):
            getattr(logger, level)(message)
            return
        
        widget = ui_components.get('log_output') or ui_components.get('status')
        if widget and hasattr(widget, 'clear_output'):
            from IPython.display import display, HTML
            color_map = {'info': '#007bff', 'success': '#28a745', 'warning': '#ffc107', 'error': '#dc3545'}
            color = color_map.get(level, '#007bff')
            html = f'<div style="color: {color}; margin: 2px 0; padding: 4px;">{message}</div>'
            
            with widget:
                display(HTML(html))
            return
            
    except Exception:
        pass
    
    print(message)