"""
File: smartcash/ui/dataset/augmentation/handlers/main_handler.py
Deskripsi: Main handler registry dengan reset capability untuk semua operations
"""

from typing import Dict, Any

def register_all_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Register semua handlers dengan reset capability"""
    handlers_map = {
        'augment_button': _create_augment_handler,
        'check_button': _create_check_handler,
        'cleanup_button': _create_cleanup_handler,
        'save_button': _create_save_handler,
        'reset_button': _create_reset_handler
    }
    
    registered = 0
    for button_key, handler_creator in handlers_map.items():
        button = ui_components.get(button_key)
        if button and hasattr(button, 'on_click'):
            # Clear existing handlers
            if hasattr(button, '_click_handlers'):
                button._click_handlers.callbacks.clear()
            
            # Register new handler
            button.on_click(handler_creator(ui_components))
            registered += 1
    
    ui_components['handlers_registered'] = registered
    return ui_components

def _create_augment_handler(ui_components: Dict[str, Any]):
    """Create augment handler dengan reset"""
    def handler(button):
        _reset_ui_state(ui_components)
        from .operation_handler import execute_augmentation
        execute_augmentation(ui_components)
    return handler

def _create_check_handler(ui_components: Dict[str, Any]):
    """Create check handler dengan reset"""
    def handler(button):
        _reset_ui_state(ui_components)
        from .operation_handler import execute_check
        execute_check(ui_components)
    return handler

def _create_cleanup_handler(ui_components: Dict[str, Any]):
    """Create cleanup handler dengan confirmation"""
    def handler(button):
        from .cleanup_handler import show_cleanup_confirmation
        show_cleanup_confirmation(ui_components)
    return handler

def _create_save_handler(ui_components: Dict[str, Any]):
    """Create save handler"""
    def handler(button):
        from .config_handler import save_configuration
        save_configuration(ui_components)
    return handler

def _create_reset_handler(ui_components: Dict[str, Any]):
    """Create reset handler"""
    def handler(button):
        from .config_handler import reset_configuration
        reset_configuration(ui_components)
    return handler

def _reset_ui_state(ui_components: Dict[str, Any]):
    """Reset progress dan log output sebelum operation"""
    # Reset progress tracker
    tracker = ui_components.get('tracker')
    tracker and hasattr(tracker, 'reset') and tracker.reset()
    
    # Clear log output
    log_output = ui_components.get('log_output')
    if log_output and hasattr(log_output, 'clear_output'):
        log_output.clear_output(wait=True)