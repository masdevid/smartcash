"""
File: smartcash/ui/dataset/augmentation/handlers/main_handler.py
Deskripsi: Main handler dengan service integration dan consolidated operations
"""

from typing import Dict, Any

def register_all_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Register handlers dengan service integration"""
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
            if hasattr(button, '_click_handlers'):
                button._click_handlers.callbacks.clear()
            
            button.on_click(handler_creator(ui_components))
            registered += 1
    
    ui_components['handlers_registered'] = registered
    ui_components['service_integrated'] = True
    return ui_components

def _create_augment_handler(ui_components: Dict[str, Any]):
    """Create augment handler dengan service reuse"""
    def handler(button):
        _reset_ui_state(ui_components)
        from smartcash.ui.dataset.augmentation.handlers.operation_handlers import execute_augmentation
        execute_augmentation(ui_components)
    return handler

def _create_check_handler(ui_components: Dict[str, Any]):
    """Create check handler dengan service status"""
    def handler(button):
        _reset_ui_state(ui_components)
        from smartcash.ui.dataset.augmentation.handlers.operation_handlers import execute_check
        execute_check(ui_components)
    return handler

def _create_cleanup_handler(ui_components: Dict[str, Any]):
    """Create cleanup handler dengan service cleanup"""
    def handler(button):
        _reset_ui_state(ui_components)
        from smartcash.ui.dataset.augmentation.handlers.cleanup_handler import show_cleanup_confirmation
        show_cleanup_confirmation(ui_components)
    return handler

def _create_save_handler(ui_components: Dict[str, Any]):
    """Create save handler dengan config management"""
    def handler(button):
        _reset_ui_state(ui_components)
        from smartcash.ui.dataset.augmentation.handlers.config_handler import save_configuration
        save_configuration(ui_components)
    return handler

def _create_reset_handler(ui_components: Dict[str, Any]):
    """Create reset handler dengan config reset"""
    def handler(button):
        _reset_ui_state(ui_components)
        from smartcash.ui.dataset.augmentation.handlers.config_handler import reset_configuration
        reset_configuration(ui_components)
    return handler

def _reset_ui_state(ui_components: Dict[str, Any]):
    """Reset progress dan log sebelum operation"""
    tracker = ui_components.get('tracker')
    tracker and hasattr(tracker, 'reset') and tracker.reset()
    
    log_output = ui_components.get('log_output')
    if log_output and hasattr(log_output, 'clear_output'):
        log_output.clear_output(wait=True)