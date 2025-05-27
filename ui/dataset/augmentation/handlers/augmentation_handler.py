"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handler.py
Deskripsi: Fixed main handler dengan SRP delegation dan robust button_state_manager
"""

from typing import Dict, Any

# SRP handlers import
from .button_handler import create_button_handler
from .operation_handlers import (
    create_augmentation_handler, create_check_handler, 
    create_cleanup_handler, create_config_handler_ops
)
from .confirmation_handler import create_confirmation_handler

def handle_augmentation_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler augmentation dengan SRP delegation"""
    button_handler = create_button_handler(ui_components)
    operation_handler = create_augmentation_handler(ui_components)
    
    button_handler.execute_operation('augmentation', operation_handler.execute)

def handle_check_dataset_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler check dataset dengan SRP delegation"""
    button_handler = create_button_handler(ui_components)
    operation_handler = create_check_handler(ui_components)
    
    button_handler.execute_operation('check', operation_handler.execute)

def handle_cleanup_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler cleanup dengan confirmation delegation"""
    confirmation_handler = create_confirmation_handler(ui_components)
    
    def confirm_cleanup(confirm_button):
        button_handler = create_button_handler(ui_components)
        operation_handler = create_cleanup_handler(ui_components)
        button_handler.execute_operation('cleanup', operation_handler.execute)
    
    confirmation_handler.show_cleanup_confirmation(confirm_cleanup)

def handle_save_config_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler save config dengan SRP delegation"""
    button_handler = create_button_handler(ui_components)
    config_handler = create_config_handler_ops(ui_components)
    
    button_handler.execute_config('save', config_handler.save_config)

def handle_reset_config_button_click(ui_components: Dict[str, Any], button: Any):
    """Handler reset config dengan SRP delegation"""
    button_handler = create_button_handler(ui_components)
    config_handler = create_config_handler_ops(ui_components)
    
    button_handler.execute_config('reset', config_handler.reset_config)

def register_augmentation_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Register handlers dengan safe button_state_manager initialization"""
    try:
        # Ensure button_state_manager exists
        button_handler = create_button_handler(ui_components)
        ui_components['button_state_manager'] = button_handler.manager
        
        # Handler mapping
        button_handlers = {
            'augment_button': handle_augmentation_button_click,
            'check_button': handle_check_dataset_button_click,
            'cleanup_button': handle_cleanup_button_click,
            'save_button': handle_save_config_button_click,
            'reset_button': handle_reset_config_button_click
        }
        
        # Register handlers dengan safe approach
        registered_count = 0
        for button_key, handler_func in button_handlers.items():
            try:
                button = ui_components.get(button_key)
                if button and hasattr(button, 'on_click'):
                    # Clear existing handlers
                    if hasattr(button, '_click_handlers'):
                        button._click_handlers.callbacks.clear()
                    
                    # Register new handler
                    button.on_click(lambda b, h=handler_func, ui=ui_components: h(ui, b))
                    registered_count += 1
            except Exception:
                continue
        
        # Update registration info
        ui_components['registered_handlers'] = {
            'total': registered_count,
            'handlers': list(button_handlers.keys()),
            'status': 'success' if registered_count > 0 else 'partial',
            'button_state_manager_active': True
        }
        
        # Log registration success
        if 'logger' in ui_components:
            ui_components['logger'].info(f"✅ {registered_count} handlers berhasil didaftarkan dengan button_state_manager")
        
        return ui_components
        
    except Exception as e:
        # Fallback registration info
        ui_components['registered_handlers'] = {
            'total': 0,
            'handlers': [],
            'status': 'error',
            'error': str(e),
            'button_state_manager_active': False
        }
        
        if 'logger' in ui_components:
            ui_components['logger'].error(f"❌ Handler registration error: {str(e)}")
        
        return ui_components