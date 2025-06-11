"""
File: smartcash/ui/dataset/preprocessing/utils/button_manager.py
Deskripsi: Simplified button manager tanpa complexity yang berlebihan
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger

def disable_operation_buttons(ui_components: Dict[str, Any]):
    """Disable operation buttons saat processing"""
    operation_buttons = ['preprocess_button', 'check_button', 'cleanup_button']
    
    for btn_key in operation_buttons:
        button = ui_components.get(btn_key)
        if button and hasattr(button, 'disabled'):
            # Store original state
            if not hasattr(button, '_original_disabled'):
                button._original_disabled = button.disabled
            button.disabled = True

def enable_operation_buttons(ui_components: Dict[str, Any]):
    """Enable operation buttons setelah processing"""
    operation_buttons = ['preprocess_button', 'check_button', 'cleanup_button']
    
    for btn_key in operation_buttons:
        button = ui_components.get(btn_key)
        if button and hasattr(button, 'disabled'):
            # Restore original state
            original_state = getattr(button, '_original_disabled', False)
            button.disabled = original_state

def set_button_processing_state(ui_components: Dict[str, Any], button_key: str, processing: bool = True):
    """Set individual button processing state"""
    button = ui_components.get(button_key)
    if not button:
        return
    
    if processing:
        # Store original state
        if not hasattr(button, '_original_description'):
            button._original_description = button.description
            button._original_style = getattr(button, 'button_style', 'primary')
        
        # Set processing state
        button.description = f"{button._original_description} ..."
        button.button_style = 'warning'
        button.disabled = True
    else:
        # Restore original state
        if hasattr(button, '_original_description'):
            button.description = button._original_description
            button.button_style = button._original_style
            delattr(button, '_original_description')
            delattr(button, '_original_style')
        
        button.disabled = False

def with_button_management(operation_name: str):
    """Decorator untuk automatic button management"""
    def decorator(operation_func):
        def wrapper(ui_components: Dict[str, Any], *args, **kwargs):
            try:
                # Disable buttons
                disable_operation_buttons(ui_components)
                
                # Set processing state for primary button
                button_map = {
                    'preprocessing': 'preprocess_button',
                    'validation': 'check_button',
                    'cleanup': 'cleanup_button'
                }
                
                primary_button = button_map.get(operation_name.lower())
                if primary_button:
                    set_button_processing_state(ui_components, primary_button, True)
                
                # Execute operation
                result = operation_func(ui_components, *args, **kwargs)
                
                return result
                
            except Exception as e:
                get_logger('button_manager').error(f"Error in {operation_name}: {str(e)}")
                raise
            finally:
                # Always restore buttons
                enable_operation_buttons(ui_components)
                if primary_button:
                    set_button_processing_state(ui_components, primary_button, False)
        
        return wrapper
    return decorator

# One-liner utilities
disable_all_buttons = lambda ui_components: disable_operation_buttons(ui_components)
enable_all_buttons = lambda ui_components: enable_operation_buttons(ui_components)
set_processing = lambda ui_components, btn_key: set_button_processing_state(ui_components, btn_key, True)
clear_processing = lambda ui_components, btn_key: set_button_processing_state(ui_components, btn_key, False)