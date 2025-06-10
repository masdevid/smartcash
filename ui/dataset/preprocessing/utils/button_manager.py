"""
File: smartcash/ui/dataset/preprocessing/utils/button_manager.py
Deskripsi: Enhanced button manager dengan comprehensive state management dan backend integration
"""

from typing import Dict, Any, Callable
import functools

def disable_all_buttons(ui_components: Dict[str, Any]):
    """Disable semua operation buttons dengan state preservation"""
    button_keys = ['preprocess_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
    
    for key in button_keys:
        button = ui_components.get(key)
        if button and hasattr(button, 'disabled'):
            # Preserve original state
            if not hasattr(button, '_original_disabled'):
                button._original_disabled = button.disabled
            button.disabled = True

def enable_all_buttons(ui_components: Dict[str, Any]):
    """Enable semua operation buttons dengan state restoration"""
    button_keys = ['preprocess_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
    
    for key in button_keys:
        button = ui_components.get(key)
        if button and hasattr(button, 'disabled'):
            # Restore original state atau enable
            original_state = getattr(button, '_original_disabled', False)
            button.disabled = original_state

def set_button_processing_state(ui_components: Dict[str, Any], button_key: str, processing: bool = True):
    """Set specific button ke processing state dengan visual feedback"""
    button = ui_components.get(button_key)
    if not button:
        return
    
    if processing:
        # Preserve original state
        if not hasattr(button, '_original_description'):
            button._original_description = button.description
            button._original_style = getattr(button, 'button_style', 'primary')
        
        # Set processing state
        button.description = f"{button._original_description} (Processing...)"
        button.button_style = 'warning'
        button.disabled = True
    else:
        # Restore original state
        if hasattr(button, '_original_description'):
            button.description = button._original_description
            button.button_style = button._original_style
            button.disabled = getattr(button, '_original_disabled', False)

def update_button_states_batch(ui_components: Dict[str, Any], button_states: Dict[str, Dict[str, Any]]):
    """Update multiple buttons dengan batch operation"""
    for btn_key, state in button_states.items():
        button = ui_components.get(btn_key)
        if button:
            for attr, value in state.items():
                if hasattr(button, attr):
                    setattr(button, attr, value)

def with_button_management(operation_func: Callable):
    """Enhanced decorator dengan comprehensive button management dan error handling"""
    @functools.wraps(operation_func)
    def wrapper(ui_components: Dict[str, Any], *args, **kwargs):
        try:
            # Disable all buttons at start
            disable_all_buttons(ui_components)
            
            # Set processing state for the active button
            button_key = operation_func.__name__.replace('_handler', '_button')
            set_button_processing_state(ui_components, button_key, True)
            
            # Execute the operation
            result = operation_func(ui_components, *args, **kwargs)
            return result
            
        except Exception as e:
            # Handle any errors
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"Error in {operation_func.__name__}: {str(e)}", exc_info=True)
            raise
            
        finally:
            # Re-enable buttons
            enable_all_buttons(ui_components)
            set_button_processing_state(ui_components, button_key, False)
    
    return wrapper

def create_button_state_manager(ui_components: Dict[str, Any]):
    """Factory untuk button state manager"""
    return ButtonStateManager(ui_components)

class ButtonStateManager:
    """Advanced button state manager dengan operation tracking"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.operation_stack = []
        self.button_states = {}
    
    def push_operation(self, operation_name: str, affected_buttons: list = None):
        """Push operation ke stack dengan button state preservation"""
        # Save current button states
        self.button_states[operation_name] = {}
        buttons = affected_buttons or ['preprocess_button', 'check_button', 'cleanup_button']
        
        for btn_key in buttons:
            button = self.ui_components.get(btn_key)
            if button and hasattr(button, 'disabled'):
                self.button_states[operation_name][btn_key] = {
                    'disabled': button.disabled,
                    'description': getattr(button, 'description', ''),
                    'button_style': getattr(button, 'button_style', '')
                }
                
        self.operation_stack.append(operation_name)
    
    def pop_operation(self):
        """Pop operation dari stack dan restore button states"""
        if not self.operation_stack:
            return
            
        operation_name = self.operation_stack.pop()
        if operation_name in self.button_states:
            for btn_key, state in self.button_states[operation_name].items():
                button = self.ui_components.get(btn_key)
                if button:
                    for attr, value in state.items():
                        if hasattr(button, attr):
                            setattr(button, attr, value)
            
            del self.button_states[operation_name]
    
    def get_current_operation(self):
        """Get current operation name"""
        return self.operation_stack[-1] if self.operation_stack else None
    
    def is_operation_active(self):
        """Check apakah ada operation yang sedang berjalan"""
        return len(self.operation_stack) > 0

def _notify_backend_operation_start(ui_components: Dict[str, Any], operation_name: str):
    """Notify backend tentang operation start"""
    backend = ui_components.get('backend')
    if hasattr(backend, 'notify_operation_start'):
        backend.notify_operation_start(operation_name)

def _notify_backend_operation_complete(ui_components: Dict[str, Any], operation_name: str, success: bool, error_msg: str = None):
    """Notify backend tentang operation completion"""
    backend = ui_components.get('backend')
    if hasattr(backend, 'notify_operation_complete'):
        backend.notify_operation_complete(operation_name, success, error_msg)

def _set_operation_indicator(ui_components: Dict[str, Any], operation_name: str, active: bool):
    """Set visual operation indicator"""
    indicator = ui_components.get('operation_indicator')
    if indicator:
        if active:
            indicator.value = f"{operation_name} in progress..."
            indicator.bar_style = 'info'
        else:
            indicator.value = ""
            indicator.bar_style = ''

def _handle_operation_error(ui_components: Dict[str, Any], operation_name: str, error: Exception):
    """Handle operation error dengan logging"""
    logger = ui_components.get('logger')
    if logger:
        logger.error(f"Error in {operation_name}: {str(error)}", exc_info=True)

# Convenience functions
disable_buttons = lambda ui_components: disable_all_buttons(ui_components)
enable_buttons = lambda ui_components: enable_all_buttons(ui_components)
set_processing = lambda ui_components, button_key: set_button_processing_state(ui_components, button_key, True)
reset_processing = lambda ui_components, button_key: set_button_processing_state(ui_components, button_key, False)

# Backward compatibility
get_button_manager = create_button_state_manager