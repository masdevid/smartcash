"""
File: smartcash/ui/dataset/augmentation/utils/button_manager.py
Deskripsi: Enhanced button manager dengan preview button support dan comprehensive state management
"""

from typing import Dict, Any, Callable
import functools

def disable_all_buttons(ui_components: Dict[str, Any]):
    """Disable semua operation buttons dengan state preservation"""
    button_keys = ['augment_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button', 'generate_button']
    
    for key in button_keys:
        button = ui_components.get(key)
        if button and hasattr(button, 'disabled'):
            # Preserve original state
            if not hasattr(button, '_original_disabled'):
                button._original_disabled = button.disabled
            button.disabled = True

def enable_all_buttons(ui_components: Dict[str, Any]):
    """Enable semua operation buttons dengan state restoration"""
    button_keys = ['augment_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button', 'generate_button']
    
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
        
        # Set processing state dengan context-aware text
        processing_text = _get_processing_text(button_key)
        button.description = f"{button._original_description} {processing_text}"
        button.button_style = 'warning'
        button.disabled = True
    else:
        # Restore original state
        if hasattr(button, '_original_description'):
            button.description = button._original_description
            button.button_style = button._original_style
        button.disabled = False

def _get_processing_text(button_key: str) -> str:
    """Get context-aware processing text untuk berbagai button types"""
    processing_texts = {
        'augment_button': '🔄',
        'check_button': '🔍',
        'cleanup_button': '🧹',
        'save_button': '💾',
        'reset_button': '🔄',
        'generate_button': '🎯'
    }
    return processing_texts.get(button_key, '🔄')

def update_button_states_batch(ui_components: Dict[str, Any], button_states: Dict[str, Dict[str, Any]]):
    """Update multiple buttons dengan batch operation"""
    for button_key, state_config in button_states.items():
        button = ui_components.get(button_key)
        if not button:
            continue
        
        # Apply state changes
        for property_name, value in state_config.items():
            if hasattr(button, property_name):
                try:
                    setattr(button, property_name, value)
                except Exception:
                    pass  # Silent fail untuk read-only properties

def with_button_management(operation_func: Callable) -> Callable:
    """Enhanced decorator dengan comprehensive button management dan preview support"""
    @functools.wraps(operation_func)
    def wrapper(ui_components: Dict[str, Any], *args, **kwargs):
        # Notify backend tentang operation start
        _notify_backend_operation_start(ui_components, operation_func.__name__)
        
        try:
            # Disable buttons dengan state preservation
            disable_all_buttons(ui_components)
            
            # Set progress indicator jika ada
            _set_operation_indicator(ui_components, operation_func.__name__, True)
            
            # Execute operation
            result = operation_func(ui_components, *args, **kwargs)
            
            # Notify backend tentang success
            _notify_backend_operation_complete(ui_components, operation_func.__name__, True)
            
            return result
            
        except Exception as e:
            # Log error menggunakan backend communicator
            _handle_operation_error(ui_components, operation_func.__name__, e)
            
            # Notify backend tentang error
            _notify_backend_operation_complete(ui_components, operation_func.__name__, False, str(e))
            
            raise  # Re-raise untuk proper error handling
            
        finally:
            # Always restore button states
            enable_all_buttons(ui_components)
            
            # Clear operation indicator
            _set_operation_indicator(ui_components, operation_func.__name__, False)
    
    return wrapper

def create_button_state_manager(ui_components: Dict[str, Any]) -> 'ButtonStateManager':
    """Factory untuk button state manager dengan preview support"""
    return ButtonStateManager(ui_components)

class ButtonStateManager:
    """Advanced button state manager dengan operation tracking dan preview support"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.operation_stack = []
        self.button_states = {}
    
    def push_operation(self, operation_name: str, affected_buttons: list = None):
        """Push operation ke stack dengan button state preservation"""
        if affected_buttons is None:
            affected_buttons = ['augment_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button', 'generate_button']
        
        # Save current states
        current_states = {}
        for button_key in affected_buttons:
            button = self.ui_components.get(button_key)
            if button:
                current_states[button_key] = {
                    'disabled': getattr(button, 'disabled', False),
                    'description': getattr(button, 'description', ''),
                    'button_style': getattr(button, 'button_style', 'primary')
                }
        
        # Push ke stack
        self.operation_stack.append({
            'operation': operation_name,
            'buttons': affected_buttons,
            'states': current_states
        })
        
        # Disable affected buttons
        for button_key in affected_buttons:
            button = self.ui_components.get(button_key)
            if button:
                button.disabled = True
    
    def pop_operation(self):
        """Pop operation dari stack dan restore button states"""
        if not self.operation_stack:
            return
        
        operation_info = self.operation_stack.pop()
        
        # Restore button states
        for button_key, state in operation_info['states'].items():
            button = self.ui_components.get(button_key)
            if button:
                for property_name, value in state.items():
                    if hasattr(button, property_name):
                        try:
                            setattr(button, property_name, value)
                        except Exception:
                            pass
    
    def get_current_operation(self) -> str:
        """Get current operation name"""
        return self.operation_stack[-1]['operation'] if self.operation_stack else None
    
    def is_operation_active(self) -> bool:
        """Check apakah ada operation yang sedang berjalan"""
        return len(self.operation_stack) > 0
    
    def set_preview_generating(self, generating: bool = True):
        """Set preview generation state"""
        generate_button = self.ui_components.get('generate_button')
        if generate_button:
            if generating:
                set_button_processing_state(self.ui_components, 'generate_button', True)
                # Update preview status
                from smartcash.ui.dataset.augmentation.utils.ui_utils import update_preview_status
                update_preview_status(self.ui_components, 'generating', 'Generating preview...')
            else:
                set_button_processing_state(self.ui_components, 'generate_button', False)

def _notify_backend_operation_start(ui_components: Dict[str, Any], operation_name: str):
    """Notify backend tentang operation start"""
    try:
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, f"🚀 Starting {operation_name}", "info")
    except Exception:
        pass  # Silent fail

def _notify_backend_operation_complete(ui_components: Dict[str, Any], operation_name: str, success: bool, error_msg: str = None):
    """Notify backend tentang operation completion"""
    try:
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        if success:
            log_to_ui(ui_components, f"✅ {operation_name} completed successfully", "success")
        else:
            log_to_ui(ui_components, f"❌ {operation_name} failed: {error_msg or 'Operation failed'}", "error")
    except Exception:
        pass  # Silent fail

def _set_operation_indicator(ui_components: Dict[str, Any], operation_name: str, active: bool):
    """Set visual operation indicator"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and active:
            progress_tracker.show()
        elif progress_tracker and not active and not _has_other_active_operations(ui_components):
            # Only hide jika tidak ada operations lain yang active
            pass  # Let individual operations handle hiding
    except Exception:
        pass  # Silent fail

def _handle_operation_error(ui_components: Dict[str, Any], operation_name: str, error: Exception):
    """Handle operation error dengan logging"""
    try:
        from smartcash.ui.dataset.augmentation.utils.ui_utils import log_to_ui
        log_to_ui(ui_components, f"❌ {operation_name} error: {str(error)}", 'error')
    except Exception:
        print(f"❌ {operation_name} error: {str(error)}")

def _has_other_active_operations(ui_components: Dict[str, Any]) -> bool:
    """Check apakah ada operations lain yang masih active"""
    try:
        state_manager = ui_components.get('button_state_manager')
        return state_manager.is_operation_active() if state_manager else False
    except Exception:
        return False

# Convenience functions dengan preview support
disable_buttons = lambda ui_components: disable_all_buttons(ui_components)
enable_buttons = lambda ui_components: enable_all_buttons(ui_components)
set_processing = lambda ui_components, button_key: set_button_processing_state(ui_components, button_key, True)
clear_processing = lambda ui_components, button_key: set_button_processing_state(ui_components, button_key, False)
set_preview_generating = lambda ui_components, generating=True: create_button_state_manager(ui_components).set_preview_generating(generating)