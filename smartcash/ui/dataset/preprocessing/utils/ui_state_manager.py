"""
File: smartcash/ui/dataset/preprocessing/utils/ui_state_manager.py
Deskripsi: Manager untuk UI state dan button management dengan consistent pattern
"""

from typing import Dict, Any, Optional
from contextlib import contextmanager

class UIStateManager:
    """Manager untuk UI state dan button operations."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.operation_states = {}
    
    @contextmanager
    def operation_context(self, operation_name: str, button_key: str, 
                         processing_text: str = "Processing...", 
                         processing_style: str = 'warning'):
        """Context manager untuk button operation state."""
        # Store original state
        button = self.ui_components.get(button_key)
        if not button:
            yield
            return
        
        original_disabled = button.disabled
        original_description = button.description
        original_style = button.button_style
        
        try:
            # Set processing state
            self._set_processing_state(operation_name, True)
            button.disabled = True
            button.description = processing_text
            button.button_style = processing_style
            
            yield
            
        finally:
            # Restore original state
            self._set_processing_state(operation_name, False)
            button.disabled = original_disabled
            button.description = original_description
            button.button_style = original_style
    
    def set_button_processing(self, button_key: str, processing: bool = True, 
                            processing_text: str = "Processing...",
                            processing_style: str = 'warning',
                            success_text: str = None,
                            success_style: str = 'success'):
        """Set button ke processing state atau restore."""
        button = self.ui_components.get(button_key)
        if not button:
            return
        
        if processing:
            # Store original state
            if not hasattr(button, '_original_state'):
                button._original_state = {
                    'disabled': button.disabled,
                    'description': button.description,
                    'button_style': button.button_style
                }
            
            button.disabled = True
            button.description = processing_text
            button.button_style = processing_style
        else:
            # Restore or set success state
            if hasattr(button, '_original_state'):
                button.disabled = button._original_state['disabled']
                button.description = success_text or button._original_state['description']
                button.button_style = success_style if success_text else button._original_state['button_style']
                delattr(button, '_original_state')
    
    def is_operation_running(self, operation_name: str) -> bool:
        """Check apakah operation sedang running."""
        return self.operation_states.get(operation_name, False)
    
    def _set_processing_state(self, operation_name: str, processing: bool):
        """Set processing state untuk operation."""
        self.operation_states[operation_name] = processing
        
        # Set global state juga
        state_key = f'{operation_name}_running'
        self.ui_components[state_key] = processing
    
    def can_start_operation(self, operation_name: str, exclude_operations: list = None) -> tuple[bool, str]:
        """Check apakah operation bisa dimulai (tidak ada conflict)."""
        exclude_operations = exclude_operations or []
        
        # Check if current operation already running
        if self.is_operation_running(operation_name):
            return False, f"{operation_name.title()} sedang berjalan"
        
        # Check conflicting operations
        conflicting_ops = [op for op, running in self.operation_states.items() 
                          if running and op not in exclude_operations and op != operation_name]
        
        if conflicting_ops:
            return False, f"Tidak dapat memulai {operation_name}, {conflicting_ops[0]} sedang berjalan"
        
        return True, "Ready to start"
    
    def cleanup_all_states(self):
        """Cleanup semua operation states."""
        for operation in list(self.operation_states.keys()):
            self._set_processing_state(operation, False)
        
        # Cleanup button states
        button_keys = ['preprocess_button', 'cleanup_button', 'check_button', 'save_button', 'reset_button']
        for button_key in button_keys:
            button = self.ui_components.get(button_key)
            if button and hasattr(button, '_original_state'):
                button.disabled = button._original_state['disabled']
                button.description = button._original_state['description']
                button.button_style = button._original_state['button_style']
                delattr(button, '_original_state')

def get_ui_state_manager(ui_components: Dict[str, Any]) -> UIStateManager:
    """Factory function untuk mendapatkan UI state manager."""
    if 'ui_state_manager' not in ui_components:
        ui_components['ui_state_manager'] = UIStateManager(ui_components)
    return ui_components['ui_state_manager']