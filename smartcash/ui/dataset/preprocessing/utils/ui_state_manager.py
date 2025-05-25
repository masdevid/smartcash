"""
File: smartcash/ui/dataset/preprocessing/utils/ui_state_manager.py
Deskripsi: Fixed UI state manager dengan proper button color management dan operation control
"""

from typing import Dict, Any, Optional, List
from contextlib import contextmanager

class UIStateManager:
    """Fixed manager untuk UI state dengan proper button styling control."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.operation_states = {}
        self.button_original_states = {}
        self.debug_mode = True
        
        self.logger = ui_components.get('logger')
        if self.debug_mode and self.logger:
            self.logger.debug("🔧 UIStateManager initialized")
    
    @contextmanager
    def operation_context(self, operation_name: str, disable_others: bool = True):
        """Fixed context manager dengan proper button state management."""
        try:
            # Set processing state
            self._set_processing_state(operation_name, True)
            
            # Disable other buttons jika diminta
            if disable_others:
                self.disable_other_buttons(operation_name)
            
            if self.debug_mode and self.logger:
                self.logger.debug(f"🔄 Operation {operation_name} started, buttons disabled")
            
            yield
            
        finally:
            # Always restore states
            self._set_processing_state(operation_name, False)
            
            if disable_others:
                self.enable_other_buttons(operation_name)
            
            if self.debug_mode and self.logger:
                self.logger.debug(f"✅ Operation {operation_name} completed, buttons restored")
    
    def disable_other_buttons(self, current_operation: str, button_keys: List[str] = None):
        """Disable other buttons without changing their style colors."""
        if button_keys is None:
            button_keys = ['preprocess_button', 'cleanup_button', 'check_button', 'save_button', 'reset_button']
        
        current_button_key = self._get_button_key_for_operation(current_operation)
        disabled_count = 0
        
        for button_key in button_keys:
            if button_key == current_button_key:
                continue
                
            button = self.ui_components.get(button_key)
            if button and not button.disabled:
                # Store ONLY the disabled state, preserve colors
                if button_key not in self.button_original_states:
                    self.button_original_states[button_key] = {
                        'disabled': button.disabled
                        # Intentionally NOT storing button_style to avoid color changes
                    }
                
                # Only disable, don't change colors
                button.disabled = True
                disabled_count += 1
        
        if self.debug_mode and self.logger:
            self.logger.debug(f"🔒 Disabled {disabled_count} buttons during {current_operation}")
    
    def enable_other_buttons(self, current_operation: str, button_keys: List[str] = None):
        """Enable other buttons dan restore original states."""
        if button_keys is None:
            button_keys = ['preprocess_button', 'cleanup_button', 'check_button', 'save_button', 'reset_button']
        
        current_button_key = self._get_button_key_for_operation(current_operation)
        enabled_count = 0
        
        for button_key in button_keys:
            if button_key == current_button_key:
                continue
                
            button = self.ui_components.get(button_key)
            if button and button_key in self.button_original_states:
                original = self.button_original_states[button_key]
                button.disabled = original['disabled']
                
                # Clear stored state
                del self.button_original_states[button_key]
                enabled_count += 1
        
        if self.debug_mode and self.logger:
            self.logger.debug(f"🔓 Enabled {enabled_count} buttons after {current_operation}")
    
    def _get_button_key_for_operation(self, operation: str) -> str:
        """Map operation ke button key."""
        operation_to_button = {
            'preprocessing': 'preprocess_button',
            'cleanup': 'cleanup_button',
            'check': 'check_button',
            'save': 'save_button',
            'reset': 'reset_button'
        }
        return operation_to_button.get(operation, f"{operation}_button")
    
    def is_operation_running(self, operation_name: str) -> bool:
        """Check apakah operation sedang running."""
        return self.operation_states.get(operation_name, False)
    
    def get_running_operations(self) -> List[str]:
        """Get list of currently running operations."""
        return [op for op, running in self.operation_states.items() if running]
    
    def _set_processing_state(self, operation_name: str, processing: bool):
        """Set processing state untuk operation."""
        old_state = self.operation_states.get(operation_name, False)
        self.operation_states[operation_name] = processing
        
        # Set global state
        state_key = f'{operation_name}_running'
        self.ui_components[state_key] = processing
        
        if self.debug_mode and self.logger and old_state != processing:
            self.logger.debug(f"🔄 Operation {operation_name}: {old_state} → {processing}")
    
    def can_start_operation(self, operation_name: str, exclude_operations: List[str] = None) -> tuple[bool, str]:
        """Check apakah operation bisa dimulai."""
        exclude_operations = exclude_operations or []
        
        if self.is_operation_running(operation_name):
            return False, f"{operation_name.title()} sedang berjalan"
        
        running_ops = self.get_running_operations()
        conflicting_ops = [op for op in running_ops 
                          if op not in exclude_operations and op != operation_name]
        
        if conflicting_ops:
            return False, f"Tidak dapat memulai {operation_name}, {conflicting_ops[0]} sedang berjalan"
        
        return True, "Ready to start"
    
    def cleanup_all_states(self):
        """Cleanup semua operation states dan button states."""
        # Reset operation states
        for operation in list(self.operation_states.keys()):
            self._set_processing_state(operation, False)
        
        # Reset button states
        for button_key, original_state in list(self.button_original_states.items()):
            button = self.ui_components.get(button_key)
            if button:
                button.disabled = original_state['disabled']
        
        self.button_original_states.clear()
        
        if self.debug_mode and self.logger:
            self.logger.debug("🧹 All UI states cleaned up")

def get_ui_state_manager(ui_components: Dict[str, Any]) -> UIStateManager:
    """Factory function untuk mendapatkan UI state manager."""
    if 'ui_state_manager' not in ui_components:
        ui_components['ui_state_manager'] = UIStateManager(ui_components)
    return ui_components['ui_state_manager']