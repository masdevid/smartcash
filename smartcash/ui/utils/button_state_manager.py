"""
File: smartcash/ui/utils/button_state_manager.py
Deskripsi: Fixed button state manager dengan guaranteed fallback dan one-liner style
"""

from typing import Dict, Any
from contextlib import contextmanager

class ButtonStateManager:
    """Button state manager dengan context management"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.operation_states = {}
    
    def can_start_operation(self, operation_name: str):
        """Check if operation dapat dimulai"""
        is_running = self.operation_states.get(operation_name, False)
        return not is_running, "Operation sudah berjalan" if is_running else "Ready to start"
    
    @contextmanager
    def operation_context(self, operation_name: str):
        """Context untuk operation dengan button state management"""
        operation_buttons = ['augment_button', 'check_button', 'cleanup_button']
        disabled_buttons = []
        
        try:
            # Set running state
            self.operation_states[operation_name] = True
            
            # Disable operation buttons
            for btn_key in operation_buttons:
                btn = self.ui_components.get(btn_key)
                if btn and hasattr(btn, 'disabled') and not btn.disabled:
                    btn.disabled = True
                    disabled_buttons.append(btn_key)
            
            yield
            
        finally:
            # Reset state
            self.operation_states[operation_name] = False
            
            # Re-enable buttons
            for btn_key in disabled_buttons:
                btn = self.ui_components.get(btn_key)
                if btn and hasattr(btn, 'disabled'):
                    btn.disabled = False
    
    @contextmanager
    def config_context(self, config_operation: str):
        """Context untuk config operations"""
        config_buttons = ['save_button', 'reset_button']
        disabled_buttons = []
        
        try:
            # Disable config buttons temporarily
            for btn_key in config_buttons:
                btn = self.ui_components.get(btn_key)
                if btn and hasattr(btn, 'disabled') and not btn.disabled:
                    btn.disabled = True
                    disabled_buttons.append(btn_key)
            
            yield
            
        finally:
            # Re-enable config buttons
            for btn_key in disabled_buttons:
                btn = self.ui_components.get(btn_key)
                if btn and hasattr(btn, 'disabled'):
                    btn.disabled = False

class MinimalManager:
    """Minimal fallback manager dengan semua required methods"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
    
    can_start_operation = lambda self, op: (True, "Minimal manager - allowed")
    
    @contextmanager
    def operation_context(self, operation_name: str):
        """Minimal operation context"""
        try:
            yield
        finally:
            pass
    
    @contextmanager
    def config_context(self, config_operation: str):
        """Minimal config context"""
        try:
            yield
        finally:
            pass

def get_button_state_manager(ui_components: Dict[str, Any]):
    """Get button state manager dengan guaranteed fallback"""
    try:
        # Try to create full manager
        return ButtonStateManager(ui_components)
    except Exception:
        # Always return minimal manager as fallback
        return MinimalManager(ui_components)

# One-liner utilities
create_manager = lambda ui_components: get_button_state_manager(ui_components)
get_safe_manager = lambda ui_components: MinimalManager(ui_components)  # Guaranteed fallback