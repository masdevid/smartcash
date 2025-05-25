"""
File: smartcash/ui/utils/button_state_manager.py
Deskripsi: Enhanced button state manager dengan global disable semua buttons saat operation berjalan
"""

from typing import Dict, Any, List, Optional
from contextlib import contextmanager

class ButtonStateManager:
    """Enhanced button state manager dengan global operation control."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.operation_states = {}
        self.button_original_states = {}
        self.logger = ui_components.get('logger')
        
        # All button keys yang akan dimanage
        self.managed_buttons = [
            'preprocess_button', 'download_button', 'cleanup_button', 
            'check_button', 'save_button', 'reset_button'
        ]
    
    @contextmanager
    def operation_context(self, operation_name: str):
        """Context manager dengan global button disable."""
        try:
            # Set processing state
            self._set_processing_state(operation_name, True)
            
            # Disable ALL buttons
            self._disable_all_buttons()
            
            self.logger and self.logger.debug(f"🔒 Operation {operation_name} started - all buttons disabled")
            
            yield
            
        finally:
            # Always restore states
            self._set_processing_state(operation_name, False)
            self._enable_all_buttons()
            
            self.logger and self.logger.debug(f"🔓 Operation {operation_name} completed - all buttons restored")
    
    def _disable_all_buttons(self):
        """Disable semua managed buttons dan store original states."""
        for button_key in self.managed_buttons:
            button = self.ui_components.get(button_key)
            if button and hasattr(button, 'disabled'):
                # Store original state
                if button_key not in self.button_original_states:
                    self.button_original_states[button_key] = {
                        'disabled': button.disabled,
                        'button_style': getattr(button, 'button_style', ''),
                        'description': getattr(button, 'description', '')
                    }
                
                # Disable button
                button.disabled = True
    
    def _enable_all_buttons(self):
        """Enable all buttons dan restore original states."""
        for button_key in self.managed_buttons:
            button = self.ui_components.get(button_key)
            if button and button_key in self.button_original_states:
                original = self.button_original_states[button_key]
                
                # Restore original state
                button.disabled = original['disabled']
                
                # Optionally restore other properties if needed
                if hasattr(button, 'button_style'):
                    button.button_style = original['button_style']
                if hasattr(button, 'description'):
                    button.description = original['description']
                
                # Clear stored state
                del self.button_original_states[button_key]
    
    def _set_processing_state(self, operation_name: str, processing: bool):
        """Set processing state untuk operation."""
        self.operation_states[operation_name] = processing
        
        # Set global state
        state_key = f'{operation_name}_running'
        self.ui_components[state_key] = processing
    
    def is_operation_running(self, operation_name: str) -> bool:
        """Check apakah operation sedang running."""
        return self.operation_states.get(operation_name, False)
    
    def get_running_operations(self) -> List[str]:
        """Get list of currently running operations."""
        return [op for op, running in self.operation_states.items() if running]
    
    def can_start_operation(self, operation_name: str) -> tuple[bool, str]:
        """Check apakah operation bisa dimulai."""
        if self.is_operation_running(operation_name):
            return False, f"{operation_name.title()} sedang berjalan"
        
        running_ops = self.get_running_operations()
        if running_ops:
            return False, f"Tidak dapat memulai {operation_name}, {running_ops[0]} sedang berjalan"
        
        return True, "Ready to start"
    
    def force_enable_all_buttons(self):
        """Force enable semua buttons (emergency recovery)."""
        for button_key in self.managed_buttons:
            button = self.ui_components.get(button_key)
            if button and hasattr(button, 'disabled'):
                button.disabled = False
        
        # Clear all stored states
        self.button_original_states.clear()
        self.operation_states.clear()
        
        self.logger and self.logger.info("🆘 Emergency button recovery executed")
    
    def cleanup_all_states(self):
        """Cleanup semua operation states."""
        # Reset operation states
        for operation in list(self.operation_states.keys()):
            self._set_processing_state(operation, False)
        
        # Force enable all buttons
        self._enable_all_buttons()
        
        self.logger and self.logger.debug("🧹 All button states cleaned up")

def get_button_state_manager(ui_components: Dict[str, Any]) -> ButtonStateManager:
    """Factory function untuk mendapatkan button state manager."""
    if 'button_state_manager' not in ui_components:
        ui_components['button_state_manager'] = ButtonStateManager(ui_components)
    return ui_components['button_state_manager']