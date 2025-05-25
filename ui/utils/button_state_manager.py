"""
File: smartcash/ui/utils/button_state_manager.py
Deskripsi: SRP manager untuk button states dan operation context handling
"""

from contextlib import contextmanager
from typing import Dict, Any, List

class ButtonStateManager:
    """Manager untuk button states dan operation contexts."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.active_operations: List[str] = []
        
        # Button mapping untuk different operations
        self.button_groups = {
            'preprocessing': ['preprocess_button', 'save_button', 'reset_button'],
            'check': ['check_button', 'save_button', 'reset_button'],
            'cleanup': ['cleanup_button', 'preprocess_button', 'check_button']
        }
    
    @contextmanager
    def operation_context(self, operation: str):
        """Context manager untuk operation dengan automatic button state management."""
        try:
            self._start_operation(operation)
            yield
        except Exception as e:
            self._handle_operation_error(operation, str(e))
            raise
        finally:
            self._end_operation(operation)
    
    def _start_operation(self, operation: str) -> None:
        """Start operation dengan button state changes."""
        self.active_operations.append(operation)
        
        # Disable relevant buttons
        buttons_to_disable = self.button_groups.get(operation, [])
        for button_key in buttons_to_disable:
            button = self.ui_components.get(button_key)
            if button and hasattr(button, 'disabled'):
                button.disabled = True
        
        # Set processing state untuk primary button
        self._set_processing_state(operation)
    
    def _end_operation(self, operation: str) -> None:
        """End operation dengan restore button states."""
        if operation in self.active_operations:
            self.active_operations.remove(operation)
        
        # Re-enable buttons jika tidak ada operation lain yang aktif
        if not self.active_operations:
            self._restore_all_button_states()
        
        # Restore primary button state
        self._restore_button_state(operation)
    
    def _handle_operation_error(self, operation: str, error_message: str) -> None:
        """Handle operation error dengan button state management."""
        # Set error state untuk primary button
        primary_button = self._get_primary_button(operation)
        if primary_button:
            self._set_button_error_state(primary_button, error_message)
    
    def _set_processing_state(self, operation: str) -> None:
        """Set processing state untuk operation button."""
        primary_button = self._get_primary_button(operation)
        if primary_button:
            processing_texts = {
                'preprocessing': 'Processing...',
                'check': 'Checking...',
                'cleanup': 'Cleaning...'
            }
            
            # Store original state jika belum ada
            if not hasattr(primary_button, '_original_description'):
                primary_button._original_description = primary_button.description
                primary_button._original_style = getattr(primary_button, 'button_style', '')
            
            primary_button.description = processing_texts.get(operation, 'Processing...')
            primary_button.button_style = 'warning'
            primary_button.disabled = True
    
    def _restore_button_state(self, operation: str) -> None:
        """Restore button ke original state."""
        primary_button = self._get_primary_button(operation)
        if primary_button and hasattr(primary_button, '_original_description'):
            primary_button.description = primary_button._original_description
            primary_button.button_style = getattr(primary_button, '_original_style', '')
            primary_button.disabled = False
    
    def _restore_all_button_states(self) -> None:
        """Restore semua button states."""
        all_buttons = set()
        for button_list in self.button_groups.values():
            all_buttons.update(button_list)
        
        for button_key in all_buttons:
            button = self.ui_components.get(button_key)
            if button and hasattr(button, 'disabled'):
                button.disabled = False
                
                # Restore original state jika ada
                if hasattr(button, '_original_description'):
                    button.description = button._original_description
                if hasattr(button, '_original_style'):
                    button.button_style = button._original_style
    
    def _set_button_error_state(self, button, error_message: str) -> None:
        """Set error state untuk button."""
        button.description = "Error!"
        button.button_style = 'danger'
        button.disabled = False
        
        # Auto-restore setelah delay
        import threading
        import time
        
        def restore_after_delay():
            time.sleep(3)
            if hasattr(button, '_original_description'):
                button.description = button._original_description
                button.button_style = getattr(button, '_original_style', '')
        
        threading.Thread(target=restore_after_delay, daemon=True).start()
    
    def _get_primary_button(self, operation: str):
        """Get primary button untuk operation."""
        primary_mapping = {
            'preprocessing': 'preprocess_button',
            'check': 'check_button', 
            'cleanup': 'cleanup_button'
        }
        
        button_key = primary_mapping.get(operation)
        return self.ui_components.get(button_key) if button_key else None
    
    def get_operation_status(self) -> Dict[str, Any]:
        """Get current operation status."""
        return {
            'active_operations': self.active_operations.copy(),
            'has_active_operations': len(self.active_operations) > 0,
            'available_buttons': list(self.ui_components.keys())
        }
    
    def force_reset_all_states(self) -> None:
        """Force reset semua button states."""
        self.active_operations.clear()
        self._restore_all_button_states()

def get_button_state_manager(ui_components: Dict[str, Any]) -> ButtonStateManager:
    """Factory untuk mendapatkan button state manager."""
    if '_button_state_manager' not in ui_components:
        ui_components['_button_state_manager'] = ButtonStateManager(ui_components)
    return ui_components['_button_state_manager']