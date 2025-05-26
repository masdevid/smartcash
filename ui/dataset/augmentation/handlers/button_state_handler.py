"""
File: smartcash/ui/dataset/augmentation/handlers/button_state_handler.py
Deskripsi: Fixed SRP handler untuk button state dengan reset progress/logs dan proper click behavior
"""

from typing import Dict, Any, Optional
import threading
import time

class ButtonStateHandler:
    """SRP handler untuk mengelola state tombol dengan reset progress/logs functionality."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.original_states = {}
        self.operation_buttons = ['augment_button', 'check_button', 'cleanup_button']
        self.config_buttons = ['save_button', 'reset_button']
    
    def reset_progress_and_logs(self) -> None:
        """One-liner reset progress bar dan clear log output."""
        try:
            # One-liner progress reset
            [(getattr(self.ui_components.get('tracker'), 'reset', lambda: None)(), 
              self.ui_components.get('reset_all', lambda: None)())]
            
            # One-liner log clearing
            [getattr(widget, 'clear_output', lambda **kw: None)(wait=True) 
             for key in ['log_output', 'status'] if (widget := self.ui_components.get(key))]
        except Exception:
            pass
    
    def prepare_for_operation(self, button_key: str) -> None:
        """One-liner prepare UI untuk operasi."""
        self.reset_progress_and_logs()
        # One-liner disable operation buttons
        [self.save_button_state(btn) or setattr(self.ui_components.get(btn), 'disabled', True) 
         for btn in self.operation_buttons if self.ui_components.get(btn)]
    
    def save_button_state(self, button_key: str) -> bool:
        """Simpan state original button."""
        button = self.ui_components.get(button_key)
        if not button or not hasattr(button, 'description'):
            return False
            
        self.original_states[button_key] = {
            'description': button.description,
            'button_style': getattr(button, 'button_style', ''),
            'disabled': getattr(button, 'disabled', False),
            'icon': getattr(button, 'icon', '')
        }
        return True
    
    def set_processing_state(self, button_key: str, processing_text: str = "Processing...", 
                           style: str = 'warning') -> bool:
        """Set button ke processing state dengan disable operation buttons."""
        button = self.ui_components.get(button_key)
        if not button:
            return False
        
        # Prepare untuk operasi
        if button_key in self.operation_buttons:
            self.prepare_for_operation(button_key)
        else:
            # Untuk config buttons, hanya reset logs
            self.reset_progress_and_logs()
            self.save_button_state(button_key)
        
        # Set processing state
        button.description = processing_text
        button.button_style = style
        button.disabled = True
        if hasattr(button, 'icon'):
            button.icon = 'hourglass'
            
        return True
    
    def restore_button_state(self, button_key: str) -> bool:
        """Restore button ke state original dan enable semua buttons."""
        button = self.ui_components.get(button_key)
        original = self.original_states.get(button_key)
        
        if not button or not original:
            return False
            
        # Restore properties
        button.description = original['description']
        button.button_style = original['button_style']
        button.disabled = original['disabled']
        if hasattr(button, 'icon'):
            button.icon = original['icon']
        
        # Enable semua operation buttons kembali
        if button_key in self.operation_buttons:
            self._enable_all_operation_buttons()
        
        # Clear saved state
        if button_key in self.original_states:
            del self.original_states[button_key]
            
        return True
    
    def _enable_all_operation_buttons(self) -> None:
        """One-liner enable semua operation buttons."""
        [setattr(self.ui_components.get(btn), 'disabled', 
                self.original_states.get(btn, {}).get('disabled', False)) 
         for btn in self.operation_buttons if self.ui_components.get(btn)]
    
    def set_success_state(self, button_key: str, success_text: str = "Selesai!", 
                         duration: int = 3) -> bool:
        """Set button ke success state dengan auto restore."""
        button = self.ui_components.get(button_key)
        if not button:
            return False
            
        # Set success state
        button.description = success_text
        button.button_style = 'success'
        if hasattr(button, 'icon'):
            button.icon = 'check'
        
        # Auto restore setelah duration
        def delayed_restore():
            time.sleep(duration)
            self.restore_button_state(button_key)
            
        threading.Thread(target=delayed_restore, daemon=True).start()
        return True
    
    def set_error_state(self, button_key: str, error_text: str = "Error!", 
                       duration: int = 3) -> bool:
        """Set button ke error state dengan auto restore."""
        button = self.ui_components.get(button_key)
        if not button:
            return False
            
        # Set error state
        button.description = error_text
        button.button_style = 'danger'
        if hasattr(button, 'icon'):
            button.icon = 'times'
        
        # Auto restore setelah duration
        def delayed_restore():
            time.sleep(duration)
            self.restore_button_state(button_key)
            
        threading.Thread(target=delayed_restore, daemon=True).start()
        return True
    
    def handle_config_button_click(self, button_key: str) -> None:
        """Special handler untuk config buttons yang reset logs tapi tidak disable other buttons."""
        # Reset progress dan logs saja
        self.reset_progress_and_logs()
        
        # Save state untuk restore nanti
        self.save_button_state(button_key)
    
    def get_button_state(self, button_key: str) -> Optional[Dict[str, Any]]:
        """Dapatkan current state button."""
        button = self.ui_components.get(button_key)
        if not button:
            return None
            
        return {
            'description': getattr(button, 'description', ''),
            'button_style': getattr(button, 'button_style', ''),
            'disabled': getattr(button, 'disabled', False),
            'icon': getattr(button, 'icon', ''),
            'has_saved_state': button_key in self.original_states,
            'is_operation_button': button_key in self.operation_buttons,
            'is_config_button': button_key in self.config_buttons
        }

# Factory function
def create_button_state_handler(ui_components: Dict[str, Any]) -> ButtonStateHandler:
    """Factory function untuk create button state handler."""
    return ButtonStateHandler(ui_components)

# One-liner utilities
prepare_operation = lambda handler, key: handler.prepare_for_operation(key)
set_button_processing = lambda handler, key, text="Processing...": handler.set_processing_state(key, text)
restore_button = lambda handler, key: handler.restore_button_state(key)
set_button_success = lambda handler, key, text="Success!": handler.set_success_state(key, text)
set_button_error = lambda handler, key, text="Error!": handler.set_error_state(key, text)
handle_config_click = lambda handler, key: handler.handle_config_button_click(key)