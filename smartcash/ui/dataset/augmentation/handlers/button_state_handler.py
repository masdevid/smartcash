"""
File: smartcash/ui/dataset/augmentation/handlers/button_state_handler.py
Deskripsi: SRP handler untuk button state management dengan restore functionality
"""

from typing import Dict, Any, Optional
import threading
import time

class ButtonStateHandler:
    """SRP handler untuk mengelola state tombol UI dengan restore functionality."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.original_states = {}
    
    def save_button_state(self, button_key: str) -> bool:
        """
        Simpan state original button.
        
        Args:
            button_key: Key button dalam ui_components
            
        Returns:
            True jika berhasil disimpan
        """
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
        """
        Set button ke processing state.
        
        Args:
            button_key: Key button dalam ui_components
            processing_text: Text saat processing
            style: Button style untuk processing
            
        Returns:
            True jika berhasil diset
        """
        button = self.ui_components.get(button_key)
        if not button:
            return False
            
        # Save original state dulu
        self.save_button_state(button_key)
        
        # Set processing state
        button.description = processing_text
        button.button_style = style
        button.disabled = True
        if hasattr(button, 'icon'):
            button.icon = 'hourglass'
            
        return True
    
    def restore_button_state(self, button_key: str) -> bool:
        """
        Restore button ke state original.
        
        Args:
            button_key: Key button dalam ui_components
            
        Returns:
            True jika berhasil direstore
        """
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
            
        # Clear saved state
        if button_key in self.original_states:
            del self.original_states[button_key]
            
        return True
    
    def set_success_state(self, button_key: str, success_text: str = "Selesai!", 
                         duration: int = 3) -> bool:
        """
        Set button ke success state sementara dengan auto restore.
        
        Args:
            button_key: Key button dalam ui_components
            success_text: Text untuk success state
            duration: Durasi dalam detik sebelum auto restore
            
        Returns:
            True jika berhasil diset
        """
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
        """
        Set button ke error state sementara dengan auto restore.
        
        Args:
            button_key: Key button dalam ui_components
            error_text: Text untuk error state
            duration: Durasi sebelum auto restore
            
        Returns:
            True jika berhasil diset
        """
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
    
    def bulk_disable_buttons(self, button_keys: list) -> Dict[str, bool]:
        """
        Disable multiple buttons sekaligus.
        
        Args:
            button_keys: List button keys untuk disable
            
        Returns:
            Dict dengan status berhasil per button
        """
        results = {}
        for button_key in button_keys:
            button = self.ui_components.get(button_key)
            if button and hasattr(button, 'disabled'):
                self.save_button_state(button_key)
                button.disabled = True
                results[button_key] = True
            else:
                results[button_key] = False
        return results
    
    def bulk_restore_buttons(self, button_keys: list) -> Dict[str, bool]:
        """
        Restore multiple buttons sekaligus.
        
        Args:
            button_keys: List button keys untuk restore
            
        Returns:
            Dict dengan status berhasil per button
        """
        results = {}
        for button_key in button_keys:
            results[button_key] = self.restore_button_state(button_key)
        return results
    
    def get_button_state(self, button_key: str) -> Optional[Dict[str, Any]]:
        """
        Dapatkan current state button.
        
        Args:
            button_key: Key button dalam ui_components
            
        Returns:
            Dict state button atau None jika tidak ada
        """
        button = self.ui_components.get(button_key)
        if not button:
            return None
            
        return {
            'description': getattr(button, 'description', ''),
            'button_style': getattr(button, 'button_style', ''),
            'disabled': getattr(button, 'disabled', False),
            'icon': getattr(button, 'icon', ''),
            'has_saved_state': button_key in self.original_states
        }

# Factory function
def create_button_state_handler(ui_components: Dict[str, Any]) -> ButtonStateHandler:
    """Factory function untuk create button state handler."""
    return ButtonStateHandler(ui_components)

# One-liner utilities
set_button_processing = lambda handler, key, text="Processing...": handler.set_processing_state(key, text)
restore_button = lambda handler, key: handler.restore_button_state(key)
set_button_success = lambda handler, key, text="Success!": handler.set_success_state(key, text)
set_button_error = lambda handler, key, text="Error!": handler.set_error_state(key, text)