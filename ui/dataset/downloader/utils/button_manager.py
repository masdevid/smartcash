"""
File: smartcash/ui/dataset/downloader/utils/button_manager.py
Deskripsi: Enhanced button state management dengan dialog support
"""

from typing import Dict, Any, List

class SimpleButtonManager:
    """Enhanced button state management dengan dialog support"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.disabled_buttons = []
        self.original_descriptions = {}
    
    def disable_buttons(self, exclude_button: str = None, processing_text: str = None):
        """Disable all buttons dengan optional processing text"""
        buttons = ['download_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
        
        for btn_key in buttons:
            if btn_key != exclude_button and btn_key in self.ui_components:
                btn = self.ui_components[btn_key]
                if btn and hasattr(btn, 'disabled') and not btn.disabled:
                    # Save original description
                    if hasattr(btn, 'description'):
                        self.original_descriptions[btn_key] = btn.description
                    
                    btn.disabled = True
                    self.disabled_buttons.append(btn_key)
                    
                    # Set processing text for active button
                    if btn_key == exclude_button and processing_text and hasattr(btn, 'description'):
                        btn.description = processing_text
    
    def enable_buttons(self):
        """Re-enable buttons dan restore original descriptions"""
        for btn_key in self.disabled_buttons:
            if btn_key in self.ui_components:
                btn = self.ui_components[btn_key]
                if btn and hasattr(btn, 'disabled'):
                    btn.disabled = False
                    
                    # Restore original description
                    if btn_key in self.original_descriptions and hasattr(btn, 'description'):
                        btn.description = self.original_descriptions[btn_key]
        
        self.disabled_buttons.clear()
        self.original_descriptions.clear()
    
    def set_button_processing(self, button_key: str, processing_text: str = "Processing..."):
        """Set specific button ke processing state"""
        if button_key in self.ui_components:
            btn = self.ui_components[button_key]
            if btn and hasattr(btn, 'disabled'):
                if hasattr(btn, 'description'):
                    self.original_descriptions[button_key] = btn.description
                    btn.description = processing_text
                btn.disabled = True
                if button_key not in self.disabled_buttons:
                    self.disabled_buttons.append(button_key)
    
    def restore_button(self, button_key: str):
        """Restore specific button state"""
        if button_key in self.ui_components:
            btn = self.ui_components[button_key]
            if btn and hasattr(btn, 'disabled'):
                btn.disabled = False
                
                # Restore description
                if button_key in self.original_descriptions and hasattr(btn, 'description'):
                    btn.description = self.original_descriptions[button_key]
                    del self.original_descriptions[button_key]
                
                # Remove from disabled list
                if button_key in self.disabled_buttons:
                    self.disabled_buttons.remove(button_key)

def get_button_manager(ui_components: Dict[str, Any]) -> SimpleButtonManager:
    """Get button manager instance"""
    if 'button_manager' not in ui_components:
        ui_components['button_manager'] = SimpleButtonManager(ui_components)
    return ui_components['button_manager']