"""
File: smartcash/ui/dataset/preprocessing/utils/button_manager.py
Deskripsi: State management untuk buttons preprocessing
"""

from typing import Dict, Any

class PreprocessingButtonManager:
    """Button state management untuk preprocessing operations"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.disabled_buttons = []
    
    def disable_buttons(self, exclude_button: str = None):
        """Disable all buttons except exclude_button"""
        buttons = ['preprocess_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
        for btn_key in buttons:
            if btn_key != exclude_button and btn_key in self.ui_components:
                btn = self.ui_components[btn_key]
                if btn and hasattr(btn, 'disabled') and not btn.disabled:
                    btn.disabled = True
                    self.disabled_buttons.append(btn_key)
    
    def enable_buttons(self):
        """Re-enable previously disabled buttons"""
        for btn_key in self.disabled_buttons:
            if btn_key in self.ui_components:
                btn = self.ui_components[btn_key]
                if btn and hasattr(btn, 'disabled'):
                    btn.disabled = False
        self.disabled_buttons.clear()

def get_button_manager(ui_components: Dict[str, Any]) -> PreprocessingButtonManager:
    """Get button manager instance"""
    if 'button_manager' not in ui_components:
        ui_components['button_manager'] = PreprocessingButtonManager(ui_components)
    return ui_components['button_manager']