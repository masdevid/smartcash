"""
File: smartcash/ui/dataset/augmentation/utils/button_manager.py
Deskripsi: Button state management untuk augmentation operations
"""

from typing import Dict, Any, List

class AugmentationButtonManager:
    """Button state manager untuk augmentation operations"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.operation_buttons = ['augment_button', 'check_button', 'cleanup_button']
        self.config_buttons = ['save_button', 'reset_button']
    
    def disable_operation_buttons(self, processing_text: str = "⏳ Processing..."):
        """Disable operation buttons dengan processing indicator"""
        for key in self.operation_buttons:
            button = self.ui_components.get(key)
            if button and hasattr(button, 'disabled'):
                button.disabled = True
                if hasattr(button, 'description'):
                    if not hasattr(button, '_original_description'):
                        button._original_description = button.description
                    button.description = processing_text
    
    def enable_operation_buttons(self):
        """Enable operation buttons dan restore descriptions"""
        for key in self.operation_buttons:
            button = self.ui_components.get(key)
            if button and hasattr(button, 'disabled'):
                button.disabled = False
                if hasattr(button, '_original_description'):
                    button.description = button._original_description
    
    def disable_config_buttons(self):
        """Disable config buttons during operations"""
        for key in self.config_buttons:
            button = self.ui_components.get(key)
            if button and hasattr(button, 'disabled'):
                button.disabled = True
    
    def enable_config_buttons(self):
        """Enable config buttons"""
        for key in self.config_buttons:
            button = self.ui_components.get(key)
            if button and hasattr(button, 'disabled'):
                button.disabled = False
    
    def disable_all_buttons(self):
        """Disable semua buttons"""
        self.disable_operation_buttons()
        self.disable_config_buttons()
    
    def enable_all_buttons(self):
        """Enable semua buttons"""
        self.enable_operation_buttons()
        self.enable_config_buttons()

# Factory function
def create_button_manager(ui_components: Dict[str, Any]) -> AugmentationButtonManager:
    """Factory untuk button manager"""
    return AugmentationButtonManager(ui_components)