"""
File: smartcash/ui/dataset/preprocessing/utils/dialog_manager.py
Deskripsi: Utility untuk mengelola dialog confirmation dengan consistent pattern
"""

from typing import Dict, Any, Callable
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog, create_destructive_confirmation
from IPython.display import display

class DialogManager:
    """Manager untuk handling dialog confirmation dengan consistent pattern."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.active_dialog = None
    
    def show_confirmation_dialog(self, title: str, message: str, 
                                on_confirm: Callable, confirm_text: str = "Ya", 
                                cancel_text: str = "Batal", danger_mode: bool = False):
        """Show confirmation dialog dengan auto cleanup."""
        self._cleanup_existing_dialog()
        
        dialog = create_confirmation_dialog(
            title=title,
            message=message,
            on_confirm=lambda b: self._handle_confirm(on_confirm),
            on_cancel=lambda b: self._handle_cancel(),
            confirm_text=confirm_text,
            cancel_text=cancel_text,
            danger_mode=danger_mode
        )
        
        self.active_dialog = dialog
        display(dialog)
    
    def show_destructive_confirmation(self, title: str, message: str,
                                    on_confirm: Callable, item_name: str = "item",
                                    confirm_text: str = None, cancel_text: str = "Batal"):
        """Show destructive confirmation dengan auto cleanup."""
        self._cleanup_existing_dialog()
        
        dialog = create_destructive_confirmation(
            title=title,
            message=message,
            on_confirm=lambda b: self._handle_confirm(on_confirm),
            on_cancel=lambda b: self._handle_cancel(),
            item_name=item_name,
            confirm_text=confirm_text,
            cancel_text=cancel_text
        )
        
        self.active_dialog = dialog
        display(dialog)
    
    def _handle_confirm(self, callback: Callable):
        """Handle confirm dengan cleanup."""
        self._cleanup_existing_dialog()
        if callback:
            callback()
    
    def _handle_cancel(self):
        """Handle cancel dengan cleanup."""
        self._cleanup_existing_dialog()
    
    def _cleanup_existing_dialog(self):
        """Cleanup dialog yang aktif."""
        if self.active_dialog:
            try:
                self.active_dialog.close()
            except:
                pass
            self.active_dialog = None
    
    def cleanup(self):
        """Public cleanup method."""
        self._cleanup_existing_dialog()

def get_dialog_manager(ui_components: Dict[str, Any]) -> DialogManager:
    """Factory function untuk mendapatkan dialog manager."""
    if 'dialog_manager' not in ui_components:
        ui_components['dialog_manager'] = DialogManager(ui_components)
    return ui_components['dialog_manager']