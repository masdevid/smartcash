"""
File: smartcash/ui/dataset/preprocessing/utils/dialog_manager.py
Deskripsi: Fixed dialog manager dengan auto-clear confirmation area dan unlimited height
"""

from typing import Dict, Any, Callable
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog, create_destructive_confirmation
from IPython.display import display, clear_output

class DialogManager:
    """Fixed dialog manager dengan auto-clear confirmation area."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.active_dialog = None
        self.confirmation_area = ui_components.get('confirmation_area')
        self.logger = ui_components.get('logger')
        
        if self.logger:
            self.logger.debug("🔧 DialogManager initialized")
    
    def show_confirmation_dialog(self, title: str, message: str, 
                                on_confirm: Callable, confirm_text: str = "Ya", 
                                cancel_text: str = "Batal", danger_mode: bool = False):
        """Show confirmation dialog dengan auto-clear functionality."""
        self._cleanup_existing_dialog()
        
        dialog = create_confirmation_dialog(
            title=title,
            message=message,
            on_confirm=lambda b: self._handle_confirm(on_confirm),
            on_cancel=lambda b: self._handle_cancel(),
            confirm_text=confirm_text,
            cancel_text=cancel_text,
            danger_mode=danger_mode,
            dialog_width="600px"  # Fixed width to prevent overflow
        )
        
        self.active_dialog = dialog
        self._display_in_confirmation_area(dialog)
        
        if self.logger:
            self.logger.debug(f"🔔 Confirmation dialog shown: {title}")
    
    def show_destructive_confirmation(self, title: str, message: str,
                                    on_confirm: Callable, item_name: str = "item",
                                    confirm_text: str = None, cancel_text: str = "Batal"):
        """Show destructive confirmation dengan auto-clear."""
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
        self._display_in_confirmation_area(dialog)
        
        if self.logger:
            self.logger.debug(f"⚠️ Destructive confirmation shown: {title}")
    
    def _display_in_confirmation_area(self, dialog):
        """Display dialog di confirmation area dengan unlimited height."""
        if self.confirmation_area:
            # Update confirmation area layout untuk unlimited height
            self.confirmation_area.layout.max_height = None  # Remove height limit
            self.confirmation_area.layout.height = 'auto'
            
            with self.confirmation_area:
                clear_output(wait=True)
                display(dialog)
            
            if self.logger:
                self.logger.debug("🎯 Dialog displayed in confirmation area")
        else:
            display(dialog)
            if self.logger:
                self.logger.warning("⚠️ Confirmation area tidak tersedia")
    
    def _handle_confirm(self, callback: Callable):
        """Handle confirm dengan auto-clear confirmation area."""
        if self.logger:
            self.logger.debug("✅ User confirmed dialog action")
        
        # Clear confirmation area BEFORE executing callback
        self._cleanup_existing_dialog()
        
        if callback:
            try:
                callback()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"❌ Error executing confirm callback: {str(e)}")
    
    def _handle_cancel(self):
        """Handle cancel dengan auto-clear confirmation area."""
        if self.logger:
            self.logger.debug("❌ User cancelled dialog action")
        
        # Clear confirmation area
        self._cleanup_existing_dialog()
    
    def _cleanup_existing_dialog(self):
        """Cleanup dialog yang aktif dari confirmation area."""
        if self.active_dialog:
            try:
                if self.confirmation_area:
                    with self.confirmation_area:
                        clear_output(wait=True)
                
                if hasattr(self.active_dialog, 'close'):
                    self.active_dialog.close()
                    
            except Exception as e:
                if self.logger:
                    self.logger.debug(f"🧹 Dialog cleanup warning: {str(e)}")
            finally:
                self.active_dialog = None
                
                if self.logger:
                    self.logger.debug("🧹 Dialog cleanup completed")
    
    def cleanup(self):
        """Public cleanup method."""
        self._cleanup_existing_dialog()
    
    def is_dialog_active(self) -> bool:
        """Check apakah ada dialog yang sedang aktif."""
        return self.active_dialog is not None

def get_dialog_manager(ui_components: Dict[str, Any]) -> DialogManager:
    """Factory function untuk mendapatkan dialog manager."""
    if 'dialog_manager' not in ui_components:
        ui_components['dialog_manager'] = DialogManager(ui_components)
    return ui_components['dialog_manager']