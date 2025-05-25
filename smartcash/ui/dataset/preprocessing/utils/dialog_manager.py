"""
File: smartcash/ui/dataset/preprocessing/utils/dialog_manager.py
Deskripsi: Enhanced dialog manager dengan persistent state handling dan auto-cleanup
"""

from typing import Dict, Any, Callable
from smartcash.ui.components.confirmation_dialog import (
    create_confirmation_dialog, 
    create_destructive_confirmation,
    cleanup_all_dialogs
)
from IPython.display import display, clear_output

class DialogManager:
    """Enhanced dialog manager dengan persistent state management dan auto-cleanup."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.active_dialog = None
        self.confirmation_area = ui_components.get('confirmation_area')
        self.logger = ui_components.get('logger')
        
        # Auto cleanup existing dialogs saat initialization
        cleanup_all_dialogs()
        
        if self.logger:
            self.logger.debug("🔧 DialogManager initialized dengan auto-cleanup")
    
    def show_confirmation_dialog(self, title: str, message: str, 
                                on_confirm: Callable, confirm_text: str = "Ya", 
                                cancel_text: str = "Batal", danger_mode: bool = False):
        """Show confirmation dialog dengan enhanced auto-cleanup."""
        # Cleanup existing dialog first
        self._cleanup_existing_dialog()
        
        # Create wrapped callbacks dengan auto-cleanup
        def wrapped_confirm():
            self._cleanup_existing_dialog()
            try:
                on_confirm()
            except Exception as e:
                self.logger and self.logger.error(f"❌ Confirm callback error: {str(e)}")
        
        def wrapped_cancel():
            self._cleanup_existing_dialog()
            self.logger and self.logger.debug("❌ User cancelled dialog action")
        
        dialog = create_confirmation_dialog(
            title=title,
            message=message,
            on_confirm=lambda b: wrapped_confirm(),
            on_cancel=lambda b: wrapped_cancel(),
            confirm_text=confirm_text,
            cancel_text=cancel_text,
            danger_mode=danger_mode,
            dialog_width="600px"
        )
        
        self.active_dialog = dialog
        self._display_in_confirmation_area(dialog)
        
        if self.logger:
            self.logger.debug(f"🔔 Confirmation dialog shown: {title}")
    
    def show_destructive_confirmation(self, title: str, message: str,
                                    on_confirm: Callable, item_name: str = "item",
                                    confirm_text: str = None, cancel_text: str = "Batal"):
        """Show destructive confirmation dengan auto-cleanup."""
        # Cleanup existing dialog first
        self._cleanup_existing_dialog()
        
        # Create wrapped callbacks
        def wrapped_confirm():
            self._cleanup_existing_dialog()
            try:
                on_confirm()
            except Exception as e:
                self.logger and self.logger.error(f"❌ Destructive confirm error: {str(e)}")
        
        def wrapped_cancel():
            self._cleanup_existing_dialog()
            self.logger and self.logger.debug("❌ User cancelled destructive action")
        
        dialog = create_destructive_confirmation(
            title=title,
            message=message,
            on_confirm=lambda b: wrapped_confirm(),
            on_cancel=lambda b: wrapped_cancel(),
            item_name=item_name,
            confirm_text=confirm_text,
            cancel_text=cancel_text
        )
        
        self.active_dialog = dialog
        self._display_in_confirmation_area(dialog)
        
        if self.logger:
            self.logger.debug(f"⚠️ Destructive confirmation shown: {title}")
    
    def _display_in_confirmation_area(self, dialog):
        """Display dialog di confirmation area dengan proper layout."""
        if self.confirmation_area:
            # Update confirmation area layout
            self.confirmation_area.layout.max_height = None
            self.confirmation_area.layout.height = 'auto'
            
            with self.confirmation_area:
                clear_output(wait=True)
                display(dialog)