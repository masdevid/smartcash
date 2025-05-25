"""
File: smartcash/ui/dataset/preprocessing/utils/dialog_manager.py
Deskripsi: Enhanced dialog manager dengan positioning di confirmation area yang sudah disiapkan
"""

from typing import Dict, Any, Callable
from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog, create_destructive_confirmation
from IPython.display import display, clear_output

class DialogManager:
    """Enhanced dialog manager dengan positioning di confirmation area untuk better UX."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.active_dialog = None
        self.confirmation_area = ui_components.get('confirmation_area')
        
        # Get logger untuk debugging
        self.logger = ui_components.get('logger')
        
        if self.logger:
            self.logger.debug("🔧 DialogManager initialized dengan confirmation area positioning")
    
    def show_confirmation_dialog(self, title: str, message: str, 
                                on_confirm: Callable, confirm_text: str = "Ya", 
                                cancel_text: str = "Batal", danger_mode: bool = False):
        """Show confirmation dialog di confirmation area yang sudah disiapkan."""
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
        self._display_in_confirmation_area(dialog)
        
        if self.logger:
            self.logger.debug(f"🔔 Confirmation dialog shown: {title}")
    
    def show_destructive_confirmation(self, title: str, message: str,
                                    on_confirm: Callable, item_name: str = "item",
                                    confirm_text: str = None, cancel_text: str = "Batal"):
        """Show destructive confirmation di confirmation area yang sudah disiapkan."""
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
    
    def show_info_confirmation(self, title: str, message: str, 
                              on_confirm: Callable, info_type: str = "info"):
        """Show info confirmation dengan styling sesuai tipe."""
        button_style = {
            'info': False,
            'warning': False, 
            'success': False
        }.get(info_type, False)
        
        self.show_confirmation_dialog(
            title=title,
            message=message,
            on_confirm=on_confirm,
            confirm_text="Lanjutkan",
            cancel_text="Batal",
            danger_mode=button_style
        )
    
    def _display_in_confirmation_area(self, dialog):
        """Display dialog di confirmation area yang sudah disiapkan."""
        if self.confirmation_area:
            # Clear existing content di confirmation area
            with self.confirmation_area:
                clear_output(wait=True)
                display(dialog)
            
            if self.logger:
                self.logger.debug("🎯 Dialog displayed in dedicated confirmation area")
        else:
            # Fallback ke display biasa jika confirmation area tidak tersedia
            display(dialog)
            
            if self.logger:
                self.logger.warning("⚠️ Confirmation area tidak tersedia, menggunakan fallback display")
    
    def _handle_confirm(self, callback: Callable):
        """Handle confirm dengan cleanup dan logging."""
        if self.logger:
            self.logger.debug("✅ User confirmed dialog action")
        
        self._cleanup_existing_dialog()
        
        if callback:
            try:
                callback()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"❌ Error executing confirm callback: {str(e)}")
    
    def _handle_cancel(self):
        """Handle cancel dengan cleanup dan logging."""
        if self.logger:
            self.logger.debug("❌ User cancelled dialog action")
        
        self._cleanup_existing_dialog()
    
    def _cleanup_existing_dialog(self):
        """Cleanup dialog yang aktif dari confirmation area."""
        if self.active_dialog:
            try:
                # Clear confirmation area
                if self.confirmation_area:
                    with self.confirmation_area:
                        clear_output(wait=True)
                
                # Close dialog jika memiliki method close
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
        """Public cleanup method untuk manual cleanup."""
        self._cleanup_existing_dialog()
    
    def is_dialog_active(self) -> bool:
        """Check apakah ada dialog yang sedang aktif."""
        return self.active_dialog is not None
    
    def get_confirmation_area_status(self) -> Dict[str, Any]:
        """Get status confirmation area untuk debugging."""
        return {
            'area_available': self.confirmation_area is not None,
            'dialog_active': self.is_dialog_active(),
            'area_layout': str(self.confirmation_area.layout) if self.confirmation_area else None
        }

def get_dialog_manager(ui_components: Dict[str, Any]) -> DialogManager:
    """Factory function untuk mendapatkan enhanced dialog manager."""
    if 'dialog_manager' not in ui_components:
        ui_components['dialog_manager'] = DialogManager(ui_components)
    return ui_components['dialog_manager']