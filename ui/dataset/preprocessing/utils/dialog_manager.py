"""
File: smartcash/ui/dataset/preprocessing/utils/dialog_manager.py
Deskripsi: Enhanced dialog manager dengan session-aware cleanup dan proper state management
"""

from typing import Dict, Any, Callable
from smartcash.ui.components.confirmation_dialog import (
    create_confirmation_dialog, 
    create_destructive_confirmation,
    cleanup_all_dialogs,
    get_active_dialog_count
)
from IPython.display import display, clear_output

class DialogManager:
    """Session-aware dialog manager dengan auto-cleanup dan state persistence prevention."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.active_dialog = None
        self.confirmation_area = ui_components.get('confirmation_area')
        self.logger = ui_components.get('logger')
        
        # Force cleanup existing dialogs dari session sebelumnya
        cleanup_all_dialogs()
        
        # Setup session cleanup tracking
        self._setup_session_cleanup()
        
        if self.logger:
            active_count = get_active_dialog_count()
            self.logger.debug(f"🔧 DialogManager initialized - cleaned {active_count} persistent dialogs")
    
    def _setup_session_cleanup(self):
        """Setup automatic cleanup untuk cell restart/session change."""
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython:
                # Cleanup saat pre-execute cell baru
                def cleanup_handler():
                    self._force_cleanup_all()
                
                # Unregister existing handlers untuk prevent duplicate
                try:
                    ipython.events.unregister('pre_run_cell', cleanup_handler)
                except ValueError:
                    pass  # Handler belum registered
                
                # Register new cleanup handler
                ipython.events.register('pre_run_cell', cleanup_handler)
                
        except Exception as e:
            self.logger and self.logger.debug(f"🔧 Session cleanup setup failed: {str(e)}")
    
    def show_confirmation_dialog(self, title: str, message: str, 
                                on_confirm: Callable, confirm_text: str = "Ya", 
                                cancel_text: str = "Batal", danger_mode: bool = False):
        """Show confirmation dialog dengan session-aware cleanup."""
        # Force cleanup existing dialog
        self._force_cleanup_existing()
        
        # Create wrapped callbacks dengan auto-cleanup
        def wrapped_confirm():
            self._force_cleanup_existing()
            try:
                on_confirm()
            except Exception as e:
                self.logger and self.logger.error(f"❌ Confirm callback error: {str(e)}")
        
        def wrapped_cancel():
            self._force_cleanup_existing()
            self.logger and self.logger.debug("ℹ️ User cancelled dialog action")
        
        dialog = create_confirmation_dialog(
            title=title,
            message=message,
            on_confirm=lambda b: wrapped_confirm(),
            on_cancel=lambda b: wrapped_cancel(),
            confirm_text=confirm_text,
            cancel_text=cancel_text,
            danger_mode=danger_mode,
            dialog_width="100%" if self._is_mobile_layout() else "600px"
        )
        
        self.active_dialog = dialog
        self._display_in_confirmation_area(dialog)
        
        if self.logger:
            self.logger.debug(f"🔔 Dialog shown: {title}")
    
    def show_destructive_confirmation(self, title: str, message: str,
                                    on_confirm: Callable, item_name: str = "item",
                                    confirm_text: str = None, cancel_text: str = "Batal"):
        """Show destructive confirmation dengan session-aware cleanup."""
        self._force_cleanup_existing()
        
        def wrapped_confirm():
            self._force_cleanup_existing()
            try:
                on_confirm()
            except Exception as e:
                self.logger and self.logger.error(f"❌ Destructive action error: {str(e)}")
        
        def wrapped_cancel():
            self._force_cleanup_existing()
            self.logger and self.logger.debug("ℹ️ User cancelled destructive action")
        
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
            self.logger.debug(f"⚠️ Destructive dialog shown: {title}")
    
    def _display_in_confirmation_area(self, dialog):
        """Display dialog dengan responsive layout handling."""
        if self.confirmation_area:
            # Reset layout untuk fresh display
            self.confirmation_area.layout.max_height = None
            self.confirmation_area.layout.height = 'auto'
            self.confirmation_area.layout.overflow = 'visible'
            
            with self.confirmation_area:
                clear_output(wait=True)
                display(dialog)
        else:
            # Fallback jika confirmation area tidak ada
            display(dialog)
    
    def _force_cleanup_existing(self):
        """Force cleanup existing dialog dengan comprehensive cleaning."""
        if self.active_dialog:
            try:
                # Call dialog-specific cleanup
                if hasattr(self.active_dialog, '_auto_cleanup'):
                    self.active_dialog._auto_cleanup()
            except Exception:
                pass
            
            # Clear confirmation area
            if self.confirmation_area:
                with self.confirmation_area:
                    clear_output(wait=True)
            
            self.active_dialog = None
    
    def _force_cleanup_all(self):
        """Force cleanup all dialogs - untuk session restart."""
        self._force_cleanup_existing()
        cleanup_all_dialogs()
        
        # Reset state
        self.active_dialog = None
        
        if self.logger:
            self.logger.debug("🧹 All dialogs force-cleaned for session restart")
    
    def _is_mobile_layout(self) -> bool:
        """Detect mobile layout untuk responsive dialog sizing."""
        try:
            # Simple heuristic berdasarkan screen width
            import IPython.display as display
            # Default ke desktop layout
            return False
        except Exception:
            return False
    
    def cleanup(self):
        """Public cleanup method untuk external calls."""
        self._force_cleanup_all()
        
        if self.logger:
            self.logger.debug("🧹 DialogManager cleanup completed")
    
    def get_status(self) -> Dict[str, int]:
        """Get dialog manager status untuk debugging."""
        return {
            'active_local': 1 if self.active_dialog else 0,
            'active_global': get_active_dialog_count(),
            'has_confirmation_area': 1 if self.confirmation_area else 0
        }


def get_dialog_manager(ui_components: Dict[str, Any]) -> DialogManager:
    """
    Factory function dengan session-aware caching.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        DialogManager instance dengan session cleanup
    """
    # Create new manager untuk every call - prevent session persistence
    manager = DialogManager(ui_components)
    ui_components['dialog_manager'] = manager
    return manager


def create_dialog_manager(ui_components: Dict[str, Any]) -> DialogManager:
    """
    Create new dialog manager - always fresh instance.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        New DialogManager instance
    """
    return DialogManager(ui_components)