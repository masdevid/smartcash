"""
File: smartcash/ui/dataset/augmentation/handlers/confirmation_handler.py
Deskripsi: SRP confirmation dialog handler dengan safe approach
"""

from typing import Dict, Any, Callable
from IPython.display import display

class ConfirmationHandler:
    """SRP handler untuk confirmation dialogs"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
    
    def show_cleanup_confirmation(self, confirm_callback: Callable):
        """Show cleanup confirmation dialog"""
        self._show_dialog(
            title="Konfirmasi Cleanup Dataset",
            message="Apakah Anda yakin ingin menghapus semua file augmented?\n\n⚠️ Tindakan ini tidak dapat dibatalkan!",
            confirm_callback=confirm_callback,
            danger_mode=True
        )
    
    def show_reset_confirmation(self, confirm_callback: Callable):
        """Show reset confirmation dialog"""
        self._show_dialog(
            title="Konfirmasi Reset Konfigurasi",
            message="Apakah Anda yakin ingin reset konfigurasi ke default?\n\nSemua pengaturan saat ini akan hilang.",
            confirm_callback=confirm_callback,
            danger_mode=False
        )
    
    def _show_dialog(self, title: str, message: str, confirm_callback: Callable, danger_mode: bool = False):
        """Show confirmation dialog dengan safe approach"""
        try:
            from smartcash.ui.components.confirmation_dialog import create_confirmation_dialog
            
            dialog = create_confirmation_dialog(
                title=title,
                message=message,
                on_confirm=confirm_callback,
                on_cancel=lambda b: self._log_cancel(),
                danger_mode=danger_mode
            )
            
            # Show dalam confirmation area
            confirmation_area = self.ui_components.get('confirmation_area')
            if confirmation_area and hasattr(confirmation_area, 'clear_output'):
                confirmation_area.clear_output()
                with confirmation_area:
                    display(dialog)
            else:
                # Fallback: direct display
                display(dialog)
                
        except ImportError as e:
            self._log_error(f"❌ Cannot show dialog: {str(e)}")
    
    def _log_cancel(self):
        """Log cancel operation"""
        self._log_message("❌ Operasi dibatalkan", 'info')
    
    def _log_error(self, message: str):
        """Log error message"""
        self._log_message(message, 'error')
    
    def _log_message(self, message: str, level: str = 'info'):
        """Log message dengan safe approach"""
        try:
            # Priority 1: Existing logger
            logger = self.ui_components.get('logger')
            if logger and hasattr(logger, level):
                getattr(logger, level)(message)
                return
            
            # Priority 2: Direct to log widget
            widget = self.ui_components.get('log_output') or self.ui_components.get('status')
            if widget and hasattr(widget, 'clear_output'):
                from IPython.display import display, HTML
                color_map = {'info': '#007bff', 'success': '#28a745', 'warning': '#ffc107', 'error': '#dc3545'}
                color = color_map.get(level, '#007bff')
                html = f'<div style="color: {color}; margin: 2px 0; padding: 4px;">{message}</div>'
                
                with widget:
                    display(HTML(html))
                return
            
            # Fallback
            print(message)
        except Exception:
            print(message)

# Factory function
def create_confirmation_handler(ui_components: Dict[str, Any]) -> ConfirmationHandler:
    """Factory untuk confirmation handler"""
    return ConfirmationHandler(ui_components)