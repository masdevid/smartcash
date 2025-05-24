"""
File: smartcash/ui/dataset/download/utils/button_state_manager.py
Deskripsi: Enhanced button state manager dengan tqdm progress integration
"""

from typing import Dict, Any, List, Optional
from contextlib import contextmanager

class ButtonStateManager:
    """Enhanced manager untuk mengontrol button state dan tqdm progress."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
        
        self.button_groups = {
            'all': ['download_button', 'check_button', 'reset_button', 'cleanup_button', 'save_button'],
            'download': ['download_button', 'check_button', 'cleanup_button'],
            'check': ['download_button', 'check_button', 'cleanup_button'], 
            'cleanup': ['download_button', 'check_button', 'cleanup_button'],
            'save_reset': ['save_button', 'reset_button']
        }
    
    def disable_buttons(self, group: str = 'all', exclude: List[str] = None) -> None:
        """Disable button group dengan optional exclude."""
        exclude = exclude or []
        buttons = [btn for btn in self.button_groups.get(group, []) if btn not in exclude]
        
        for button_key in buttons:
            self._safe_set_button_state(button_key, disabled=True)
    
    def enable_buttons(self, group: str = 'all', only: List[str] = None) -> None:
        """Enable button group atau hanya button tertentu."""
        buttons = only if only else self.button_groups.get(group, [])
        
        for button_key in buttons:
            self._safe_set_button_state(button_key, disabled=False)
    
    def _safe_set_button_state(self, button_key: str, disabled: bool) -> None:
        """Safely set button state dengan error handling."""
        try:
            if button_key in self.ui_components and self.ui_components[button_key]:
                button = self.ui_components[button_key]
                if hasattr(button, 'disabled'):
                    button.disabled = disabled
        except Exception as e:
            self.logger and self.logger.debug(f"🔘 Error setting {button_key} state: {str(e)}")
    
    def setup_progress_for_operation(self, operation: str) -> None:
        """Setup tqdm progress bars untuk operation."""
        if 'show_for_operation' in self.ui_components:
            self.ui_components['show_for_operation'](operation)
        else:
            self.logger and self.logger.warning("⚠️ tqdm progress tracking tidak tersedia")
    
    def complete_operation(self, operation: str, message: str = "Selesai") -> None:
        """Complete operation dengan tqdm progress update."""
        if 'complete_operation' in self.ui_components:
            self.ui_components['complete_operation'](message)
        
        self.enable_buttons('all')
    
    def error_operation(self, operation: str, message: str = "Error") -> None:
        """Handle error state untuk operation dengan tqdm."""
        if 'error_operation' in self.ui_components:
            self.ui_components['error_operation'](message)
        
        self.enable_buttons('all')
    
    @contextmanager
    def operation_context(self, operation: str, button_group: str = None):
        """Context manager dengan tqdm progress integration."""
        button_group = button_group or operation
        
        try:
            self.disable_buttons(button_group)
            self.setup_progress_for_operation(operation)
            
            self.logger and self.logger.debug(f"🚀 Started {operation} operation")
            
            yield self
            
            self.complete_operation(operation, f"{operation.title()} selesai")
            
        except Exception as e:
            self.error_operation(operation, str(e))
            raise
        
        finally:
            self.enable_buttons('all')
            self.logger and self.logger.debug(f"🏁 Finished {operation} operation")


def get_button_state_manager(ui_components: Dict[str, Any]) -> ButtonStateManager:
    """Factory function untuk ButtonStateManager."""
    return ButtonStateManager(ui_components)


def disable_download_buttons(ui_components: Dict[str, Any], disabled: bool) -> None:
    """Legacy function untuk backward compatibility.""" 
    manager = get_button_state_manager(ui_components)
    if disabled:
        manager.disable_buttons('all')
    else:
        manager.enable_buttons('all')