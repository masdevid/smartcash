"""
File: smartcash/ui/dataset/augmentation/handlers/button_handler.py
Deskripsi: Fixed button handler yang tidak clear seluruh UI saat operation
"""

from typing import Dict, Any
from smartcash.ui.utils.button_state_manager import get_button_state_manager

class ButtonHandler:
    """Fixed button handler tanpa excessive UI clearing"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.manager = self._get_or_create_manager()
    
    def _get_or_create_manager(self):
        """Get existing or create button state manager"""
        existing_manager = self.ui_components.get('button_state_manager')
        if existing_manager and hasattr(existing_manager, 'can_start_operation'):
            return existing_manager
        
        manager = get_button_state_manager(self.ui_components)
        self.ui_components['button_state_manager'] = manager
        return manager
    
    def execute_operation(self, operation_name: str, operation_func):
        """Execute operation tanpa excessive UI clearing"""
        try:
            can_start, message = self.manager.can_start_operation(operation_name)
            if not can_start:
                self._log_message(f"⚠️ {message}", 'warning')
                return
            
            # Reset hanya log area, bukan seluruh UI
            self._reset_log_area_only()
            
            with self.manager.operation_context(operation_name):
                operation_func()
                self._log_message(f"✅ {operation_name.title()} berhasil", 'success')
                
        except Exception as e:
            error_msg = f"❌ {operation_name.title()} gagal: {str(e)}"
            self._log_message(error_msg, 'error')
    
    def execute_config(self, config_name: str, config_func):
        """Execute config operation tanpa UI clearing"""
        try:
            with self.manager.config_context(config_name):
                config_func()
                self._log_message(f"✅ {config_name.title()} berhasil", 'success')
                
        except Exception as e:
            error_msg = f"❌ {config_name.title()} gagal: {str(e)}"
            self._log_message(error_msg, 'error')
    
    def _reset_log_area_only(self):
        """Reset hanya log output area, bukan seluruh UI"""
        # Reset progress tracker
        tracker = self.ui_components.get('tracker')
        if tracker and hasattr(tracker, 'reset'):
            tracker.reset()
        elif 'reset_all' in self.ui_components:
            self.ui_components['reset_all']()
    
    def _log_message(self, message: str, level: str = 'info'):
        """Log message tanpa clear output"""
        logger = self.ui_components.get('logger')
        
        # Try logger first
        if logger and hasattr(logger, level):
            getattr(logger, level)(message)
            return
        
        # Try direct widget display tanpa clear
        widget = self.ui_components.get('log_output') or self.ui_components.get('status')
        if widget and hasattr(widget, 'append_outputs'):
            try:
                from IPython.display import display, HTML
                color = {'info': '#007bff', 'success': '#28a745', 'warning': '#ffc107', 'error': '#dc3545'}.get(level, '#007bff')
                html = f'<div style="color: {color}; margin: 2px 0; padding: 4px;">{message}</div>'
                widget.append_outputs((HTML(html),))
                return
            except Exception:
                pass
        
        # Fallback print
        print(message)

# Factory function
create_button_handler = lambda ui_components: ButtonHandler(ui_components)