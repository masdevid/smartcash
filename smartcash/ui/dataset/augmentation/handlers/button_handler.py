"""
File: smartcash/ui/dataset/augmentation/handlers/button_handler.py
Deskripsi: SRP button handler using shared button_state_manager dengan one-liner style
"""

from typing import Dict, Any
from smartcash.ui.utils.button_state_manager import get_button_state_manager

class ButtonHandler:
    """SRP button handler using shared button_state_manager"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.manager = self._get_or_create_manager()
    
    def _get_or_create_manager(self):
        """Get existing or create new button_state_manager"""
        # Check if already exists in ui_components
        existing_manager = self.ui_components.get('button_state_manager')
        if existing_manager and hasattr(existing_manager, 'can_start_operation'):
            return existing_manager
        
        # Create new shared manager
        manager = get_button_state_manager(self.ui_components)
        self.ui_components['button_state_manager'] = manager
        return manager
    
    def execute_operation(self, operation_name: str, operation_func):
        """Execute operation dengan safe context dan error handling"""
        try:
            # Check if can start - one-liner validation
            can_start, message = self.manager.can_start_operation(operation_name)
            not can_start and self._log_message(f"⚠️ {message}", 'warning') or None
            if not can_start: return
            
            # Reset UI state
            self._reset_ui_state()
            
            # Execute dengan context
            with self.manager.operation_context(operation_name):
                operation_func()
                self._log_message(f"✅ {operation_name.title()} berhasil", 'success')
                
        except Exception as e:
            error_msg = f"❌ {operation_name.title()} gagal: {str(e)}"
            self._log_message(error_msg, 'error')
            self._update_status(error_msg, 'error')
    
    def execute_config(self, config_name: str, config_func):
        """Execute config operation dengan safe context"""
        try:
            self._reset_ui_state()
            
            with self.manager.config_context(config_name):
                config_func()
                self._log_message(f"✅ {config_name.title()} berhasil", 'success')
                
        except Exception as e:
            error_msg = f"❌ {config_name.title()} gagal: {str(e)}"
            self._log_message(error_msg, 'error')
    
    def _reset_ui_state(self):
        """One-liner UI state reset dengan priority order"""
        [widget.clear_output(wait=True) for widget_key in ['log_output', 'status', 'output'] 
         if (widget := self.ui_components.get(widget_key)) and hasattr(widget, 'clear_output')][:1] or None
        
        # Reset progress tracker
        tracker = self.ui_components.get('tracker')
        hasattr(tracker, 'reset') and tracker.reset() or self.ui_components.get('reset_all', lambda: None)()
    
    def _log_message(self, message: str, level: str = 'info'):
        """One-liner log message dengan fallback chain"""
        logger = self.ui_components.get('logger')
        
        # Try logger first
        if logger and hasattr(logger, level):
            getattr(logger, level)(message)
            return
        
        # Try direct widget display
        widget = self.ui_components.get('log_output') or self.ui_components.get('status')
        if widget and hasattr(widget, 'clear_output'):
            try:
                from IPython.display import display, HTML
                color = {'info': '#007bff', 'success': '#28a745', 'warning': '#ffc107', 'error': '#dc3545'}.get(level, '#007bff')
                with widget: display(HTML(f'<div style="color: {color}; margin: 2px 0; padding: 4px;">{message}</div>'))
                return
            except Exception:
                pass
        
        # Fallback print
        print(message)
    
    def _update_status(self, message: str, status_type: str):
        """One-liner status panel update"""
        panel = self.ui_components.get('status_panel')
        if panel and hasattr(panel, 'value'):
            color = {'success': '#28a745', 'error': '#dc3545', 'warning': '#ffc107', 'info': '#007bff'}.get(status_type, '#007bff')
            panel.value = f'<div style="color: {color}; padding: 8px;">{message}</div>'

# One-liner factory dengan guaranteed complete manager
create_button_handler = lambda ui_components: ButtonHandler(ui_components)