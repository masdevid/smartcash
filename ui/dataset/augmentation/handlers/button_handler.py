"""
File: smartcash/ui/dataset/augmentation/handlers/button_handler.py
Deskripsi: SRP button handler dengan safe button_state_manager initialization
"""

from typing import Dict, Any
from smartcash.ui.utils.button_state_manager import get_button_state_manager

class ButtonHandler:
    """SRP button handler dengan safe state management"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.manager = self._ensure_button_manager()
    
    def _ensure_button_manager(self):
        """Ensure button manager ada dan valid"""
        try:
            return get_button_state_manager(self.ui_components)
        except Exception as e:
            # Simple fallback manager inline
            return self._create_inline_fallback()
    
    def _create_inline_fallback(self):
        """Simple inline fallback tanpa import dependencies"""
        from contextlib import contextmanager
        
        class InlineFallback:
            def __init__(self, ui_components):
                self.ui_components = ui_components
            
            @contextmanager
            def operation_context(self, operation_name: str):
                # Disable operation buttons
                buttons = ['augment_button', 'check_button', 'cleanup_button']
                disabled = []
                
                try:
                    for btn_key in buttons:
                        btn = self.ui_components.get(btn_key)
                        if btn and hasattr(btn, 'disabled') and not btn.disabled:
                            btn.disabled = True
                            disabled.append(btn_key)
                    yield
                finally:
                    # Re-enable
                    for btn_key in disabled:
                        btn = self.ui_components.get(btn_key)
                        if btn and hasattr(btn, 'disabled'):
                            btn.disabled = False
            
            @contextmanager
            def config_context(self, config_operation: str):
                try:
                    yield
                finally:
                    pass
            
            def can_start_operation(self, operation_name: str):
                return True, "Inline fallback - allowed"
        
        return InlineFallback(self.ui_components)
    
    def execute_operation(self, operation_name: str, operation_func):
        """Execute operation dengan safe context"""
        try:
            # Check if can start
            can_start, message = self.manager.can_start_operation(operation_name)
            if not can_start:
                self._log_message(f"⚠️ {message}", 'warning')
                return
            
            # Clear logs dan progress
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
        """Reset UI state dengan safe approach"""
        try:
            # Clear logs - priority order
            for widget_key in ['log_output', 'status', 'output']:
                widget = self.ui_components.get(widget_key)
                if widget and hasattr(widget, 'clear_output'):
                    widget.clear_output(wait=True)
                    break
            
            # Reset progress
            tracker = self.ui_components.get('tracker')
            if tracker and hasattr(tracker, 'reset'):
                tracker.reset()
            elif 'reset_all' in self.ui_components:
                self.ui_components['reset_all']()
        except Exception:
            pass
    
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
    
    def _update_status(self, message: str, status_type: str):
        """Update status panel safely"""
        try:
            panel = self.ui_components.get('status_panel')
            if panel and hasattr(panel, 'value'):
                color_map = {'success': '#28a745', 'error': '#dc3545', 'warning': '#ffc107', 'info': '#007bff'}
                color = color_map.get(status_type, '#007bff')
                panel.value = f'<div style="color: {color}; padding: 8px;">{message}</div>'
        except Exception:
            pass

# Factory function
def create_button_handler(ui_components: Dict[str, Any]) -> ButtonHandler:
    """Factory untuk button handler dengan safe initialization"""
    return ButtonHandler(ui_components)