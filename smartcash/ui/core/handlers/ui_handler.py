"""
File: smartcash/ui/core/handlers/ui_handler.py

UI Handler - Container-Aware Updates untuk layout berbasis container.
"""

from typing import Dict, Any, Optional, List, Callable
import logging
from abc import ABC

from smartcash.ui.core.handlers.base_handler import BaseHandler


from IPython.display import display

class UIHandler(BaseHandler):
    """Handler dengan UI-specific utilities yang container-aware."""
    
    def __init__(self, module_name: str, parent_module: str = None):
        super().__init__(module_name, parent_module)
        self.ui_components = {}
        
    def _find_component(self, component_type: str, attribute_name: str = None) -> Optional[Any]:
        """Find component dalam container hierarchy.
        
        Args:
            component_type: Type/name component yang dicari
            attribute_name: Attribute name jika berbeda dari component_type
            
        Returns:
            Component instance atau None
        """
        attr_name = attribute_name or component_type
        
        # Direct access
        if component_type in self.ui_components:
            return self.ui_components[component_type]
        
        # Container-based search
        container_keys = ['main_container', 'summary_container', 'action_container', 'form_container', 'footer_container']
        
        for container_key in container_keys:
            if container_key in self.ui_components:
                container = self.ui_components[container_key]
                
                # Check container attributes
                if hasattr(container, attr_name):
                    return getattr(container, attr_name)
                
                # Check container content
                if hasattr(container, 'content') and hasattr(container.content, attr_name):
                    return getattr(container.content, attr_name)
                
                # Check container children
                if hasattr(container, 'children'):
                    for child in container.children:
                        if hasattr(child, attr_name):
                            return getattr(child, attr_name)
                        # Check by class name
                        if getattr(child, '__class__', None).__name__ == component_type:
                            return child
        
        return None
    
    def update_status(self, message: str, status_type: str = 'info'):
        """Update status panel dengan container-aware access."""
        try:
            status_panel = self._find_component('status_panel', 'status_panel')
            if status_panel and hasattr(status_panel, 'update'):
                status_panel.update(message, status_type)
                return
            
            # Fallback: log status
            self.logger.info(f"📢 Status: {message} ({status_type})")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update status: {str(e)}")
    
    def update_progress(self, value: float, message: str = None):
        """Update progress tracker dengan container-aware access."""
        try:
            progress_tracker = self._find_component('ProgressTracker', 'progress_tracker')
            if progress_tracker and hasattr(progress_tracker, 'update'):
                progress_tracker.update(value, message)
                return
            
            # Fallback: log progress
            self.logger.info(f"📊 Progress: {value*100:.1f}% - {message or 'Processing...'}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update progress: {str(e)}")
    
    def log_message(self, message: str, level: str = 'info'):
        """Log message dengan container-aware access."""
        try:
            log_component = self._find_component('log_accordion', 'log_output')
            if log_component:
                self._log_to_component(log_component, message, level)
                return
            
            # Fallback: standard logging
            getattr(self.logger, level.lower(), self.logger.info)(message)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log message: {str(e)}")
    
    def _log_to_component(self, log_component: Any, message: str, level: str):
        """Log message ke component."""
        if hasattr(log_component, 'log') and callable(log_component.log):
            log_component.log(message, level)
        elif hasattr(log_component, 'append_stdout') and callable(log_component.append_stdout):
            log_component.append_stdout(f"[{level.upper()}] {message}\n")
        elif hasattr(log_component, 'value'):
            log_component.value += f"<div>[{level.upper()}] {message}</div>"
        else:
            self.logger.warning(f"Log component has no supported method")
    
    def show_dialog(self, title: str, message: str, dialog_type: str = 'info'):
        """Show dialog dengan container-aware access."""
        try:
            dialog_area = self._find_component('confirmation_area', 'dialog_area')
            if dialog_area:
                self._show_dialog(dialog_area, title, message, dialog_type)
                return
            
            # Fallback: log dialog
            self.logger.info(f"💬 Dialog: {title} - {message}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to show dialog: {str(e)}")
    
    def _show_dialog(self, dialog_component: Any, title: str, message: str, dialog_type: str):
        """Show dialog pada component."""
        if hasattr(dialog_component, 'show_dialog') and callable(dialog_component.show_dialog):
            dialog_component.show_dialog(title, message, dialog_type)
        elif hasattr(dialog_component, 'value'):
            dialog_component.value = f"<div><h4>{title}</h4><p>{message}</p></div>"
        else:
            self.logger.warning(f"Dialog component has no supported method")
    
    def update_summary(self, content: str):
        """Update summary dengan container-aware access."""
        try:
            summary_widget = self._find_component('setup_summary', 'summary_container')
            if summary_widget:
                self._set_summary_content(summary_widget, content)
                return
            
            # Fallback: log summary
            self.logger.info(f"📋 Summary: {content}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update summary: {str(e)}")
    
    def _set_summary_content(self, summary_widget: Any, content: str):
        """Set content pada summary widget."""
        if hasattr(summary_widget, 'value'):
            summary_widget.value = content
        elif hasattr(summary_widget, 'set_content'):
            summary_widget.set_content(content)
        else:
            self.logger.warning(f"Summary widget has no supported method")
    
    def enable_button(self, button_name: str, enabled: bool = True):
        """Enable/disable button dengan container-aware access."""
        try:
            button = self._find_component(button_name)
            if button and hasattr(button, 'disabled'):
                button.disabled = not enabled
                return
            
            # Fallback: log button state
            self.logger.info(f"🔘 Button {button_name}: {'enabled' if enabled else 'disabled'}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update button: {str(e)}")
    
    def reset_component(self, component_type: str):
        """Reset component dengan container-aware access."""
        try:
            component = self._find_component(component_type)
            if component:
                # Try different reset methods
                if hasattr(component, 'reset'):
                    component.reset()
                elif hasattr(component, 'clear'):
                    component.clear()
                elif hasattr(component, 'clear_output'):
                    component.clear_output(wait=True)
                elif hasattr(component, 'value'):
                    component.value = ""
                return
            
            self.logger.warning(f"Component {component_type} not found for reset")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to reset component: {str(e)}")
    
    def clear_all_components(self):
        """Clear all UI components safely."""
        try:
            components_to_clear = ['progress_tracker', 'status_panel', 'log_accordion', 'confirmation_area']
            
            for component_type in components_to_clear:
                self.reset_component(component_type)
                
            self.logger.info("🧹 All UI components cleared")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to clear all components: {str(e)}")
    
    def update_component_safely(self, component_type: str, update_func: Callable):
        """Update component dengan safe execution."""
        try:
            component = self._find_component(component_type)
            if component:
                update_func(component)
            else:
                self.logger.warning(f"Component {component_type} not found")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to update component {component_type}: {str(e)}")
    
    def get_component_status(self, component_type: str) -> Dict[str, Any]:
        """Get status component dengan container-aware access."""
        try:
            component = self._find_component(component_type)
            if component:
                return {
                    'exists': True,
                    'type': type(component).__name__,
                    'has_value': hasattr(component, 'value'),
                    'has_update': hasattr(component, 'update'),
                    'has_reset': hasattr(component, 'reset')
                }
            else:
                return {
                    'exists': False,
                    'type': None
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to get component status: {str(e)}")
            return {
                'exists': False,
                'error': str(e)
            }
    
    def show_confirmation(self, title: str, message: str, on_confirm: Callable, on_cancel: Callable = None):
        """Show confirmation dialog dengan container-aware access."""
        try:
            # Try to use dialog component
            dialog_area = self._find_component('confirmation_area', 'dialog_area')
            if dialog_area and hasattr(dialog_area, 'show_confirmation'):
                dialog_area.show_confirmation(title, message, on_confirm, on_cancel)
                return
            
            # Fallback: direct confirmation
            self.logger.info(f"💬 Confirmation: {title} - {message}")
            if on_confirm:
                on_confirm()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to show confirmation: {str(e)}")
    
    def update_widget_value(self, widget_name: str, value: Any):
        """Update widget value dengan container-aware access."""
        try:
            widget = self._find_component(widget_name)
            if widget and hasattr(widget, 'value'):
                widget.value = value
                return
            
            self.logger.warning(f"Widget {widget_name} not found or has no value attribute")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update widget value: {str(e)}")
    
    def get_widget_value(self, widget_name: str) -> Any:
        """Get widget value dengan container-aware access."""
        try:
            widget = self._find_component(widget_name)
            if widget and hasattr(widget, 'value'):
                return widget.value
            
            self.logger.warning(f"Widget {widget_name} not found or has no value attribute")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get widget value: {str(e)}")
            return None
    
    def show_error_ui(self, error_result: Dict[str, Any]):
        """Display error UI and hide standard components.

        Args:
            error_result: Dict returned from operations that includes
                ``error`` bool, ``message`` string, and optionally ``container`` widget
                or traceback information.
        """
        if not error_result:
            return
            
        try:
            # Hide normal components
            self.clear_all_components()
            
            # Use the centralized error handler to create and display the error UI
            from smartcash.ui.core.shared.error_handler import get_error_handler
            error_handler = get_error_handler(self.full_module_name)
            error_handler.create_error_ui(error_result)
            
        except Exception as e:
            # Fall back to logging if even error UI fails
            self.logger.error(f"❌ Failed to display error UI: {e}")
            import traceback
            self.logger.debug(f"Error details: {traceback.format_exc()}")

    def cleanup(self):
        """Cleanup UI handler."""
        try:
            self.clear_all_components()
            self.ui_components.clear()
            super().cleanup()
            self.logger.debug(f"🧹 Cleaned up UIHandler for {self.full_module_name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup UIHandler: {str(e)}")


class ModuleUIHandler(UIHandler):
    """Module-specific UI handler dengan config integration."""
    
    def __init__(self, module_name: str, parent_module: str = None):
        super().__init__(module_name, parent_module)
        self._status_history = []
        self._progress_history = []
    
    def track_status(self, message: str, status_type: str = 'info'):
        """Track dan update status dengan history."""
        self._status_history.append({
            'message': message,
            'type': status_type,
            'timestamp': self._get_timestamp()
        })
        self.update_status(message, status_type)
    
    def track_progress(self, value: float, message: str = None):
        """Track dan update progress dengan history."""
        self._progress_history.append({
            'value': value,
            'message': message,
            'timestamp': self._get_timestamp()
        })
        self.update_progress(value, message)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def get_status_history(self) -> List[Dict[str, Any]]:
        """Get status history."""
        return self._status_history.copy()
    
    def get_progress_history(self) -> List[Dict[str, Any]]:
        """Get progress history."""
        return self._progress_history.copy()
    
    def clear_history(self):
        """Clear status dan progress history."""
        self._status_history.clear()
        self._progress_history.clear()
    
    def get_ui_state(self) -> Dict[str, Any]:
        """Get comprehensive UI state."""
        return {
            'status_history': self.get_status_history(),
            'progress_history': self.get_progress_history(),
            'components_status': {
                component_type: self.get_component_status(component_type)
                for component_type in ['progress_tracker', 'status_panel', 'log_accordion', 'confirmation_area']
            }
        }
    
    def restore_ui_state(self, state: Dict[str, Any]):
        """Restore UI state dari saved state."""
        try:
            if 'status_history' in state:
                self._status_history = state['status_history']
            if 'progress_history' in state:
                self._progress_history = state['progress_history']
                
            # Restore latest status dan progress
            if self._status_history:
                latest_status = self._status_history[-1]
                self.update_status(latest_status['message'], latest_status['type'])
                
            if self._progress_history:
                latest_progress = self._progress_history[-1]
                self.update_progress(latest_progress['value'], latest_progress['message'])
                
        except Exception as e:
            self.logger.error(f"❌ Failed to restore UI state: {str(e)}")