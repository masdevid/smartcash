"""
File: smartcash/ui/dataset/download/utils/button_state_manager.py
Deskripsi: Fixed progress bar visibility dengan explicit widget showing
"""

from typing import Dict, Any, List, Optional, Callable
from contextlib import contextmanager

class ButtonStateManager:
    """Manager untuk mengontrol state button dan progress dengan visible progress bars."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
        
        # Button groups untuk berbagai konteks
        self.button_groups = {
            'all': ['download_button', 'check_button', 'reset_button', 'cleanup_button', 'save_button'],
            'download': ['download_button', 'check_button', 'cleanup_button'],
            'check': ['download_button', 'check_button', 'cleanup_button'], 
            'cleanup': ['download_button', 'check_button', 'cleanup_button'],
            'save_reset': ['save_button', 'reset_button']
        }
        
        # Progress bar configurations untuk berbagai operasi
        self.progress_configs = {
            'download': {'overall': True, 'step': True, 'current': False},
            'check': {'overall': True, 'step': False, 'current': False},
            'cleanup': {'overall': True, 'step': False, 'current': True},
            'save': {'overall': False, 'step': False, 'current': False}
        }
    
    def disable_buttons(self, group: str = 'all', exclude: List[str] = None) -> None:
        """Disable button group dengan optional exclude."""
        exclude = exclude or []
        buttons = [btn for btn in self.button_groups.get(group, []) if btn not in exclude]
        
        for button_key in buttons:
            self._safe_set_button_state(button_key, disabled=True)
    
    def enable_buttons(self, group: str = 'all', only: List[str] = None) -> None:
        """Enable button group atau hanya button tertentu."""
        if only:
            buttons = only
        else:
            buttons = self.button_groups.get(group, [])
        
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
            if self.logger:
                self.logger.debug(f"🔘 Error setting {button_key} state: {str(e)}")
    
    def setup_progress_for_operation(self, operation: str) -> None:
        """Setup progress bars dengan explicit visibility."""
        config = self.progress_configs.get(operation, {'overall': True, 'step': False, 'current': False})
        
        # Show progress container
        self._show_progress_container()
        
        # Configure dan force show progress bars
        self._configure_and_show_progress_bars(config)
        
        # Reset progress values
        self._reset_progress_values()
    
    def _show_progress_container(self) -> None:
        """Show progress container dengan explicit widget visibility."""
        try:
            if 'progress_container' in self.ui_components:
                container_or_dict = self.ui_components['progress_container']
                
                # Method 1: Dict dengan show_container method
                if isinstance(container_or_dict, dict) and 'show_container' in container_or_dict:
                    container_or_dict['show_container']()
                    return
                
                # Method 2: Widget langsung
                if hasattr(container_or_dict, 'layout'):
                    container_or_dict.layout.visibility = 'visible'
                    container_or_dict.layout.display = 'block'
                    
                    # Force show semua child widgets juga
                    if hasattr(container_or_dict, 'children'):
                        for child in container_or_dict.children:
                            if hasattr(child, 'layout'):
                                child.layout.visibility = 'visible'
                                child.layout.display = 'block'
                    return
                
        except Exception as e:
            if self.logger:
                self.logger.debug(f"📦 Error showing progress container: {str(e)}")
    
    def _configure_and_show_progress_bars(self, config: Dict[str, bool]) -> None:
        """Configure dan force show progress bars yang diperlukan."""
        # Mapping widget keys berdasarkan progress_tracking structure
        progress_mapping = {
            'overall': ['overall_progress', 'progress_bar', 'overall_label'],
            'step': ['step_progress', 'step_label'], 
            'current': ['current_progress', 'current_label']
        }
        
        for progress_type, should_show in config.items():
            widgets = progress_mapping.get(progress_type, [])
            
            for widget_key in widgets:
                self._force_widget_visibility(widget_key, should_show)
    
    def _force_widget_visibility(self, widget_key: str, visible: bool) -> None:
        """Force widget visibility dengan explicit styling."""
        try:
            if widget_key in self.ui_components and self.ui_components[widget_key]:
                widget = self.ui_components[widget_key]
                
                if hasattr(widget, 'layout'):
                    if visible:
                        widget.layout.visibility = 'visible'
                        widget.layout.display = 'block'
                        
                        # Extra styling untuk progress bars
                        if 'progress' in widget_key and hasattr(widget, 'value'):
                            widget.layout.width = '100%'
                            widget.layout.height = '20px'
                            
                            # Force bar style untuk visibility
                            if hasattr(widget, 'bar_style'):
                                if widget.value == 0:
                                    widget.bar_style = ''  # Default bar style
                                else:
                                    widget.bar_style = 'info'
                                    
                            # Ensure widget style
                            if hasattr(widget, 'style'):
                                widget.style.bar_width = '100%'
                                
                    else:
                        widget.layout.visibility = 'hidden'
                        widget.layout.display = 'none'
                        
        except Exception as e:
            if self.logger:
                self.logger.debug(f"📊 Error setting {widget_key} visibility: {str(e)}")
    
    def _reset_progress_values(self) -> None:
        """Reset progress values dengan explicit bar showing."""
        # Widget mapping berdasarkan progress_tracking structure
        progress_widgets = [
            'overall_progress',  # Main overall progress
            'progress_bar',      # Alias untuk overall_progress 
            'step_progress',     # Step progress
            'current_progress'   # Current progress
        ]
        
        for widget_key in progress_widgets:
            self._safe_reset_and_show_progress_widget(widget_key)
        
        # Reset labels
        label_widgets = ['overall_label', 'step_label', 'current_label']
        for label_key in label_widgets:
            self._safe_reset_label_widget(label_key)
    
    def _safe_reset_and_show_progress_widget(self, widget_key: str) -> None:
        """Reset dan explicitly show progress widget."""
        try:
            if widget_key in self.ui_components and self.ui_components[widget_key]:
                widget = self.ui_components[widget_key]
                
                if hasattr(widget, 'value'):
                    widget.value = 0
                
                if hasattr(widget, 'layout'):
                    widget.layout.visibility = 'visible'
                    widget.layout.display = 'block'
                    widget.layout.width = '100%'
                    
                    # Specific height untuk setiap progress type
                    if 'overall' in widget_key or widget_key == 'progress_bar':
                        widget.layout.height = '25px'
                    elif 'step' in widget_key:
                        widget.layout.height = '20px'
                    elif 'current' in widget_key:
                        widget.layout.height = '15px'
                
                # Force bar style
                if hasattr(widget, 'bar_style'):
                    widget.bar_style = ''  # Reset to default
                    
                # Force style
                if hasattr(widget, 'style'):
                    widget.style.bar_width = '100%'
                    
        except Exception:
            pass
    
    def _safe_reset_label_widget(self, label_key: str) -> None:
        """Safely reset label widget."""
        try:
            if label_key in self.ui_components and self.ui_components[label_key]:
                widget = self.ui_components[label_key]
                if hasattr(widget, 'value'):
                    widget.value = ""
                if hasattr(widget, 'layout'):
                    widget.layout.visibility = 'visible'
        except Exception:
            pass
    
    def complete_operation(self, operation: str, message: str = "Selesai") -> None:
        """Complete operation dengan progress update ke 100%."""
        config = self.progress_configs.get(operation, {})
        
        # Update progress ke 100% dengan explicit showing
        if config.get('overall', False):
            self._safe_complete_progress('overall_progress', message)
            self._safe_complete_progress('progress_bar', message)  # Update alias
        if config.get('step', False):
            self._safe_complete_progress('step_progress', message)
        if config.get('current', False):
            self._safe_complete_progress('current_progress', message)
        
        # Enable buttons kembali
        self.enable_buttons('all')
    
    def _safe_complete_progress(self, widget_key: str, message: str) -> None:
        """Complete progress dengan explicit showing."""
        try:
            if widget_key in self.ui_components and self.ui_components[widget_key]:
                widget = self.ui_components[widget_key]
                
                if hasattr(widget, 'value'):
                    widget.value = 100
                    
                if hasattr(widget, 'bar_style'):
                    widget.bar_style = 'success'
                    
                if hasattr(widget, 'layout'):
                    widget.layout.visibility = 'visible'
                    widget.layout.display = 'block'
                    
        except Exception:
            pass
    
    def error_operation(self, operation: str, message: str = "Error") -> None:
        """Handle error state untuk operation."""
        config = self.progress_configs.get(operation, {})
        
        # Reset progress dengan error state
        if config.get('overall', False):
            self._safe_error_progress('overall_progress', message)
            self._safe_error_progress('progress_bar', message)
        
        # Enable buttons kembali
        self.enable_buttons('all')
    
    def _safe_error_progress(self, widget_key: str, message: str) -> None:
        """Set error state dengan visible widget."""
        try:
            if widget_key in self.ui_components and self.ui_components[widget_key]:
                widget = self.ui_components[widget_key]
                
                if hasattr(widget, 'value'):
                    widget.value = 0
                    
                if hasattr(widget, 'bar_style'):
                    widget.bar_style = 'danger'
                    
                if hasattr(widget, 'layout'):
                    widget.layout.visibility = 'visible'
                    widget.layout.display = 'block'
                    
        except Exception:
            pass
    
    @contextmanager
    def operation_context(self, operation: str, button_group: str = None):
        """Context manager dengan explicit progress bar showing."""
        button_group = button_group or operation
        
        try:
            # Setup untuk operasi
            self.disable_buttons(button_group)
            self.setup_progress_for_operation(operation)
            
            if self.logger:
                self.logger.debug(f"🚀 Started {operation} operation")
            
            yield self
            
            # Success cleanup
            self.complete_operation(operation, f"{operation.title()} selesai")
            
        except Exception as e:
            # Error cleanup
            self.error_operation(operation, str(e))
            raise
        
        finally:
            # Always enable buttons back
            self.enable_buttons('all')
            
            if self.logger:
                self.logger.debug(f"🏁 Finished {operation} operation")


def get_button_state_manager(ui_components: Dict[str, Any]) -> ButtonStateManager:
    """Factory function untuk ButtonStateManager."""
    return ButtonStateManager(ui_components)


# Backward compatibility functions (deprecated)
def disable_download_buttons(ui_components: Dict[str, Any], disabled: bool) -> None:
    """Legacy function untuk backward compatibility.""" 
    manager = get_button_state_manager(ui_components)
    if disabled:
        manager.disable_buttons('all')
    else:
        manager.enable_buttons('all')