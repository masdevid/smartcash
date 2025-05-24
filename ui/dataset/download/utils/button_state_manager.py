"""
File: smartcash/ui/dataset/download/utils/button_state_manager.py
Deskripsi: Fixed button state manager dengan proper progress_tracking integration
"""

from typing import Dict, Any, List, Optional, Callable
from contextlib import contextmanager

class ButtonStateManager:
    """Manager untuk mengontrol state button dan progress dengan proper integration."""
    
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
        """Setup progress bars sesuai konfigurasi operasi dengan proper progress_tracking integration."""
        config = self.progress_configs.get(operation, {'overall': True, 'step': False, 'current': False})
        
        # Show progress container menggunakan method dari progress_tracking
        self._show_progress_container()
        
        # Configure visibility berdasarkan operation
        self._configure_progress_visibility(config)
        
        # Reset progress values
        self._reset_progress_values()
    
    def _show_progress_container(self) -> None:
        """Show progress container menggunakan method dari progress_tracking."""
        try:
            # Method 1: Direct progress_container dengan show_container method
            if 'progress_container' in self.ui_components:
                container_or_dict = self.ui_components['progress_container']
                
                # Jika container adalah dict dengan method show_container
                if isinstance(container_or_dict, dict) and 'show_container' in container_or_dict:
                    container_or_dict['show_container']()
                    return
                
                # Jika container adalah widget langsung
                if hasattr(container_or_dict, 'layout'):
                    container_or_dict.layout.visibility = 'visible'
                    container_or_dict.layout.display = 'block'
                    return
            
            # Method 2: Import dan gunakan progress_tracking functions
            try:
                from smartcash.ui.components.progress_tracking import _show_progress_container
                _show_progress_container(self.ui_components.get('progress_container'))
            except ImportError:
                pass
                
        except Exception as e:
            if self.logger:
                self.logger.debug(f"📦 Error showing progress container: {str(e)}")
    
    def _configure_progress_visibility(self, config: Dict[str, bool]) -> None:
        """Configure visibility progress bars dengan mapping yang tepat."""
        # Mapping widget keys berdasarkan progress_tracking structure
        progress_mapping = {
            'overall': ['overall_progress', 'progress_bar', 'overall_label'],  # progress_bar = alias
            'step': ['step_progress', 'step_label'], 
            'current': ['current_progress', 'current_label']
        }
        
        for progress_type, should_show in config.items():
            widgets = progress_mapping.get(progress_type, [])
            visibility = 'visible' if should_show else 'hidden'
            display = 'block' if should_show else 'none'
            
            for widget_key in widgets:
                self._safe_set_widget_visibility(widget_key, visibility, display)
    
    def _safe_set_widget_visibility(self, widget_key: str, visibility: str, display: str) -> None:
        """Safely set widget visibility dengan proper error handling."""
        try:
            if widget_key in self.ui_components and self.ui_components[widget_key]:
                widget = self.ui_components[widget_key]
                if hasattr(widget, 'layout'):
                    widget.layout.visibility = visibility
                    widget.layout.display = display
        except Exception as e:
            if self.logger:
                self.logger.debug(f"📊 Error setting {widget_key} visibility: {str(e)}")
    
    def _hide_progress_container(self) -> None:
        """Hide progress container menggunakan method dari progress_tracking."""
        try:
            # Method 1: Direct progress_container dengan hide_container method
            if 'progress_container' in self.ui_components:
                container_or_dict = self.ui_components['progress_container']
                
                # Jika container adalah dict dengan method hide_container
                if isinstance(container_or_dict, dict) and 'hide_container' in container_or_dict:
                    container_or_dict['hide_container']()
                    return
                
                # Jika container adalah widget langsung
                if hasattr(container_or_dict, 'layout'):
                    container_or_dict.layout.visibility = 'hidden' 
                    container_or_dict.layout.display = 'none'
                    return
            
            # Method 2: Import dan gunakan progress_tracking functions
            try:
                from smartcash.ui.components.progress_tracking import _hide_progress_container
                _hide_progress_container(self.ui_components.get('progress_container'))
            except ImportError:
                pass
                
        except Exception as e:
            if self.logger:
                self.logger.debug(f"📦 Error hiding progress container: {str(e)}")
    
    def _reset_progress_values(self) -> None:
        """Reset semua progress values ke 0 dengan proper widget mapping."""
        # Widget mapping berdasarkan progress_tracking structure
        progress_widgets = [
            'overall_progress',  # Main overall progress
            'progress_bar',      # Alias untuk overall_progress 
            'step_progress',     # Step progress
            'current_progress'   # Current progress
        ]
        
        for widget_key in progress_widgets:
            self._safe_reset_progress_widget(widget_key)
        
        # Reset labels juga
        label_widgets = ['overall_label', 'step_label', 'current_label']
        for label_key in label_widgets:
            self._safe_reset_label_widget(label_key)
    
    def _safe_reset_progress_widget(self, widget_key: str) -> None:
        """Safely reset progress widget."""
        try:
            if widget_key in self.ui_components and self.ui_components[widget_key]:
                widget = self.ui_components[widget_key]
                if hasattr(widget, 'value'):
                    widget.value = 0
                if hasattr(widget, 'description'):
                    widget.description = "Progress: 0%"
        except Exception:
            pass
    
    def _safe_reset_label_widget(self, label_key: str) -> None:
        """Safely reset label widget."""
        try:
            if label_key in self.ui_components and self.ui_components[label_key]:
                widget = self.ui_components[label_key]
                if hasattr(widget, 'value'):
                    widget.value = ""
        except Exception:
            pass
    
    def complete_operation(self, operation: str, message: str = "Selesai") -> None:
        """Complete operation dengan cleanup state dan progress update."""
        # Update progress ke 100% untuk operation ini
        config = self.progress_configs.get(operation, {})
        if config.get('overall', False):
            self._safe_update_progress('overall_progress', 100, message)
            self._safe_update_progress('progress_bar', 100, message)  # Update alias juga
        if config.get('step', False):
            self._safe_update_progress('step_progress', 100, message)
        if config.get('current', False):
            self._safe_update_progress('current_progress', 100, message)
        
        # Enable buttons kembali
        self.enable_buttons('all')
    
    def error_operation(self, operation: str, message: str = "Error") -> None:
        """Handle error state untuk operation."""
        # Reset progress dengan error message
        config = self.progress_configs.get(operation, {})
        if config.get('overall', False):
            self._safe_update_progress('overall_progress', 0, f"❌ {message}")
            self._safe_update_progress('progress_bar', 0, f"❌ {message}")
        
        # Enable buttons kembali
        self.enable_buttons('all')
    
    def _safe_update_progress(self, widget_key: str, value: int, description: str) -> None:
        """Safely update progress widget dengan proper handling."""
        try:
            if widget_key in self.ui_components and self.ui_components[widget_key]:
                widget = self.ui_components[widget_key]
                if hasattr(widget, 'value'):
                    widget.value = value
                if hasattr(widget, 'description'):
                    widget.description = description
        except Exception:
            pass
    
    @contextmanager
    def operation_context(self, operation: str, button_group: str = None):
        """Context manager untuk mengatur button state dan progress selama operasi."""
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