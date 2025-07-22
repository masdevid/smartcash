"""
Display mixin for UI modules.

Provides standard display functionality and UI component management.
"""

from typing import Dict, Any, Optional, Union
from smartcash.ui.utils.display_utils import safe_display


class DisplayMixin:
    """
    Mixin providing common display functionality.
    
    This mixin provides:
    - Standard UI component display
    - Component creation and management
    - Display state management
    - Error handling for display operations
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._display_state: Dict[str, Any] = {
            'displayed': False,
            'main_component': None,
            'last_display_result': None
        }
    
    def get_ui_components(self) -> Dict[str, Any]:
        """
        Get UI components dictionary.
        
        Returns:
            UI components dictionary or error dict
        """
        try:
            if not getattr(self, '_is_initialized', False):
                if hasattr(self, 'initialize'):
                    if not self.initialize():
                        return {'error': 'Failed to initialize module'}
                else:
                    return {'error': 'Module not initialized'}
            
            if not hasattr(self, '_ui_components') or not self._ui_components:
                return {'error': 'UI components not available'}
            
            return self._ui_components.copy()
            
        except Exception as e:
            return {'error': f'Failed to get UI components: {str(e)}'}
    
    def get_main_widget(self) -> Optional[Any]:
        """
        Get the main widget for display.
        
        Returns:
            Main widget or None
        """
        try:
            components = self.get_ui_components()
            
            if 'error' in components:
                return None
            
            # Use standardized main_container key only
            if 'main_container' in components and components['main_container'] is not None:
                return components['main_container']
            
            # Try to find any displayable component
            for key, component in components.items():
                if component is not None and hasattr(component, '_ipython_display_'):
                    return component
            
            return None
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to get main widget: {e}")
            return None
    
    def display_ui(self, clear_output: bool = True) -> Optional[Dict[str, Any]]:
        """
        Display the UI components.
        
        Args:
            clear_output: Whether to clear output before displaying
            
        Returns:
            Display result dictionary or None
        """
        try:
            if clear_output:
                try:
                    from IPython.display import clear_output as ipy_clear_output
                    ipy_clear_output(wait=True)
                except ImportError:
                    pass
            
            main_widget = self.get_main_widget()
            
            if main_widget is None:
                error_msg = "No main widget available for display"
                # Use self.log() method if available (supports buffering), otherwise use logger
                if hasattr(self, 'log') and hasattr(self, '_ui_components') and self._ui_components and 'operation_container' in self._ui_components:
                    self.log(error_msg, 'error')
                elif hasattr(self, 'logger'):
                    self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            # Display the widget
            try:
                safe_display(main_widget)
                
                # Update display state
                self._display_state.update({
                    'displayed': True,
                    'main_component': main_widget,
                    'last_display_result': {'success': True}
                })
                
                # Use self.log() method if available (supports buffering), otherwise suppress
                if hasattr(self, 'log') and hasattr(self, '_ui_components') and self._ui_components and 'operation_container' in self._ui_components:
                    self.log("✅ UI displayed successfully", 'info')
                elif hasattr(self, 'logger'):
                    # Use debug level to avoid console spam during testing
                    self.logger.debug("✅ UI displayed successfully")
                
                return {'success': True, 'message': 'UI displayed successfully'}
                
            except Exception as display_error:
                error_msg = f"Failed to display UI: {str(display_error)}"
                # Use self.log() method if available (supports buffering), otherwise use logger
                if hasattr(self, 'log') and hasattr(self, '_ui_components') and self._ui_components and 'operation_container' in self._ui_components:
                    self.log(error_msg, 'error')
                elif hasattr(self, 'logger'):
                    self.logger.error(error_msg)
                
                self._display_state['last_display_result'] = {
                    'success': False,
                    'error': str(display_error)
                }
                
                return {'success': False, 'message': error_msg}
                
        except Exception as e:
            error_msg = f"Display operation failed: {str(e)}"
            # Use self.log() method if available (supports buffering), otherwise use logger
            if hasattr(self, 'log') and hasattr(self, '_ui_components') and self._ui_components and 'operation_container' in self._ui_components:
                self.log(error_msg, 'error')
            elif hasattr(self, 'logger'):
                self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def is_displayed(self) -> bool:
        """
        Check if UI is currently displayed.
        
        Returns:
            True if UI is displayed
        """
        return self._display_state.get('displayed', False)
    
    def get_display_state(self) -> Dict[str, Any]:
        """
        Get current display state.
        
        Returns:
            Display state dictionary
        """
        return self._display_state.copy()
    
    def clear_display(self) -> None:
        """Clear the display."""
        try:
            from IPython.display import clear_output
            clear_output(wait=True)
            
            self._display_state.update({
                'displayed': False,
                'main_component': None
            })
            
        except ImportError:
            pass
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to clear display: {e}")
    
    def refresh_display(self) -> Optional[Dict[str, Any]]:
        """
        Refresh the display.
        
        Returns:
            Refresh result dictionary
        """
        try:
            # Clear current display
            self.clear_display()
            
            # Re-display UI
            return self.display_ui(clear_output=False)
            
        except Exception as e:
            error_msg = f"Failed to refresh display: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def show_component(self, component_name: str) -> Optional[Dict[str, Any]]:
        """
        Show a specific component.
        
        Args:
            component_name: Name of component to show
            
        Returns:
            Show result dictionary
        """
        try:
            components = self.get_ui_components()
            
            if 'error' in components:
                return {'success': False, 'message': components['error']}
            
            if component_name not in components:
                return {'success': False, 'message': f'Component {component_name} not found'}
            
            component = components[component_name]
            if component is None:
                return {'success': False, 'message': f'Component {component_name} is None'}
            
            # Display the component
            safe_display(component)
            
            return {'success': True, 'message': f'Component {component_name} displayed'}
            
        except Exception as e:
            error_msg = f"Failed to show component {component_name}: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def hide_component(self, component_name: str) -> Optional[Dict[str, Any]]:
        """
        Hide a specific component.
        
        Args:
            component_name: Name of component to hide
            
        Returns:
            Hide result dictionary
        """
        try:
            components = self.get_ui_components()
            
            if 'error' in components:
                return {'success': False, 'message': components['error']}
            
            if component_name not in components:
                return {'success': False, 'message': f'Component {component_name} not found'}
            
            component = components[component_name]
            if component is None:
                return {'success': False, 'message': f'Component {component_name} is None'}
            
            # Hide the component if it supports it
            if hasattr(component, 'layout') and hasattr(component.layout, 'display'):
                component.layout.display = 'none'
            elif hasattr(component, 'style') and hasattr(component.style, 'display'):
                component.style.display = 'none'
            
            return {'success': True, 'message': f'Component {component_name} hidden'}
            
        except Exception as e:
            error_msg = f"Failed to hide component {component_name}: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def toggle_component(self, component_name: str) -> Optional[Dict[str, Any]]:
        """
        Toggle a specific component visibility.
        
        Args:
            component_name: Name of component to toggle
            
        Returns:
            Toggle result dictionary
        """
        try:
            components = self.get_ui_components()
            
            if 'error' in components:
                return {'success': False, 'message': components['error']}
            
            if component_name not in components:
                return {'success': False, 'message': f'Component {component_name} not found'}
            
            component = components[component_name]
            if component is None:
                return {'success': False, 'message': f'Component {component_name} is None'}
            
            # Check current visibility
            is_hidden = False
            if hasattr(component, 'layout') and hasattr(component.layout, 'display'):
                is_hidden = component.layout.display == 'none'
            elif hasattr(component, 'style') and hasattr(component.style, 'display'):
                is_hidden = component.style.display == 'none'
            
            # Toggle visibility
            if is_hidden:
                return self.show_component(component_name)
            else:
                return self.hide_component(component_name)
                
        except Exception as e:
            error_msg = f"Failed to toggle component {component_name}: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}