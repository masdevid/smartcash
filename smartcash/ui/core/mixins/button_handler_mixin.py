"""
Button handler mixin for UI modules.

Provides standard button handling functionality with automatic registration.
"""

from typing import Dict, Any, Optional, Callable
# Removed problematic import for now


class ButtonHandlerMixin:
    """
    Mixin providing common button handling functionality.
    
    This mixin provides:
    - Automatic button handler registration
    - Standard button operation patterns
    - Button state management
    - Error handling for button operations
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._button_handlers: Dict[str, Callable] = {}
        self._button_states: Dict[str, Dict[str, Any]] = {}
    
    def register_button_handler(self, button_id: str, handler: Callable) -> None:
        """
        Register a button handler.
        
        Args:
            button_id: Button identifier
            handler: Handler function
        """
        self._button_handlers[button_id] = handler
        
        if hasattr(self, 'logger'):
            self.logger.debug(f"🔘 Registered button handler: {button_id}")
    
    def _setup_button_handlers(self) -> None:
        """Setup button click handlers for UI operations."""
        try:
            if not hasattr(self, '_ui_components') or not self._ui_components:
                if hasattr(self, 'logger'):
                    self.logger.warning("Cannot setup button handlers - UI components not initialized")
                return
            
            # Get action container
            action_container = self._ui_components.get('action_container')
            if not action_container:
                if hasattr(self, 'logger'):
                    self.logger.warning("Action container not found in UI components")
                return
            
            # Get buttons from action container
            buttons = action_container.get('buttons', {})
            if not buttons:
                if hasattr(self, 'logger'):
                    self.logger.warning("No buttons found in action container")
                return
            
            # Setup registered handlers
            for button_id, handler in self._button_handlers.items():
                button = buttons.get(button_id)
                if button and hasattr(button, 'on_click'):
                    # Wrap handler with error handling
                    wrapped_handler = self._wrap_button_handler(button_id, handler)
                    button.on_click(wrapped_handler)
                    
                    if hasattr(self, 'logger'):
                        self.logger.debug(f"✅ Setup button handler: {button_id}")
                else:
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"Button not found or not clickable: {button_id}")
            
            # Setup default handlers if not already registered
            self._setup_default_button_handlers(buttons)
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to setup button handlers: {e}")
    
    def _wrap_button_handler(self, button_id: str, handler: Callable) -> Callable:
        """
        Wrap button handler with error handling and state management.
        
        Args:
            button_id: Button identifier
            handler: Original handler function
            
        Returns:
            Wrapped handler function
        """
        def wrapped_handler(button_widget):
            try:
                # Update button state
                self._set_button_state(button_id, 'processing', True)
                
                # Call original handler
                result = handler(button_widget)
                
                # Update button state on success
                self._set_button_state(button_id, 'processing', False)
                self._set_button_state(button_id, 'last_result', result)
                
                return result
                
            except Exception as e:
                # Update button state on error
                self._set_button_state(button_id, 'processing', False)
                self._set_button_state(button_id, 'last_error', str(e))
                
                if hasattr(self, 'logger'):
                    self.logger.error(f"Button handler {button_id} failed: {e}")
                raise
        
        return wrapped_handler
    
    def _setup_default_button_handlers(self, buttons: Dict[str, Any]) -> None:
        """
        Setup default button handlers for common operations.
        
        Args:
            buttons: Dictionary of button widgets
        """
        # Setup save button
        if 'save' in buttons and 'save' not in self._button_handlers:
            if hasattr(self, 'save_config'):
                self.register_button_handler('save', lambda _: self.save_config())
        
        # Setup reset button
        if 'reset' in buttons and 'reset' not in self._button_handlers:
            if hasattr(self, 'reset_config'):
                self.register_button_handler('reset', lambda _: self.reset_config())
        
        # Setup load button
        if 'load' in buttons and 'load' not in self._button_handlers:
            if hasattr(self, 'load_config'):
                self.register_button_handler('load', lambda _: self.load_config())
        
        # Setup any other common buttons
        for button_id in buttons:
            if button_id not in self._button_handlers:
                # Try to find matching method
                method_name = f"{button_id}_operation"
                if hasattr(self, method_name):
                    method = getattr(self, method_name)
                    if callable(method):
                        self.register_button_handler(button_id, lambda _: method())
    
    def _set_button_state(self, button_id: str, key: str, value: Any) -> None:
        """
        Set button state value.
        
        Args:
            button_id: Button identifier
            key: State key
            value: State value
        """
        if button_id not in self._button_states:
            self._button_states[button_id] = {}
        
        self._button_states[button_id][key] = value
    
    def _get_button_state(self, button_id: str, key: str, default: Any = None) -> Any:
        """
        Get button state value.
        
        Args:
            button_id: Button identifier
            key: State key
            default: Default value if not found
            
        Returns:
            State value or default
        """
        return self._button_states.get(button_id, {}).get(key, default)
    
    def is_button_processing(self, button_id: str) -> bool:
        """
        Check if a button is currently processing.
        
        Args:
            button_id: Button identifier
            
        Returns:
            True if button is processing
        """
        return self._get_button_state(button_id, 'processing', False)
    
    def get_button_last_result(self, button_id: str) -> Optional[Any]:
        """
        Get the last result from a button operation.
        
        Args:
            button_id: Button identifier
            
        Returns:
            Last result or None
        """
        return self._get_button_state(button_id, 'last_result')
    
    def get_button_last_error(self, button_id: str) -> Optional[str]:
        """
        Get the last error from a button operation.
        
        Args:
            button_id: Button identifier
            
        Returns:
            Last error or None
        """
        return self._get_button_state(button_id, 'last_error')
    
    def disable_button(self, button_id: str) -> None:
        """
        Disable a button.
        
        Args:
            button_id: Button identifier
        """
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                action_container = self._ui_components.get('action_container')
                if action_container:
                    buttons = action_container.get('buttons', {})
                    button = buttons.get(button_id)
                    if button and hasattr(button, 'disabled'):
                        button.disabled = True
                        
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to disable button {button_id}: {e}")
    
    def enable_button(self, button_id: str) -> None:
        """
        Enable a button.
        
        Args:
            button_id: Button identifier
        """
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                action_container = self._ui_components.get('action_container')
                if action_container:
                    buttons = action_container.get('buttons', {})
                    button = buttons.get(button_id)
                    if button and hasattr(button, 'disabled'):
                        button.disabled = False
                        
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to enable button {button_id}: {e}")
    
    def get_button_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all button states.
        
        Returns:
            Dictionary of button states
        """
        return self._button_states.copy()