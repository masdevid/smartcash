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
            
            # Try different container sources for buttons
            buttons = {}
            
            # Method 1: Get from action container
            action_container = self._ui_components.get('action_container')
            if action_container and isinstance(action_container, dict):
                buttons.update(action_container.get('buttons', {}))
            
            # Method 2: Get buttons directly from UI components (for individual button widgets)
            button_keys = [key for key in self._ui_components.keys() if key.endswith('_button')]
            for button_key in button_keys:
                button_id = button_key.replace('_button', '')
                buttons[button_id] = self._ui_components[button_key]
            
            if not buttons:
                if hasattr(self, 'logger'):
                    self.logger.warning("No buttons found in UI components")
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
        # Setup save button (skip if already registered by BaseUIModule)
        if 'save' in buttons and 'save' not in self._button_handlers:
            if hasattr(self, 'save_config') and not hasattr(self, '_handle_save_config'):
                self.register_button_handler('save', lambda _: self.save_config())
        
        # Setup reset button (skip if already registered by BaseUIModule)
        if 'reset' in buttons and 'reset' not in self._button_handlers:
            if hasattr(self, 'reset_config') and not hasattr(self, '_handle_reset_config'):
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
    
    def disable_all_buttons(self, message: str = "⏳ Operation in progress...", button_id: str = None) -> Dict[str, Any]:
        """
        Disable buttons and store their previous states.
        
        Args:
            message: Message to display for operation buttons (save/reset keep original text)
            button_id: If provided, only disable this specific button. If None, disable all buttons.
            
        Returns:
            Dictionary containing previous button states for restoration
        """
        try:
            if not hasattr(self, '_ui_components') or not self._ui_components:
                return {}
            
            button_states = {}
            
            # Get buttons from different sources
            buttons = {}
            
            # Method 1: From action container
            action_container = self._ui_components.get('action_container')
            if action_container and isinstance(action_container, dict):
                buttons.update(action_container.get('buttons', {}))
            
            # Method 2: Direct button widgets
            button_keys = [key for key in self._ui_components.keys() if key.endswith('_button')]
            for button_key in button_keys:
                btn_id = button_key.replace('_button', '')
                buttons[btn_id] = self._ui_components[button_key]
            
            # If a specific button_id is provided, only process that button
            if button_id is not None:
                buttons = {k: v for k, v in buttons.items() if k == button_id}
            
            # Disable each button and store its previous state
            for btn_id, button in buttons.items():
                if button and hasattr(button, 'disabled') and hasattr(button, 'description'):
                    # Store original state
                    button_states[btn_id] = {
                        'disabled': button.disabled,
                        'description': button.description,
                        'modified': False  # Track if we modified this button
                    }
                    
                    # Only modify if not already disabled
                    if not button.disabled:
                        button.disabled = True
                        button_states[btn_id]['modified'] = True
                    
                    # Update description only for operation buttons, not save/reset
                    if btn_id not in ['save', 'reset'] and btn_id != button_id:
                        button_states[btn_id]['original_description'] = button.description
                        button.description = message
            
            # Store the button states for later restoration
            if not hasattr(self, '_button_states_backup'):
                self._button_states_backup = {}
            self._button_states_backup.update(button_states)
            
            return button_states
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error disabling buttons: {e}")
            return {}
    
    def enable_all_buttons(self, button_states: Dict[str, Any] = None, button_id: str = None) -> None:
        """
        Re-enable buttons and restore their previous states.
        
        Args:
            button_states: Previous button states to restore. If None, use internal backup.
            button_id: If provided, only enable this specific button. If None, enable all buttons.
        """
        try:
            if not hasattr(self, '_ui_components') or not self._ui_components:
                return
            
            # Use internal backup if no button_states provided
            if button_states is None and hasattr(self, '_button_states_backup'):
                button_states = self._button_states_backup
            
            if not button_states:
                return
            
            # Get buttons from different sources
            buttons = {}
            
            # Method 1: From action container
            action_container = self._ui_components.get('action_container')
            if action_container and isinstance(action_container, dict):
                buttons.update(action_container.get('buttons', {}))
            
            # Method 2: Direct button widgets
            button_keys = [key for key in self._ui_components.keys() if key.endswith('_button')]
            for button_key in button_keys:
                btn_id = button_key.replace('_button', '')
                buttons[btn_id] = self._ui_components[button_key]
            
            # If a specific button_id is provided, only process that button
            if button_id is not None:
                buttons = {k: v for k, v in buttons.items() if k == button_id}
            
            # Restore button states
            for btn_id, button in buttons.items():
                if btn_id in button_states and button and hasattr(button, 'disabled'):
                    state = button_states[btn_id]
                    # Only modify if we previously modified this button
                    if state.get('modified', True):
                        button.disabled = state.get('disabled', False)
                        
                        # Restore original description if we have it
                        if 'original_description' in state and hasattr(button, 'description'):
                            button.description = state['original_description']
                        elif 'description' in state and hasattr(button, 'description'):
                            button.description = state['description']
            
            # Clean up the backup
            if button_id is not None and hasattr(self, '_button_states_backup'):
                if button_id in self._button_states_backup:
                    del self._button_states_backup[button_id]
            else:
                # Clear all backups if we're enabling all buttons
                if hasattr(self, '_button_states_backup'):
                    del self._button_states_backup
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error enabling buttons: {e}")
                
    def set_button_message(self, button_id: str, message: str) -> None:
        """
        Set message for a specific button.
        
        Args:
            button_id: Button identifier
            message: Message to set
        """
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                # Try direct button access first
                button = self._ui_components.get(f'{button_id}_button')
                if not button:
                    # Try action container
                    action_container = self._ui_components.get('action_container')
                    if action_container and isinstance(action_container, dict):
                        buttons = action_container.get('buttons', {})
                        button = buttons.get(button_id)
                
                if button and hasattr(button, 'description'):
                    button.description = message
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to set button message for {button_id}: {e}")