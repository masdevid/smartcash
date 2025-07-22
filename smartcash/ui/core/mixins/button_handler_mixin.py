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
        self._is_oneclick_mode: bool = False
        self._primary_button_id: Optional[str] = None
        self._registered_widgets: Dict[str, Any] = {}  # Track registered button widgets
    
    def register_button_handler(self, button_id: str, handler: Callable) -> None:
        """
        Register a button handler.
        
        Args:
            button_id: Button identifier
            handler: Handler function
        """
        # Prevent duplicate registration of the same handler
        if button_id in self._button_handlers and self._button_handlers[button_id] == handler:
            return
            
        self._button_handlers[button_id] = handler
    
    def _setup_button_handlers(self) -> None:
        """Setup button click handlers for UI operations.
        
        This method handles the complete button registration flow:
        1. Discovers all available button widgets
        2. Detects one-click button pattern if applicable
        3. Sets up explicitly registered handlers
        4. Sets up default handlers for common operations
        """
        try:
            if not hasattr(self, '_ui_components') or not self._ui_components:
                if hasattr(self, 'log_debug'):
                    self.log_debug("UI components not initialized yet")
                return
            
            # Discover available button widgets
            buttons = self._discover_button_widgets()
            
            if not buttons:
                if hasattr(self, 'log_warning'):
                    self.log_warning("⚠️ No buttons found in UI components")
                return
            
            # Detect one-click button pattern
            self._detect_oneclick_pattern(buttons)
            
            # Track which buttons we've already processed
            processed_buttons = set()
            
            # Setup registered handlers first (explicit registrations take precedence)
            registered_count = self._setup_registered_handlers(buttons, processed_buttons)
            
            # Setup default handlers for any remaining buttons
            default_count = self._setup_default_button_handlers(buttons, processed_buttons)
            
            # Minimal success logging
            if hasattr(self, 'log_debug'):
                total_handlers = registered_count + default_count
                if total_handlers > 0:
                    self.log_debug(f"Button handlers ready: {total_handlers}")
            
        except Exception as e:
            if hasattr(self, 'log_error'):
                self.log_error(f"Failed to setup button handlers: {e}", exc_info=True)
    
    def _normalize_button_name(self, name: str) -> str:
        """Normalize button name by removing '_button' suffix if present.
        
        Args:
            name: Original button name
            
        Returns:
            Normalized button name without '_button' suffix
        """
        if name.endswith('_button'):
            return name[:-7]  # Remove '_button' suffix
        return name
        
    def _extract_buttons_from_container(self, container, recursive: bool = False) -> Dict[str, Any]:
        """Extract button widgets from a container object.
        
        Args:
            container: Container object to extract buttons from
            recursive: Whether to recursively search nested containers
            
        Returns:
            Dictionary of button name to button widget mappings
        """
        buttons = {}
        
        if not container:
            return buttons
        
        # Method 1: If container is a dictionary with a 'buttons' key
        if isinstance(container, dict) and 'buttons' in container:
            if isinstance(container['buttons'], dict):
                buttons.update(container['buttons'])
                
        # Method 2: If container is an object with a 'buttons' attribute
        elif hasattr(container, 'buttons'):
            container_buttons = getattr(container, 'buttons', {})
            if isinstance(container_buttons, dict):
                buttons.update(container_buttons)
                
        # Method 3: If container has a get_buttons method
        if hasattr(container, 'get_buttons'):
            try:
                container_buttons = container.get_buttons()
                if container_buttons and isinstance(container_buttons, dict):
                    buttons.update(container_buttons)
            except Exception:
                pass  # Ignore errors from get_buttons method
                
        # Method 4: If container is an ActionContainer object, try to access internal buttons
        if hasattr(container, 'buttons') and hasattr(container.buttons, 'get'):
            try:
                # Handle ActionContainer.buttons which might have primary/action structure
                container_buttons = getattr(container, 'buttons', {})
                if isinstance(container_buttons, dict):
                    # Check for primary button
                    if 'primary' in container_buttons and container_buttons['primary']:
                        buttons['primary'] = container_buttons['primary']
                    
                    # Check for action buttons
                    if 'action' in container_buttons:
                        action_buttons = container_buttons['action']
                        if isinstance(action_buttons, dict):
                            buttons.update(action_buttons)
                        elif hasattr(action_buttons, 'on_click'):
                            buttons['action'] = action_buttons
                            
                    # Check for save/reset buttons
                    if 'save_reset' in container_buttons:
                        buttons['save'] = getattr(container, 'save_button', None)
                        buttons['reset'] = getattr(container, 'reset_button', None)
                        
                    # Check for individual save/reset buttons
                    if 'save' in container_buttons and container_buttons['save']:
                        buttons['save'] = container_buttons['save']
                    if 'reset' in container_buttons and container_buttons['reset']:
                        buttons['reset'] = container_buttons['reset']
                        
            except Exception:
                pass  # Ignore errors during button extraction
        
        # Method 5: Recursive search if requested
        if recursive and isinstance(container, dict):
            for key, value in container.items():
                if key != 'buttons' and (isinstance(value, dict) or hasattr(value, 'buttons')):
                    nested_buttons = self._extract_buttons_from_container(value, recursive=False)
                    for btn_name, btn_widget in nested_buttons.items():
                        # Prefix with container key to avoid conflicts
                        prefixed_name = f"{key}_{btn_name}" if btn_name not in buttons else btn_name
                        buttons[prefixed_name] = btn_widget
                        
        # Filter out None values and non-button widgets
        return {name: widget for name, widget in buttons.items() 
                if widget is not None and hasattr(widget, 'on_click')}
        
    def _discover_button_widgets(self) -> Dict[str, Any]:
        """Discover available button widgets from UI components.
        
        Returns:
            Dictionary mapping normalized button IDs to button widgets
            with duplicates removed (prefers non-suffixed names)
        """
        buttons = {}
        
        # Method 1: Get from action container (try both 'actions' and 'action_container')
        for container_key in ['actions', 'action_container']:
            action_container = self._ui_components.get(container_key)
            if action_container:
                container_buttons = self._extract_buttons_from_container(action_container)
                
                # Add buttons with normalized names, preferring non-suffixed names
                for btn_name, btn_widget in container_buttons.items():
                    if btn_widget and hasattr(btn_widget, 'on_click'):
                        normalized_name = self._normalize_button_name(btn_name)
                        # Only add if we don't already have this button (prefer first occurrence)
                        if normalized_name not in buttons:
                            buttons[normalized_name] = btn_widget
        
        # Method 2: Look for buttons in form container (common pattern)
        form_container = self._ui_components.get('form_container')
        if form_container:
            form_buttons = self._extract_buttons_from_container(form_container, recursive=True)
            for btn_name, btn_widget in form_buttons.items():
                if btn_widget and hasattr(btn_widget, 'on_click'):
                    normalized_name = self._normalize_button_name(btn_name)
                    if normalized_name not in buttons:
                        buttons[normalized_name] = btn_widget
        
        # Method 3: Look for buttons in components
        if 'components' in self._ui_components and hasattr(self._ui_components['components'], 'get'):
            components = self._ui_components['components']
            for key, widget in components.items():
                if hasattr(widget, 'on_click'):
                    buttons[key] = widget
        
        # Method 4: Look for direct button widgets in _ui_components (skip containers)
        container_keys = ['action_container', 'actions', 'form_container', 'header_container', 
                         'operation_container', 'main_container', 'footer_container']
        for key, widget in self._ui_components.items():
            if key in container_keys or not hasattr(widget, 'on_click'):
                continue
            buttons[key] = widget
        
        # Log the buttons we found for debugging (optimized for one-click mode) - only once during init
        if buttons and not hasattr(self, '_button_discovery_logged'):
            if hasattr(self, 'log_info'):
                # In one-click mode, just log that buttons were found
                if len(buttons) == 1 and any(self._is_primary_button(widget) for widget in buttons.values()):
                    self.log_info(f"🔧 Found one-click button: {list(buttons.keys())[0]}")
                else:
                    self.log_info(f"🔧 Found {len(buttons)} button(s) in UI components: {list(buttons.keys())}")
                
                # Mark as logged to prevent spam
                self._button_discovery_logged = True
            
            # Detailed logging only in debug mode and non-one-click scenarios
            if hasattr(self, 'log_debug'):
                one_click_detected = len(buttons) == 1 and any(self._is_primary_button(widget) for widget in buttons.values())
                if not one_click_detected:
                    for btn_id, btn_widget in buttons.items():
                        widget_type = type(btn_widget).__name__ if btn_widget else "None"
                        has_onclick = hasattr(btn_widget, 'on_click') if btn_widget else False
                        self.log_debug(f"  - {btn_id}: {widget_type} (has_on_click: {has_onclick})")
        elif hasattr(self, 'log_debug'):
            # Debug: Log available UI component keys
            ui_keys = list(self._ui_components.keys())
            self.log_debug(f"Available UI component keys: {ui_keys}")
        
        return buttons
    
    def _detect_oneclick_pattern(self, buttons: Dict[str, Any]) -> None:
        """Detect if this is a one-click button pattern.
        
        A one-click pattern is identified by:
        - Only one button with 'primary' style in the action container
        - This indicates a single operation that runs multiple sub-operations
        
        Args:
            buttons: Dictionary of discovered button widgets
        """
        try:
            primary_buttons = []
            
            # Check each button for 'primary' style
            for button_id, button_widget in buttons.items():
                if self._is_primary_button(button_widget):
                    primary_buttons.append(button_id)
            
            # One-click pattern: exactly one primary button
            if len(primary_buttons) == 1:
                self._is_oneclick_mode = True
                self._primary_button_id = primary_buttons[0]
            else:
                self._is_oneclick_mode = False
                self._primary_button_id = None
                    
        except Exception as e:
            # Fallback to multi-button mode on error
            self._is_oneclick_mode = False
            self._primary_button_id = None
            
            if hasattr(self, 'log_debug'):
                self.log_debug(f"Button pattern detection failed: {e}")
    
    def _is_primary_button(self, button_widget) -> bool:
        """Check if a button widget has primary style.
        
        Args:
            button_widget: Button widget to check
            
        Returns:
            True if button has primary style
        """
        try:
            # Check button_style attribute
            if hasattr(button_widget, 'button_style'):
                return button_widget.button_style == 'primary'
            
            # Check style attribute
            if hasattr(button_widget, 'style'):
                style = button_widget.style
                if hasattr(style, 'button_color'):
                    # Primary buttons typically have blue/primary color
                    return style.button_color in ['primary', 'blue', '#007bff']
            
            # Check class names or other attributes that might indicate primary
            if hasattr(button_widget, 'add_class') or hasattr(button_widget, 'remove_class'):
                # This suggests it's an ipywidgets Button
                return getattr(button_widget, 'button_style', '') == 'primary'
            
            return False
            
        except Exception:
            return False
    
    def _setup_registered_handlers(self, buttons: Dict[str, Any], processed_buttons: set = None) -> int:
        """Setup handlers that were explicitly registered.
        
        Args:
            buttons: Dictionary of discovered button widgets
            processed_buttons: Set to track which buttons have been processed
            
        Returns:
            Number of handlers successfully registered
        """
        if processed_buttons is None:
            processed_buttons = set()
            
        registered_count = 0
        
        for button_id, handler in self._button_handlers.items():
            # Skip if already processed
            if button_id in processed_buttons:
                if hasattr(self, 'log_debug') and not self._is_oneclick_mode:
                    self.log_debug(f"  ✓ Skipping already processed button: {button_id}")
                continue
                
            # Reduce logging noise for one-click mode
            if hasattr(self, 'log_debug') and not self._is_oneclick_mode:
                self.log_debug(f"🔍 Looking for button widget for handler '{button_id}'")
            
            # Try different naming patterns
            button_variants = [
                button_id,  # exact match
                f"{button_id}_button",  # with _button suffix
                f"btn_{button_id}",  # with btn_ prefix
                button_id.replace('_', ' ').title().replace(' ', ''),  # CamelCase
                button_id.lower().replace('_', '')  # lowercase no underscores
            ]
            
            button = None
            for variant in button_variants:
                if variant in buttons:
                    button = buttons[variant]
                    if hasattr(self, 'log_debug') and not self._is_oneclick_mode and variant != button_id:
                        self.log_debug(f"  - Found as variant: {variant}")
                    break
            
            if button and hasattr(button, 'on_click'):
                # Check if this widget is already registered to prevent duplicates
                if button_id in self._registered_widgets and self._registered_widgets[button_id] is button:
                    processed_buttons.add(button_id)
                    continue
                
                # Wrap handler with error handling
                wrapped_handler = self._wrap_button_handler(button_id, handler)
                button.on_click(wrapped_handler)
                
                # Track registered widget to prevent duplicates
                self._registered_widgets[button_id] = button
                
                # Mark as processed
                processed_buttons.add(button_id)
                registered_count += 1
                
                # Minimal logging for registration
                if hasattr(self, 'log_debug') and not self._is_oneclick_mode:
                    self.log_debug(f"✅ Handler registered: {button_id}")
            else:
                # Only log warnings for critical issues in one-click mode
                if button is None and hasattr(self, 'log_warning'):
                    if self._is_oneclick_mode and button_id == self._primary_button_id:
                        self.log_warning(f"⚠️ Primary button '{button_id}' widget not found")
                    elif not self._is_oneclick_mode:
                        self.log_warning(f"⚠️ No button widget found for handler '{button_id}'")
                elif hasattr(self, 'log_warning'):
                    if self._is_oneclick_mode and button_id == self._primary_button_id:
                        self.log_warning(f"⚠️ Primary button '{button_id}' has no on_click method")
                    elif not self._is_oneclick_mode:
                        self.log_warning(f"⚠️ Button widget for '{button_id}' has no on_click method")
                        
        return registered_count
    
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
                
                if hasattr(self, 'log_error'):
                    self.log_error(f"Button handler {button_id} failed: {e}")
                raise
        
        return wrapped_handler
    
    def _setup_default_button_handlers(self, buttons: Dict[str, Any], processed_buttons: set = None) -> int:
        """
        Setup default button handlers for common operations.
        
        Args:
            buttons: Dictionary of button widgets
            processed_buttons: Set of button IDs that have already been processed
            
        Returns:
            Number of default handlers registered
        """
        if processed_buttons is None:
            processed_buttons = set()
            
        registered_count = 0
        button_ids = set(buttons.keys())
        
        # Define default button configurations
        default_buttons = [
            # (button_id, method_name, required_method, skip_if_handled)
            ('save', 'save_config', 'save_config', True),
            ('reset', 'reset_config', 'reset_config', True),
            ('load', 'load_config', 'load_config', False),
            ('refresh', '_on_refresh_click', '_on_refresh_click', False),
            ('preprocessed', '_on_preprocessed_click', '_on_preprocessed_click', False),
            ('augmented', '_on_augmented_click', '_on_augmented_click', False)
        ]
        
        for btn_id, method_name, required_method, skip_if_handled in default_buttons:
            # Skip if already processed by registered handlers
            if btn_id in processed_buttons:
                if hasattr(self, 'log_debug') and not self._is_oneclick_mode:
                    self.log_debug(f"  ✓ Skipping already processed button: {btn_id}")
                continue
                
            # Skip if button doesn't exist in UI
            if btn_id not in button_ids:
                continue
                
            # Skip if we should skip when already handled and it's in _button_handlers
            if skip_if_handled and btn_id in self._button_handlers:
                continue
                
            # Check if the required method exists
            if hasattr(self, required_method):
                method = getattr(self, required_method)
                if not callable(method):
                    continue
                    
                # Register the handler
                self.register_button_handler(btn_id, lambda _, m=method: m())
                processed_buttons.add(btn_id)
                registered_count += 1
                
                if hasattr(self, 'log_debug') and not self._is_oneclick_mode:
                    self.log_debug(f"  ✓ Registered default handler for: {btn_id}")
        
        return registered_count
    
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
                    
                    # DO NOT change button labels - keep original description
                    # Users requested buttons to stay disabled but keep original text
            
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
    
    def is_oneclick_mode(self) -> bool:
        """Check if button handler is in one-click mode.
        
        Returns:
            True if this module uses a single primary button for multiple operations
        """
        return getattr(self, '_is_oneclick_mode', False)
    
    def get_primary_button_id(self) -> Optional[str]:
        """Get the primary button ID for one-click mode.
        
        Returns:
            Primary button ID if in one-click mode, None otherwise
        """
        return getattr(self, '_primary_button_id', None)
    
    def should_reduce_operation_logging(self, operation_name: str = None) -> bool:
        """Check if operation logging should be reduced for this module.
        
        Args:
            operation_name: Optional operation name to check
            
        Returns:
            True if logging should be reduced (e.g., for one-click operations)
        """
        # In one-click mode, reduce logging for all operations except errors
        if self._is_oneclick_mode:
            return True
        
        # For specific operation names that are typically part of sequences
        sequence_operations = ['init', 'mount', 'create', 'setup', 'verify', 'sync']
        if operation_name and any(seq_op in operation_name.lower() for seq_op in sequence_operations):
            return True
            
        return False