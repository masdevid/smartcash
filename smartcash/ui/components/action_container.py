"""
Action Container Component

This module provides a container for action buttons with three distinct types:
1. Primary button - Large, centered button for main actions with phase support
2. Save/Reset button - For form submission and reset actions
3. Action button - Standard button for secondary actions

Note: This container handles only button management. For dialogs and other UI elements,
use the OperationContainer class.
"""
from ipywidgets import widgets, Layout, VBox, HBox, HTML, Button, ToggleButton, ButtonStyle
from typing import Dict, List, Any, Tuple, Literal, Union, Optional, Callable, TypedDict
import ipywidgets as widgets
from .save_reset_buttons import create_save_reset_buttons
from .action_buttons import create_action_buttons

# Predefined phases for Colab environment setup with emoji icons
COLAB_PHASES = {
    'initial': {
        'text': '🚀 Initialize Environment',
        'icon': '🚀',
        'style': 'primary',
        'tooltip': 'Start setting up the Colab environment',
        'disabled': False
    },
    'init': {
        'text': '⚙️ Initializing...',
        'icon': '⚙️',
        'style': 'info',
        'tooltip': 'Initializing environment setup',
        'disabled': True
    },
    'drive': {
        'text': '📁 Mounting Drive...',
        'icon': '📁',
        'style': 'info',
        'tooltip': 'Mounting Google Drive for persistent storage',
        'disabled': True
    },
    'symlink': {
        'text': '🔗 Creating Symlinks...',
        'icon': '🔗',
        'style': 'info',
        'tooltip': 'Creating symbolic links for project structure',
        'disabled': True
    },
    'folders': {
        'text': '📂 Creating Folders...',
        'icon': '📂',
        'style': 'info',
        'tooltip': 'Setting up project directory structure',
        'disabled': True
    },
    'config': {
        'text': '🔄 Syncing Config...',
        'icon': '🔄',
        'style': 'info',
        'tooltip': 'Synchronizing configuration files',
        'disabled': True
    },
    'env': {
        'text': '🌍 Setting Environment...',
        'icon': '🌍',
        'style': 'info',
        'tooltip': 'Configuring environment variables',
        'disabled': True
    },
    'verify': {
        'text': '🔍 Verifying Setup...',
        'icon': '🔍',
        'style': 'info',
        'tooltip': 'Verifying environment setup',
        'disabled': True
    },
    'complete': {
        'text': '✅ Environment Ready!',
        'icon': '✅',
        'style': 'success',
        'tooltip': 'Environment setup complete',
        'disabled': False
    },
    'error': {
        'text': '❌ Setup Failed',
        'icon': '❌',
        'style': 'danger',
        'tooltip': 'Click to see error details or retry',
        'disabled': False
    }
}

def create_action_container(
    buttons: List[Dict[str, Any]],
    title: str = None,
    container_margin: str = "12px 0",
    show_save_reset: bool = True,
    phases: Dict[str, dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """Create an action container with fixed layout and the specified buttons.
    
    Layout order: Save/Reset → Divider → Title → Primary → Action Buttons
    
    Args:
        buttons: List of button configurations. Each config should include:
            - id: Unique identifier for the button (preferred) or button_id (legacy)
            - text: Display text for the button
            - style: Button style ('primary', 'success', 'info', 'warning', 'danger')
            - icon: Optional icon to display before text (e.g., 'plus', 'save')
            - tooltip: Optional tooltip text
            - disabled: Whether the button is initially disabled
        title: Optional title for the container
        container_margin: Margin around the container (e.g., '12px 0')
        show_save_reset: Whether to show save/reset buttons (default: True)
        phases: Dictionary of phase configurations for the primary button
        **kwargs: Additional arguments to pass to ActionContainer
        
    Returns:
        Dict containing the container and button references:
        - 'container': The main container widget
        - 'buttons': Dictionary of button widgets by button_id
        - 'primary': Reference to the primary button if it exists
        - 'action_container': Reference to the ActionContainer instance
    """
    # Create container with specified margin and save/reset visibility
    action_container = ActionContainer(
        container_margin=container_margin,
        show_save_reset=show_save_reset,
        title=title,
        phases=phases,  # Pass phases to ActionContainer
        **kwargs
    )
    
    # Add buttons to container
    button_widgets = {}
    
    # Separate primary and action button configs
    primary_configs = [btn for btn in buttons if btn.get('style') == 'primary']
    action_configs = [btn for btn in buttons if btn.get('style') != 'primary']
    
    # Process primary button if provided
    if primary_configs:
        if len(primary_configs) > 1:
            print("Warning: Only one primary button is allowed. Using the first one.")
            
        primary_config = primary_configs[0].copy()
        btn_id = primary_config.pop('id', primary_config.pop('button_id', 'primary'))
        
        try:
            # Add primary button
            button = action_container.add_button(
                button_id=btn_id,
                **primary_config
            )
            if button:
                button_widgets[btn_id] = button
        except Exception as e:
            print(f"Warning: Failed to add primary button {btn_id}: {str(e)}")
    
    # Process action buttons if no primary button exists
    elif action_configs:
        for btn_config in action_configs:
            btn_config = btn_config.copy()
            btn_id = btn_config.pop('id', btn_config.pop('button_id', None))
            if not btn_id:
                continue
                
            try:
                button = action_container.add_button(
                    button_id=btn_id,
                    **btn_config
                )
                if button:
                    button_widgets[btn_id] = button
            except Exception as e:
                print(f"Warning: Failed to add button {btn_id}: {str(e)}")
    
    # Include save/reset buttons in the returned buttons dictionary if they exist
    if action_container._show_save_reset:
        if hasattr(action_container, 'save_button') and action_container.save_button is not None:
            button_widgets['save'] = action_container.save_button
        if hasattr(action_container, 'reset_button') and action_container.reset_button is not None:
            button_widgets['reset'] = action_container.reset_button
    
    # Return container and utility methods
    return {
        'container': action_container.container,
        'buttons': button_widgets,
        'primary': action_container.buttons.get('primary'),  # Using consistent naming without _button suffix
        'action_container': action_container,  # Provide access to the full container instance
        'set_phase': action_container.set_phase,
        'set_phases': action_container.set_phases,
        'enable_all': action_container.enable_all,
        'disable_all': action_container.disable_all,
        'set_all_buttons_enabled': action_container.set_all_buttons_enabled
    }


class ActionContainer:
    """A container for action buttons with three distinct types.
    
    Features:
    - Primary button: Large, centered button with phase support for environment setup
    - Save/Reset button: For form submission and reset actions
    - Action button: Standard button for secondary actions
    """
    
    def __init__(self, container_margin: str = "12px 0", phases: Dict[str, dict] = None, 
                 show_save_reset: bool = True, title: str = None):
        """Initialize the ActionContainer.
        
        Args:
            container_margin: Margin around the container (default: '12px 0')
            show_save_reset: Whether to show save/reset buttons (default: True)
            title: Optional title for the container
        """
        self.container = widgets.VBox(layout=widgets.Layout(
            width='100%',
            margin=container_margin,
            padding='0',
            border='none',
            display='flex',
            flex_flow='column nowrap',
            align_items='stretch',
            justify_content='flex-start'
        ))
        
        self.container_margin = container_margin
        self._show_save_reset = show_save_reset
        self._title = title
        
        # Initialize buttons dictionary
        self.buttons = {
            'primary': None,
            'save_reset': None,
            'action': {}
        }
        
        # Track initialization and update state to prevent recursion
        self._initializing = False
        self._initialized = False
        self._updating = False
        
        # Set default phases if not already set
        self.phases = phases or COLAB_PHASES
        self.current_phase = 'initial' if 'initial' in self.phases else next(iter(self.phases.keys()), '')
        
        # Initialize buttons
        self._init_buttons()
        
    def _get_button_colors(self, style: str) -> Tuple[str, str]:
        """Get button colors based on style.
        
        Args:
            style: Button style name
            
        Returns:
            Tuple of (background_color, text_color)
        """
        # Default colors
        bg_color = '#007bff'  # Default blue
        text_color = '#ffffff'  # White text
        
        # Map style to colors - all buttons use white text for better visibility
        style_colors = {
            'primary': ('#007bff', '#ffffff'),
            'success': ('#28a745', '#ffffff'),
            'info': ('#17a2b8', '#ffffff'),
            'warning': ('#ffc107', '#ffffff'),  # Changed from #212529 to #ffffff
            'danger': ('#dc3545', '#ffffff'),
            'secondary': ('#6c757d', '#ffffff'),
            'light': ('#f8f9fa', '#ffffff'),   # Changed from #212529 to #ffffff
            'dark': ('#343a40', '#ffffff'),
        }
        
        return style_colors.get(style.lower(), (bg_color, text_color))
        
    def _get_button_color(self, style: str) -> str:
        """Get the background color for a button style.
        
        Args:
            style: Button style name
            
        Returns:
            Background color as a hex string
        """
        return self._get_button_colors(style)[0]
        
    def _init_buttons(self):
        """Initialize all button widgets if they don't exist."""
        # Prevent re-entrancy
        if self._initializing:
            return
            
        self._initializing = True
        
        try:
            # Create save/reset buttons if needed
            if self.buttons['save_reset'] is None and self._show_save_reset:
                save_reset_buttons = create_save_reset_buttons(
                    save_label='💾 Save',
                    reset_label='🔄 Reset',
                    button_width='120px',
                    container_width='auto',
                    save_tooltip='Save current configuration',
                    reset_tooltip='Reset to default values',
                    with_sync_info=False,
                    show_icons=True,
                    alignment='right'
                )
                self.buttons['save_reset'] = save_reset_buttons['container']
                self.save_button = save_reset_buttons.get('save_button')
                self.reset_button = save_reset_buttons.get('reset_button')
                
                # Also store individual buttons in the buttons dict for handler compatibility
                if self.save_button is not None:
                    self.buttons['save'] = self.save_button
                if self.reset_button is not None:
                    self.buttons['reset'] = self.reset_button
            
            # Initialize action buttons as empty dict
            if not isinstance(self.buttons['action'], dict):
                self.buttons['action'] = {}
            
            # Mark as initialized
            self._initialized = True
                
        finally:
            self._initializing = False
            
    def set_phases(self, phases: Union[Dict[str, dict], List[dict], None]) -> None:
        """Set available phases for the primary button.
        
        Args:
            phases: Either a dictionary of phase configurations or a list of phase configs with 'id' keys
            
        Raises:
            TypeError: If phases is not a dictionary or list, or if list items don't have 'id' keys
        """
        try:
            processed_phases = {}
            
            # Handle None case
            if phases is None:
                processed_phases = COLAB_PHASES
            # Handle dictionary input (preferred format)
            elif isinstance(phases, dict):
                processed_phases = phases.copy()
            # Handle list input (legacy format)
            elif isinstance(phases, list):
                for phase in phases:
                    if not isinstance(phase, dict) or 'id' not in phase:
                        raise ValueError("Each phase in the list must be a dictionary with an 'id' key")
                    phase_id = phase['id']
                    processed_phases[phase_id] = {
                        'text': phase.get('text', phase_id.capitalize()),
                        'style': phase.get('style', 'primary'),
                        'description': phase.get('description', ''),
                        'icon': phase.get('icon', '')
                    }
            else:
                raise TypeError(f"Expected phases to be a dictionary or list, got {type(phases).__name__}")
            
            # Ensure we have at least the default phases
            if not processed_phases:
                processed_phases = COLAB_PHASES
                
            self.phases = processed_phases
            
            # Set default phase if current_phase is not set or invalid
            if not hasattr(self, 'current_phase') or self.current_phase not in self.phases:
                self.current_phase = next(iter(self.phases.keys()), '')
                
            # Debug logging
            if hasattr(self, 'logger'):
                self.logger.debug(f"Set phases: {list(self.phases.keys())}, current_phase: {self.current_phase}")
                
            # Update primary button with current phase if it exists
            if self.current_phase and self.buttons['primary'] is not None:
                self.set_phase(self.current_phase)
                
        except Exception as e:
            error_msg = f"Error in set_phases: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg, exc_info=True)
            else:
                print(f"ERROR: {error_msg}")
                import traceback
                traceback.print_exc()
            raise
    
    def set_phase(self, phase_id: str) -> None:
        """Set the current phase of the primary button.
        
        Args:
            phase_id: ID of the phase to activate
            
        Raises:
            ValueError: If the phase_id is not found in the phases dictionary
        """
        if phase_id not in self.phases:
            raise ValueError(f"Unknown phase: {phase_id}")
            
        self.current_phase = phase_id
        phase_config = self.phases[phase_id]
        
        # Create primary button if it doesn't exist
        if self.buttons['primary'] is None:
            self.buttons['primary'] = widgets.Button(
                description='',
                layout=widgets.Layout(
                    width='auto',
                    height='auto',
                    margin='12px auto',
                    padding='15px 30px',
                    font_weight='bold',
                    min_width='200px',
                    max_width='400px',
                    border_radius='6px',
                    box_shadow='0 2px 4px rgba(0,0,0,0.1)'
                ),
                disabled=False
            )
            
        if self.buttons['primary'] is not None:
            # Update button properties from phase config
            for prop, value in phase_config.items():
                if prop == 'text':
                    self.buttons['primary'].description = value
                elif prop == 'style' or prop == 'color':
                    # Get predefined colors based on style
                    bg_color, text_color = self._get_button_colors(phase_config.get('style', 'primary'))
                    style = ButtonStyle()
                    style.button_color = phase_config.get('color', bg_color)
                    style.font_weight = 'bold'
                    style.text_color = text_color
                    self.buttons['primary'].style = style
                elif hasattr(self.buttons['primary'], prop):
                    setattr(self.buttons['primary'], prop, value)
                elif prop == 'disabled':
                    self.buttons['primary'].disabled = value
        
        # Handle icon if present
        if 'icon' in phase_config:
            self.buttons['primary'].icon = phase_config['icon']
            
        # Update container if not currently updating
        if not self._updating:
            self._update_container()
    
    def update_phase_property(self, phase_id: str, prop: str, value: Any) -> None:
        """Update a property of a specific phase.
        
        Args:
            phase_id: ID of the phase to update
            prop: Property name to update
            value: New value for the property
        """
        if phase_id not in self.phases:
            raise ValueError(f"Unknown phase: {phase_id}")
            
        self.phases[phase_id][prop] = value
        
        # Update button if this is the current phase
        if self.current_phase == phase_id:
            self.set_phase(phase_id)
    
    def get_current_phase(self) -> str:
        """Get the current phase ID."""
        return self.current_phase
    
    def is_phase(self, phase_id: str) -> bool:
        """Check if the current phase matches the given ID."""
        return self.current_phase == phase_id
    
    # Convenience methods for common phase transitions
    def set_initial(self) -> None:
        """Set to initial phase."""
        self.set_phase('initial')
    
    def set_installing_deps(self) -> None:
        """Set to installing dependencies phase."""
        self.set_phase('installing_deps')
    
    def set_downloading_models(self) -> None:
        """Set to downloading models phase."""
        self.set_phase('downloading_models')
    
    def set_verifying(self) -> None:
        """Set to verifying setup phase."""
        self.set_phase('verifying')
    
    def set_ready(self) -> None:
        """Set to ready phase."""
        self.set_phase('ready')
    
    def set_error(self, message: str = None) -> None:
        """Set to error phase with optional message.
        
        Args:
            message: Optional error message to show in tooltip
        """
        if message:
            self.update_phase_property('error', 'tooltip', message)
        self.set_phase('error')
    
    def enable_all(self) -> None:
        """Enable all buttons in the container.
        
        DEPRECATED: Use module-level button state management instead.
        This method is kept for backward compatibility but delegates
        to ButtonHandlerMixin when available.
        """
        # Try to delegate to parent module's ButtonHandlerMixin first
        if hasattr(self, '_parent_module') and self._parent_module:
            if hasattr(self._parent_module, 'enable_all_buttons'):
                self._parent_module.enable_all_buttons()
                return
        
        # Fallback to direct manipulation (deprecated pattern)
        self.set_all_buttons_enabled(True)
    
    def disable_all(self) -> None:
        """Disable all buttons in the container.
        
        DEPRECATED: Use module-level button state management instead.
        This method is kept for backward compatibility but delegates
        to ButtonHandlerMixin when available.
        """
        # Try to delegate to parent module's ButtonHandlerMixin first
        if hasattr(self, '_parent_module') and self._parent_module:
            if hasattr(self._parent_module, 'disable_all_buttons'):
                self._parent_module.disable_all_buttons()
                return
        
        # Fallback to direct manipulation (deprecated pattern)
        self.set_all_buttons_enabled(False)
    
    def set_parent_module(self, parent_module) -> None:
        """Link this ActionContainer to its parent module.
        
        This enables delegation to the parent module's ButtonHandlerMixin
        for unified button state management.
        
        Args:
            parent_module: The module instance that contains ButtonHandlerMixin
        """
        self._parent_module = parent_module
    
    def set_all_buttons_enabled(self, enabled: bool):
        """Set enabled state for all buttons.
        
        This is the fallback method for direct button manipulation
        when ButtonHandlerMixin delegation is not available.
        
        Args:
            enabled: Whether to enable (True) or disable (False) all buttons
        """
        for btn in self.buttons.values():
            if btn is not None:
                btn.disabled = not enabled
                
    def _create_button(self, button_id: str, text: str, style: str = 'primary',
                     icon: str = None, tooltip: str = None, order: int = 0,
                     disabled: bool = False, **kwargs):
        """Create a button with the specified properties.
        
        Args:
            button_id: Unique identifier for the button
            text: Display text for the button
            style: Button style ('primary', 'success', 'info', 'warning', 'danger')
            icon: Optional icon to display before text (e.g., 'plus', 'save')
            tooltip: Optional tooltip text
            order: Order in which to display the button (lower numbers come first)
            disabled: Whether the button is initially disabled
            **kwargs: Additional arguments to pass to the Button constructor
            
        Returns:
            The created Button widget
        """
        # Format the button text with icon if provided
        button_text = text
        if icon:
            # Check if icon is already an emoji (starts with a Unicode character)
            if len(icon) > 0 and ord(icon[0]) > 127:
                # It's already an emoji, just add it
                button_text = f"{icon} {button_text}"
            else:
                # Convert common icon names to emojis
                emoji_map = {'plus': '➕', 'add': '➕', 'minus': '➖', 'remove': '➖', 'trash': '🗑️', 'delete': '🗑️', 'save': '💾', 'download': '📥', 'upload': '📤', 'play': '▶️', 'pause': '⏸️', 'stop': '⏹️', 'search': '🔍', 'check': '✅', 'warning': '⚠️', 'error': '❌', 'info': 'ℹ️', 'settings': '⚙️', 'refresh': '🔄', 'sync': '🔄', 'edit': '✏️', 'pencil': '✏️', 'file': '📄', 'folder': '📁', 'home': '🏠', 'user': '👤', 'users': '👥', 'chart': '📊', 'graph': '📈', 'calendar': '📅', 'clock': '🕒', 'star': '⭐', 'heart': '❤️', 'link': '🔗', 'lock': '🔒', 'unlock': '🔓', 'key': '🔑', 'gear': '⚙️', 'cog': '⚙️', 'wrench': '🔧', 'tools': '🛠️', 'tag': '🏷️', 'tags': '🏷️', 'flag': '🚩', 'bookmark': '🔖', 'camera': '📷', 'video': '🎥', 'music': '🎵', 'volume': '🔊', 'mute': '🔇', 'bell': '🔔', 'notification': '🔔', 'mail': '📧', 'email': '📧', 'phone': '📱', 'mobile': '📱', 'desktop': '🖥️', 'laptop': '💻', 'tablet': '📱', 'comment': '💬', 'comments': '💬', 'chat': '💬', 'globe': '🌐', 'world': '🌎', 'location': '📍', 'map': '🗺️', 'pin': '📌', 'marker': '📍', 'share': '📤', 'wifi': '📶', 'bluetooth': '📶', 'signal': '📶', 'rss': '📡', 'print': '🖨️', 'fax': '📠', 'calculator': '🧮', 'shopping-cart': '🛒', 'cart': '🛒', 'credit-card': '💳', 'money': '💰', 'dollar': '💵', 'euro': '💶', 'pound': '💷', 'yen': '💴', 'bitcoin': '₿', 'gift': '🎁', 'trophy': '🏆', 'medal': '🏅', 'certificate': '📜', 'graduation': '🎓', 'book': '📚', 'books': '📚', 'library': '📚', 'code': '💻', 'terminal': '💻', 'cloud': '☁️', 'database': '🗄️', 'server': '🖥️', 'hdd': '💽', 'ssd': '💾', 'usb': '📲', 'bluetooth': '📲', 'battery': '🔋', 'power': '⚡', 'lightbulb': '💡', 'idea': '💡', 'fire': '🔥', 'hot': '🔥', 'trending': '🔥', 'snow': '❄️', 'cold': '❄️', 'sun': '☀️', 'moon': '🌙', 'cloud': '☁️', 'umbrella': '☔', 'rain': '🌧️', 'snow': '❄️', 'wind': '💨', 'tornado': '🌪️', 'rainbow': '🌈', 'earth': '🌍', 'globe': '🌐', 'planet': '🪐', 'rocket': '🚀', 'satellite': '🛰️', 'microscope': '🔬', 'telescope': '🔭', 'atom': '⚛️', 'dna': '🧬', 'virus': '🦠', 'pill': '💊', 'syringe': '💉', 'stethoscope': '🩺', 'hospital': '🏥', 'ambulance': '🚑', 'police': '👮', 'fire-truck': '🚒', 'car': '🚗', 'bus': '🚌', 'truck': '🚚', 'bicycle': '🚲', 'motorcycle': '🏍️', 'airplane': '✈️', 'ship': '🚢', 'anchor': '⚓', 'ticket': '🎫', 'film': '🎬', 'game': '🎮', 'puzzle': '🧩', 'dice': '🎲', 'chess': '♟️', 'football': '⚽', 'basketball': '🏀', 'baseball': '⚾', 'tennis': '🎾', 'bowling': '🎳', 'golf': '⛳', 'fishing': '🎣', 'skiing': '⛷️', 'swimming': '🏊', 'surfing': '🏄', 'sailing': '⛵', 'running': '🏃', 'walking': '🚶', 'hiking': '🥾', 'climbing': '🧗', 'cycling': '🚴', 'yoga': '🧘', 'meditation': '🧘', 'spa': '💆', 'haircut': '💇', 'shower': '🚿', 'bath': '🛁', 'toilet': '🚽', 'toothbrush': '🪥', 'razor': '🪒', 'lipstick': '💄', 'nail-polish': '💅', 'crown': '👑', 'hat': '🎩', 'glasses': '👓', 'necktie': '👔', 'shirt': '👕', 'pants': '👖', 'dress': '👗', 'shoe': '👞', 'boot': '👢', 'sandal': '👡', 'high-heel': '👠', 'socks': '🧦', 'scarf': '🧣', 'gloves': '🧤', 'coat': '🧥', 'handbag': '👜', 'briefcase': '💼', 'backpack': '🎒', 'school': '🏫', 'notebook': '📓', 'pen': '🖊️', 'crayon': '🖍️', 'paintbrush': '🖌️', 'palette': '🎨', 'thread': '🧵', 'yarn': '🧶', 'knot': '🪢', 'scissors': '✂️', 'ruler': '📏', 'paperclip': '📎', 'pushpin': '📌', 'chain': '⛓️', 'axe': '🪓', 'pick': '⛏️', 'screwdriver': '🪛', 'nut-and-bolt': '🔩', 'brick': '🧱', 'magnet': '🧲', 'alembic': '⚗️', 'test-tube': '🧪', 'petri-dish': '🧫', 'adhesive-bandage': '🩹', 'door': '🚪', 'elevator': '🛗', 'mirror': '🪞', 'window': '🪟', 'bed': '🛏️', 'couch': '🛋️', 'chair': '🪑', 'bathtub': '🛁', 'lotion': '🧴', 'soap': '🧼', 'broom': '🧹', 'basket': '🧺', 'roll-of-paper': '🧻', 'bucket': '🪣', 'mouse-trap': '🪤', 'safety-pin': '🧷', 'teddy-bear': '🧸', 'frame': '🖼️', 'photo': '🖼️', 'picture': '🖼️', 'candle': '🕯️', 'hourglass': '⌛', 'keyboard': '⌨️', 'computer-mouse': '🖱️', 'trackball': '🖲️', 'joystick': '🕹️', 'compression': '🗜️', 'minidisc': '💽', 'floppy-disk': '💾', 'cd': '💿', 'dvd': '📀', 'vhs': '📼', 'movie-camera': '🎥', 'film-projector': '📽️', 'film-frames': '🎞️', 'telephone': '☎️', 'mobile-phone': '📱', 'pager': '📟', 'headphone': '🎧', 'loudspeaker': '📢', 'megaphone': '📣', 'postal-horn': '📯', 'no-bell': '🔕', 'musical-score': '🎼', 'musical-note': '🎵', 'musical-notes': '🎶', 'studio-microphone': '🎙️', 'level-slider': '🎚️', 'control-knobs': '🎛️', 'saxophone': '🎷', 'guitar': '🎸', 'musical-keyboard': '🎹', 'trumpet': '🎺', 'violin': '🎻', 'drum': '🥁', 'clapper': '🎬', 'bow-and-arrow': '🏹', 'shield': '🛡️', 'carpentry-saw': '🪚', 'hammer-and-pick': '⚒️', 'hammer-and-wrench': '🛠️', 'dagger': '🗡️', 'crossed-swords': '⚔️', 'pistol': '🔫', 'smoking': '🚬', 'coffin': '⚰️', 'funeral-urn': '⚱️', 'moyai': '🗿', 'placard': '🪧', 'identification-card': '🪪'}
                
                # Use the emoji map or default to a generic icon emoji
                emoji = emoji_map.get(icon.lower().replace('fa-', ''), '🔹')
                button_text = f"{emoji} {button_text}"
            
        # Create button with appropriate styling
        button_style = widgets.ButtonStyle()
        button_style.button_color = self._get_button_color(style)
        button_style.font_weight = 'bold'
        button_style.text_color = '#ffffff'  # Ensure white text on all buttons
        
        # Set up layout based on button type
        if style == 'primary':
            layout = widgets.Layout(
                width='auto',
                height='auto',
                margin='12px auto',
                padding='15px 30px',
                font_weight='bold',
                min_width='200px',
                max_width='400px',
                border_radius='6px',
                box_shadow='0 2px 4px rgba(0,0,0,0.1)'
            )
        else:
            layout = widgets.Layout(
                width='auto',
                height='auto',
                min_width='120px',
                max_width='300px',
                padding='8px 16px',
                border_radius='6px',
                box_shadow='0 2px 4px rgba(0,0,0,0.1)',
                font_weight='500',
                margin='0 4px'
            )
        
        # Create the button
        button = widgets.Button(
            description=button_text,
            style=button_style,
            disabled=disabled,
            layout=layout,
            **kwargs
        )
        
        # Set button_style attribute for one-click detection
        button.button_style = style
        
        # Store order for sorting
        button._order = order
        
        # Add tooltip if provided
        if tooltip:
            button.tooltip = tooltip
            
        return button
        
    def _normalize_button_id(self, button_id: str) -> str:
        """Normalize button ID by removing common suffixes and prefixes.
        
        Args:
            button_id: The button ID to normalize
            
        Returns:
            Normalized button ID
        """
        # Remove common suffixes
        for suffix in ['_button', '_btn', 'Button', 'Btn']:
            if button_id.endswith(suffix):
                button_id = button_id[:-len(suffix)]
                
        # Remove common prefixes
        for prefix in ['btn_', 'button_']:
            if button_id.startswith(prefix):
                button_id = button_id[len(prefix):]
                
        return button_id
        
    def add_button(self, button_id: str, text: str, style: str = 'primary', 
                  icon: str = None, tooltip: str = None, order: int = 0, 
                  disabled: bool = False, **kwargs):
        """Add a button to the container.
        
        Args:
            button_id: Unique identifier for the button (will be normalized)
            text: Display text for the button
            style: Button style ('primary', 'success', 'info', 'warning', 'danger')
            icon: Optional icon to display before text (e.g., 'plus', 'save')
            tooltip: Optional tooltip text
            order: Order in which to display the button (lower numbers come first)
            disabled: Whether the button is initially disabled
            **kwargs: Additional arguments to pass to the Button constructor
            
        Returns:
            The created Button widget
            
        Raises:
            ValueError: If button_id is 'primary' or 'save_reset' (reserved)
            ValueError: If trying to mix primary and action buttons
            ValueError: If a button with the same normalized ID already exists
        """
        # Normalize the button ID
        normalized_id = self._normalize_button_id(button_id)
        
        # Check for reserved IDs
        if normalized_id in ['primary', 'save', 'reset', 'save_reset']:
            raise ValueError(f"Button ID '{button_id}' (normalized to '{normalized_id}') is reserved for internal use")
            
        # Check for mutual exclusion between primary and action buttons
        if style == 'primary':
            # If we have any action buttons, raise an error
            if isinstance(self.buttons.get('action'), dict) and self.buttons['action']:
                raise ValueError("Cannot add primary button when action buttons exist. "
                              "Use either primary button or action buttons, not both.")
            
            # If we already have a primary button, remove it first
            if self.buttons.get('primary') is not None:
                if hasattr(self.buttons['primary'], 'close'):
                    self.buttons['primary'].close()
                self.buttons['primary'] = None
                
            # Create a new primary button with the original ID for backward compatibility
            button = self._create_button(button_id, text, style, icon, tooltip, order, disabled, **kwargs)
            self.buttons['primary'] = button
            if not self._updating:  # Prevent recursion
                self._update_container()
            return button
            
        else:  # Action button
            # If we have a primary button, raise an error
            if self.buttons.get('primary') is not None:
                raise ValueError("Cannot add action buttons when a primary button exists. "
                              "Use either primary button or action buttons, not both.")
            
            # Initialize action buttons dict if needed
            if not isinstance(self.buttons.get('action'), dict):
                self.buttons['action'] = {}
            
            # Check for existing button with the same normalized ID
            existing_id = next((id for id in self.buttons['action'] 
                              if self._normalize_button_id(id) == normalized_id), None)
            
            if existing_id is not None:
                # If a button with the same normalized ID exists, replace it
                if hasattr(self.buttons['action'][existing_id], 'close'):
                    self.buttons['action'][existing_id].close()
                del self.buttons['action'][existing_id]
            
            # Create a new action button with the original ID for display
            button = self._create_button(button_id, text, style, icon, tooltip, order, disabled, **kwargs)
            self.buttons['action'][button_id] = button
            
            # Store the normalized ID in the button object for reference
            button._normalized_id = normalized_id
            
            if not self._updating:  # Prevent recursion
                self._update_container()
            return button

    def get_button(self, button_id: str) -> Optional[widgets.Button]:
        """Get a button by its ID.
        
        Args:
            button_id: ID of the button to retrieve (can be original or normalized ID)
            
        Returns:
            The button widget or None if not found
        """
        # Normalize the input button ID
        normalized_id = self._normalize_button_id(button_id)
        
        # Check if it's a primary or save_reset button
        if button_id in self.buttons and button_id in ['primary', 'save_reset']:
            return self.buttons[button_id]
            
        # Check if it's in the action buttons dictionary by original ID
        if isinstance(self.buttons.get('action'), dict):
            # First try exact match
            if button_id in self.buttons['action']:
                return self.buttons['action'][button_id]
                
            # Then try normalized ID match
            for btn_id, button in self.buttons['action'].items():
                if hasattr(button, '_normalized_id') and button._normalized_id == normalized_id:
                    return button
                    
            # For backwards compatibility, try direct match with normalized ID
            if normalized_id in self.buttons['action']:
                return self.buttons['action'][normalized_id]
            
        # For backwards compatibility, check if it's the single action button (old style)
        if button_id == 'action' and not isinstance(self.buttons.get('action'), dict) and self.buttons.get('action') is not None:
            return self.buttons['action']
            
        return None
        
    def set_save_reset_visible(self, visible: bool = True):
        """Show or hide the save/reset buttons.
        
        Args:
            visible: Whether to show the save/reset buttons (default: True)
        """
        self._show_save_reset = visible
        self._update_container()
    
    def _should_show_button(self, button_type: str) -> bool:
        """Check if a button should be shown based on its type and current state.
        
        Args:
            button_type: Type of button ('primary', 'save_reset') or button ID for action buttons
            
        Returns:
            bool: True if the button should be shown, False otherwise
        """
        # Handle action buttons (stored in the 'action' dictionary)
        action_buttons = self.buttons.get('action', {})
        if isinstance(action_buttons, dict) and button_type in action_buttons:
            return True  # Show all action buttons by default
            
        # Handle primary and save_reset buttons
        if button_type in self.buttons and self.buttons[button_type] is not None:
            # For primary button, check if current phase is valid
            if button_type == 'primary':
                if hasattr(self, 'current_phase') and self.current_phase:
                    return self.current_phase in self.phases
                return True
            return True
            
        return False
        
    def _update_container(self):
        """Update the container's children with fixed layout order: 
        save_reset (top, right-aligned), title, primary (center)/actions (left).
        """
        # Prevent re-entrancy during initialization and updates
        if self._initializing or not hasattr(self, 'container'):
            return
            
        # Set a flag to prevent recursive updates
        if getattr(self, '_updating', False):
            return
            
        self._updating = True
        
        try:
            # Ensure buttons are initialized
            if not self._initialized:
                self._init_buttons()
                
            # Check which buttons should be shown
            has_primary = (self.buttons.get('primary') is not None and 
                          self._should_show_button('primary'))
            
            # Handle action buttons
            has_action = False
            action_buttons = []
            if isinstance(self.buttons.get('action'), dict):
                action_buttons = [
                    btn for btn_id, btn in self.buttons['action'].items()
                    if self._should_show_button(btn_id)
                ]
                has_action = len(action_buttons) > 0
                
            has_save_reset = (self._show_save_reset and 
                             self.buttons.get('save_reset') is not None and 
                             self._should_show_button('save_reset'))
            
            # Create main container sections
            container_children = []
            
            # 1. Save/Reset buttons (top, right-aligned)
            if has_save_reset and self.buttons['save_reset'] is not None:
                save_reset_section = widgets.HBox(
                    [self.buttons['save_reset']],
                    layout=widgets.Layout(
                        width='100%',
                        justify_content='flex-end',
                        margin='0 0 8px 0'
                    )
                )
                container_children.append(save_reset_section)
            
            # 2. Divider (if we have save/reset and other content)
            if (has_save_reset and self.buttons['save_reset'] is not None and
                (self._title or has_primary or has_action)):
                divider = widgets.HTML(
                    value="<hr style='margin: 8px 0; border: none; border-top: 1px solid #e0e0e0;'>"
                )
                container_children.append(divider)
            
            # 3. Title
            if self._title:
                title_widget = widgets.HTML(
                    value=f"<h4 style='margin: 8px 0 12px 0; color: #333;'>{self._title}</h4>"
                )
                container_children.append(title_widget)
            
            # 4. Primary button (centered)
            if has_primary and self.buttons['primary'] is not None:
                primary_section = widgets.HBox(
                    [self.buttons['primary']],
                    layout=widgets.Layout(
                        width='100%',
                        justify_content='center',
                        margin='10px 0'
                    )
                )
                container_children.append(primary_section)
            
            # 4. Add action buttons (left-aligned) if they exist and no primary button
            elif has_action and action_buttons:
                # Sort buttons by their order
                action_buttons.sort(key=lambda x: getattr(x, '_order', 0))
                
                action_section = widgets.HBox(
                    action_buttons,
                    layout=widgets.Layout(
                        width='100%',
                        justify_content='flex-start',
                        margin='10px 0',
                        flex_wrap='wrap',
                        gap='8px'  # Add some space between buttons
                    )
                )
                container_children.append(action_section)
            
            # If no buttons are set, create a default primary button
            if not container_children and not has_primary and not has_action and not has_save_reset:
                style = ButtonStyle()
                style.button_color = '#007bff'  # Default blue
                style.font_weight = 'bold'
                
                self.buttons['primary'] = widgets.Button(
                    description='🔄 Default Action',
                    style=style,
                    disabled=False,
                    layout=widgets.Layout(
                        width='auto',
                        height='auto',  # Flexible height
                        min_width='200px',
                        max_width='400px',  # Prevent too wide
                        padding='15px 30px',  # Generous padding for text visibility
                        border_radius='6px',
                        box_shadow='0 2px 4px rgba(0,0,0,0.1)',
                        font_weight='bold'
                    )
                )
                
                primary_section = widgets.HBox(
                    [self.buttons['primary']],
                    layout=widgets.Layout(
                        width='100%',
                        justify_content='center',
                        margin='10px 0'
                    )
                )
                container_children.append(primary_section)
            
            # Update container with proper layout
            if container_children:  # Only update if we have children to add
                self.container.children = tuple(container_children)
                self.container.layout = widgets.Layout(
                    width='100%',
                    align_items='stretch',
                    margin=self.container_margin,
                    padding='0',
                    border='none',
                    display='flex',
                    flex_flow='column nowrap',
                    justify_content='flex-start'
                )
        finally:
            self._updating = False

# Note: Dialog methods have been removed. Use OperationContainer for dialog functionality.
