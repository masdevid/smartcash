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
from typing import Dict, Any, Optional, Callable, Literal, TypedDict, List
import ipywidgets as widgets
from .save_reset_buttons import create_save_reset_buttons
from .action_buttons import create_action_buttons

# Predefined phases for Colab environment setup
COLAB_PHASES = {
    'initial': {
        'text': '🚀 Initialize Environment',
        'icon': 'rocket',
        'style': 'primary',
        'tooltip': 'Start setting up the Colab environment',
        'disabled': False
    },
    'init': {
        'text': '⚙️ Initializing...',
        'icon': 'cog',
        'style': 'info',
        'tooltip': 'Initializing environment setup',
        'disabled': True
    },
    'drive': {
        'text': '📁 Mounting Drive...',
        'icon': 'cloud',
        'style': 'info',
        'tooltip': 'Mounting Google Drive for persistent storage',
        'disabled': True
    },
    'symlink': {
        'text': '🔗 Creating Symlinks...',
        'icon': 'link',
        'style': 'info',
        'tooltip': 'Creating symbolic links for project structure',
        'disabled': True
    },
    'folders': {
        'text': '📂 Creating Folders...',
        'icon': 'folder',
        'style': 'info',
        'tooltip': 'Setting up project directory structure',
        'disabled': True
    },
    'config': {
        'text': '⚙️ Syncing Config...',
        'icon': 'sync',
        'style': 'info',
        'tooltip': 'Synchronizing configuration files',
        'disabled': True
    },
    'env': {
        'text': '🌍 Setting Environment...',
        'icon': 'globe',
        'style': 'info',
        'tooltip': 'Configuring environment variables',
        'disabled': True
    },
    'verify': {
        'text': '🔍 Verifying Setup...',
        'icon': 'search',
        'style': 'info',
        'tooltip': 'Verifying environment setup',
        'disabled': True
    },
    'complete': {
        'text': '✅ Environment Ready!',
        'icon': 'check',
        'style': 'success',
        'tooltip': 'Environment setup complete',
        'disabled': False
    },
    'error': {
        'text': '❌ Setup Failed',
        'icon': 'exclamation-triangle',
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
    alignment: Literal['left', 'center', 'right'] = None,  # Deprecated, kept for compatibility
    **kwargs
) -> Dict[str, Any]:
    """Create an action container with fixed layout and the specified buttons.
    
    Layout is fixed as: Save/Reset (right) → Divider → Primary (center) → Actions (left)
    
    Args:
        buttons: List of button configurations. Each config should include:
            - button_id: Unique identifier for the button
            - text: Display text for the button
            - style: Button style ('primary', 'success', 'info', 'warning', 'danger')
            - icon: Optional icon to display before text (e.g., 'plus', 'save')
            - tooltip: Optional tooltip text
            - disabled: Whether the button is initially disabled
        title: Optional title for the container
        container_margin: Margin around the container (e.g., '12px 0')
        show_save_reset: Whether to show save/reset buttons (default: True)
        alignment: [DEPRECATED] This parameter is ignored - layout is now fixed
        **kwargs: Additional arguments to pass to ActionContainer
        
    Returns:
        Dict containing the container and button references:
        - 'container': The main container widget
        - 'buttons': Dictionary of button widgets by button_id
        - 'primary_button': Reference to the primary button if it exists
        - 'action_container': Reference to the ActionContainer instance
    """
    # Create container with specified margin and save/reset visibility
    action_container = ActionContainer(
        container_margin=container_margin,
        show_save_reset=show_save_reset,
        **kwargs
    )
    
    # Add buttons to container
    button_widgets = {}
    for btn_config in buttons:
        btn_id = btn_config.get('button_id')
        if not btn_id:
            continue
            
        # Map old parameter names to new ones if needed
        btn_config = btn_config.copy()
        if 'button_id' in btn_config:
            btn_id = btn_config.pop('button_id')
            btn_config['id'] = btn_id
            
        # Add button to container
        action_container.add_button(**btn_config)
        
        # Store reference to the button
        button_widgets[btn_id] = action_container.get_button(btn_id)
    
    # Store title to be added after container is initialized
    action_container._title = title
    
    # Return container and utility methods
    return {
        'container': action_container.container,
        'buttons': button_widgets,
        'primary_button': action_container.buttons.get('primary'),
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
                 show_save_reset: bool = True, **kwargs):
        """Initialize the ActionContainer.
        
        Args:
            container_margin: Margin around the container (default: "12px 0")
            phases: Dictionary of phase configurations for primary button (optional)
            show_save_reset: Whether to show save/reset buttons (default: True)
            **kwargs: Additional keyword arguments (ignored for compatibility)
        """
        # Store container margin
        self.container_margin = container_margin
        
        # Initialize button storage
        self.buttons = {
            'primary': None,
            'save_reset': None,
            'action': None
        }
        
        # Set up save/reset visibility
        self._show_save_reset = show_save_reset
        
        # Initialize phases with default values if none provided
        self.phases = phases or COLAB_PHASES
        self.current_phase = 'initial' if 'initial' in self.phases else next(iter(self.phases.keys()), '')
        
        # Set default Colab environment setup phases
        self.set_phases(COLAB_PHASES)
        
        # Create the container with vertical layout
        self.container = widgets.VBox(
            layout=widgets.Layout(
                width='100%',
                margin=container_margin,
                display='flex',
                flex_flow='column nowrap',
                align_items='stretch',
                gap='8px'
            )
        )
        
        # Create a horizontal container for buttons
        self.buttons_container = widgets.HBox(
            layout=widgets.Layout(
                width='100%',
                display='flex',
                flex_flow='row wrap',
                align_items='center',
                justify_content='flex-start',
                gap='8px'
            )
        )
        
        # Initialize buttons
        self._init_buttons()
    
    def _init_buttons(self):
        """Initialize all button widgets if they don't exist."""
        # Primary button (large, centered) with phase support and flexible styling
        self.buttons['primary'] = widgets.Button(
            layout=widgets.Layout(
                width='auto',
                height='auto',  # Allow height to be flexible
                margin='12px auto',
                padding='15px 30px',  # Increased padding for better text visibility
                font_weight='bold',
                min_width='200px',
                max_width='400px',  # Prevent button from becoming too wide
                border_radius='6px',
                box_shadow='0 2px 4px rgba(0,0,0,0.1)'
            ),
            disabled=False
        )
        
        # Create save/reset buttons using the dedicated component
        save_reset_buttons = create_save_reset_buttons(
            save_label='💾 Save',
            reset_label='🔄 Reset',
            button_width='120px',
            container_width='auto',
            save_tooltip='Save current configuration',
            reset_tooltip='Reset to default values',
            with_sync_info=False,
            show_icons=True,
            alignment='left'
        )
        self.buttons['save_reset'] = save_reset_buttons['container']
        self.save_button = save_reset_buttons['save_button']
        self.reset_button = save_reset_buttons['reset_button']
        
    def _get_button_colors(self, style_name: str) -> tuple:
        """Get button background and text color based on style name.
        
        Args:
            style_name: Name of the button style
            
        Returns:
            tuple: (background_color, text_color)
        """
        colors = {
            'primary': ('#007bff', '#ffffff'),    # Blue with white text
            'success': ('#28a745', '#ffffff'),    # Green with white text
            'info': ('#17a2b8', '#ffffff'),       # Cyan with white text
            'warning': ('#ffc107', '#212529'),    # Yellow with dark text
            'danger': ('#dc3545', '#ffffff'),     # Red with white text
            'secondary': ('#6c757d', '#ffffff'),  # Gray with white text
            'light': ('#f8f9fa', '#212529'),      # Light gray with dark text
            'dark': ('#343a40', '#ffffff')        # Dark gray with white text
        }
        return colors.get(style_name.lower(), colors['primary'])
        
    def _init_buttons(self):
        """Initialize all button widgets if they don't exist."""
        # Create save/reset buttons using the dedicated component
        save_reset_buttons = create_save_reset_buttons(
            save_label='💾 Save',
            reset_label='🔄 Reset',
            container_width='auto',
            with_sync_info=False,
            show_icons=True,
            alignment='left'
        )
        self.buttons['save_reset'] = save_reset_buttons['container']
        self.save_button = save_reset_buttons['save_button']
        self.reset_button = save_reset_buttons['reset_button']
        
        # Create action buttons using the dedicated component
        if self.buttons['primary'] is None:
            action_buttons = create_action_buttons(
                buttons=[{
                    'button_id': 'configure',
                    'text': '⚙️ Configure',
                    'style': 'info',
                    'tooltip': 'Configure settings',
                    'order': 1
                }],
                alignment='left',
                container_width='auto',
                button_spacing='8px',
                container_margin='6px 0'
            )
            self.buttons['action'] = action_buttons['container']
        else:
            self.buttons['action'] = None
            
        # Add buttons to container, filtering out None values
        children = []
        
        if self.buttons['primary'] is not None:
            children.append(self.buttons['primary'])
            
        if self.buttons['save_reset'] is not None:
            children.append(self.buttons['save_reset'])
            
        if self.buttons['action'] is not None:
            children.append(self.buttons['action'])
            
        self.container.children = children
        
        # Apply initial phase
        self.set_phase('initial')
    
    def set_phases(self, phases: Dict[str, dict]) -> None:
        """Set available phases for the primary button.
        
        Args:
            phases: Dictionary of phase configurations
        """
        self.phases = phases or {}
        
        # Ensure we have at least the default phases
        if not self.phases and hasattr(self, 'phases'):
            self.phases = COLAB_PHASES
            
        # Set default phase if current_phase is not set or invalid
        if not hasattr(self, 'current_phase') or self.current_phase not in self.phases:
            self.current_phase = next(iter(self.phases.keys()), '')
            
        # Update primary button with current phase if it exists
        if self.current_phase and self.buttons['primary'] is not None:
            self.set_phase(self.current_phase)
    
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
        
        # Initialize primary button if it doesn't exist
        if self.buttons['primary'] is None:
            self._init_buttons()
            
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
        """Enable all buttons in the container."""
        self.set_all_buttons_enabled(True)
    
    def disable_all(self) -> None:
        """Disable all buttons in the container."""
        self.set_all_buttons_enabled(False)
    
    def set_all_buttons_enabled(self, enabled: bool):
        """Set enabled state for all buttons.
        
        Args:
            enabled: Whether to enable (True) or disable (False) all buttons
        """
        for btn in self.buttons.values():
            if btn is not None:
                btn.disabled = not enabled
                
    def add_button(self, button_id: str, text: str, style: str = 'primary', 
                  icon: str = None, tooltip: str = None, order: int = 0, 
                  disabled: bool = False, **kwargs):
        """Add a button to the container.
        
        Args:
            button_id: Unique identifier for the button
            text: Button text
            style: Button style ('primary', 'success', 'info', 'warning', 'danger')
            icon: Optional icon to display before text
            tooltip: Optional tooltip text
            order: Display order (lower numbers appear first)
            disabled: Whether the button is initially disabled
            **kwargs: Additional arguments to pass to Button constructor
            
        Raises:
            ValueError: If trying to add a primary button when action buttons exist,
                       or action buttons when a primary button exists
        """
        # Validate button type
        if style == 'primary' and self.buttons['action']:
            raise ValueError(
                "Cannot add primary button when action buttons exist. "
                "Use either primary button or action buttons, not both."
            )
        
        if button_id and button_id != 'primary' and self.buttons['primary']:
            raise ValueError(
                "Cannot add action buttons when a primary button exists. "
                "Use either primary button or action buttons, not both."
            )
        
        if icon and not icon.startswith('fa-'):
            icon = f'fa-{icon}'
            
        # Get colors based on button style
        bg_color, text_color = self._get_button_colors(style)
        button_style = ButtonStyle()
        button_style.button_color = bg_color
        button_style.font_weight = 'bold'
        button_style.text_color = text_color
        
        button = widgets.Button(
            description=f"{text}",
            layout=widgets.Layout(
                width='auto',
                min_width='120px',  # Ensure minimum width for better appearance
                padding='8px 16px',  # Add padding for better clickability
                margin='2px'         # Add small margin between buttons
            ),
            disabled=disabled,
            style=button_style,
            **kwargs
        )
        
        if tooltip:
            button.tooltip = tooltip
            
        # Store order for sorting
        button._order = order
        
        # Store button reference
        self.buttons[button_id] = button
        
        # Update container children
        self._update_container()

    def get_button(self, button_id: str) -> Optional[widgets.Button]:
        """Get a button by its ID.
        
        Args:
            button_id: ID of the button to retrieve
            
        Returns:
            The button widget or None if not found
        """
        # Check if the button is stored directly
        if button_id in self.buttons:
            return self.buttons[button_id]
            
        # For backwards compatibility, check if it's the single action button
        if button_id == 'action' and not isinstance(self.buttons['action'], dict) and self.buttons['action'] is not None:
            return self.buttons['action']
            
        # Check if it's in the action buttons dictionary (old style)
        if isinstance(self.buttons.get('action'), dict) and button_id in self.buttons['action']:
            return self.buttons['action'][button_id]
            
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
            button_type: Type of button ('primary', 'action', or 'save_reset')
            
        Returns:
            bool: True if the button should be shown, False otherwise
        """
        if button_type not in self.buttons or self.buttons[button_type] is None:
            return False
            
        # For primary button, check if current phase is valid
        if button_type == 'primary' and hasattr(self, 'current_phase') and self.current_phase:
            return self.current_phase in self.phases
            
        return True
        
    def _update_container(self):
        """Update the container's children with fixed layout order: 
        save_reset (top, right-aligned), divider, title, primary (center)/actions (left).
        """
        # Ensure all buttons are initialized
        self._init_buttons()
        
        # Check which buttons should be shown
        has_primary = (self.buttons.get('primary') is not None and 
                      self._should_show_button('primary'))
        has_action = (self.buttons.get('action') is not None and 
                     self._should_show_button('action'))
        has_save_reset = (self._show_save_reset and 
                         self.buttons.get('save_reset') is not None and 
                         self._should_show_button('save_reset'))
        
        # Create main container sections
        container_sections = []
        
        # 1. Save/Reset section (top, right-aligned)
        if has_save_reset:
            save_reset_section = widgets.HBox(
                [self.buttons['save_reset']],
                layout=widgets.Layout(
                    width='100%',
                    justify_content='flex-end',  # Right alignment
                    margin='0 0 10px 0'
                )
            )
            container_sections.append(save_reset_section)
            
            # Add divider after save/reset
            divider = widgets.HTML(
                '<hr style="margin: 10px 0; border: 0; border-top: 1px solid #e0e0e0;">',
                layout=widgets.Layout(width='100%')
            )
            container_sections.append(divider)
        
        # 2. Add title after divider if it exists
        if hasattr(self, '_title') and self._title:
            title_section = widgets.HTML(
                f"<h4 style='margin: 0 0 15px 0;'>{self._title}</h4>"
            )
            container_sections.append(title_section)
        
        # 3. Primary button section (center-aligned)
        if has_primary:
            primary_section = widgets.HBox(
                [self.buttons['primary']],
                layout=widgets.Layout(
                    width='100%',
                    justify_content='center',  # Center alignment for primary
                    margin='10px 0'
                )
            )
            container_sections.append(primary_section)
        # 4. Action buttons section (left-aligned) - only if no primary button
        elif has_action:
            action_section = widgets.HBox(
                [self.buttons['action']],
                layout=widgets.Layout(
                    width='100%',
                    justify_content='flex-start',  # Left alignment for actions
                    margin='10px 0'
                )
            )
            container_sections.append(action_section)
        
        # If no buttons are set, create a default primary button
        if not container_sections:
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
            container_sections.append(primary_section)
        
        # Update container with proper layout
        self.container = widgets.VBox(
            children=container_sections,
            layout=widgets.Layout(
                width='100%',
                align_items='stretch',
                margin=self.container_margin if hasattr(self, 'container_margin') else '12px 0',
                padding='0',
                border='none',
                display='flex',
                flex_flow='column nowrap',
                justify_content='flex-start'
            )
        )
        
    def add_button(self, button_id: str, text: str, style: str = 'primary',
                  icon: str = None, tooltip: str = None, order: int = 0, 
                  disabled: bool = False, **kwargs):
        """Add a button to the container.
        
        Args:
            button_id: Unique identifier for the button
            text: Button text
            style: Button style ('primary', 'success', 'info', 'warning', 'danger')
            icon: Optional icon to display before text
            tooltip: Optional tooltip text
            order: Display order (lower numbers appear first)
            disabled: Whether the button is initially disabled
            **kwargs: Additional arguments to pass to Button constructor
        """
        # Get colors based on button style
        bg_color, text_color = self._get_button_colors(style)
        button_style = ButtonStyle()
        button_style.button_color = bg_color
        button_style.font_weight = 'bold'
        button_style.text_color = text_color
        
        # Add icon if provided
        if icon:
            if not icon.startswith('fa-'):
                icon = f'fa-{icon}'
            text = f"<i class='fa {icon} fa-fw'></i> {text}"
        
        button = widgets.Button(
            description=text,
            layout=widgets.Layout(
                width='auto',
                min_width='120px',  # Ensure minimum width for better appearance
                padding='8px 16px',  # Add padding for better clickability
                margin='2px'         # Add small margin between buttons
            ),
            disabled=disabled,
            style=button_style,
            **kwargs
        )
        
        if tooltip:
            button.tooltip = tooltip
            
        # Store order for sorting
        button._order = order
        
        # Store button reference
        self.buttons[button_id] = button
        
        # Update container children
        self._update_container()

# Note: Dialog methods have been removed. Use OperationContainer for dialog functionality.
