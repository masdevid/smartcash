"""
Action Container Component

This module provides a container for action buttons with three distinct types:
1. Primary button - Large, centered button for main actions with phase support
2. Save/Reset button - For form submission and reset actions
3. Action button - Standard button for secondary actions

Note: This container handles only button management. For dialogs and other UI elements,
use the OperationContainer class.
"""
from typing import Dict, Any, Optional, List, TypedDict, Literal
from typing import Dict, Any, Optional, Callable, Literal, TypedDict, List
import ipywidgets as widgets

# Predefined phases for Colab environment setup
COLAB_PHASES = {
    'initial': {
        'text': '🚀 Initialize Environment',
        'icon': 'rocket',
        'style': 'primary',
        'tooltip': 'Start setting up the Colab environment',
        'disabled': False
    },
    'installing_deps': {
        'text': '⏳ Installing Dependencies...',
        'icon': 'spinner spin',
        'style': 'info',
        'tooltip': 'Installing required dependencies',
        'disabled': True
    },
    'downloading_models': {
        'text': '📥 Downloading Models...',
        'icon': 'download',
        'style': 'info',
        'tooltip': 'Downloading required model files',
        'disabled': True
    },
    'verifying': {
        'text': '🔍 Verifying Setup...',
        'icon': 'search',
        'style': 'info',
        'tooltip': 'Verifying environment setup',
        'disabled': True
    },
    'ready': {
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
        'tooltip': 'Click to see error details',
        'disabled': False
    }
}

def create_action_container(
    buttons: List[Dict[str, Any]],
    title: str = None,
    alignment: Literal['left', 'center', 'right'] = 'center',
    container_margin: str = "12px 0",
    **kwargs
) -> Dict[str, Any]:
    """Create an action container with the specified buttons.
    
    Args:
        buttons: List of button configurations
        title: Optional title for the container
        alignment: Alignment of buttons ('left', 'center', or 'right')
        container_margin: Margin around the container
        **kwargs: Additional arguments to pass to ActionContainer
        
    Returns:
        Dict containing the container and button references
    """
    container = ActionContainer(container_margin=container_margin, **kwargs)
    
    # Add buttons to container
    button_widgets = {}
    for btn_config in buttons:
        btn_id = btn_config.get('button_id')
        if btn_id:
            container.add_button(**btn_config)
            button_widgets[btn_id] = container.get_button(btn_id)
    
    # Apply alignment with proper flexbox values
    alignment_map = {
        'left': 'flex-start',
        'center': 'center',
        'right': 'flex-end'
    }
    
    # Default to 'center' if invalid alignment is provided
    flex_align = alignment_map.get(alignment, 'center')
    container.container.layout.align_items = flex_align
    
    # Add title if provided
    if title:
        title_widget = widgets.HTML(f"<h4 style='margin: 0 0 10px 0;'>{title}</h4>")
        container.container.children = (title_widget,) + container.container.children
    
    return {
        'container': container.container,
        'buttons': button_widgets,
        'set_phase': container.set_phase,
        'set_phases': container.set_phases,
        'enable_all': container.enable_all,
        'disable_all': container.disable_all,
        'set_all_buttons_enabled': container.set_all_buttons_enabled
    }


class ActionContainer:
    """A container for action buttons with three distinct types.
    
    Features:
    - Primary button: Large, centered button with phase support for environment setup
    - Save/Reset button: For form submission and reset actions
    - Action button: Standard button for secondary actions
    """
    
    def __init__(self, container_margin: str = "12px 0"):
        """Initialize the ActionContainer.
        
        Args:
            container_margin: Margin around the container (default: "12px 0")
        """
        # Button storage
        self.buttons = {
            'primary': None,
            'save_reset': None,
            'action': None
        }
        
        # Primary button phases
        self.phases = {}
        self.current_phase = 'initial'
        
        # Set default Colab environment setup phases
        self.set_phases(COLAB_PHASES)
        
        # Create the container
        self.container = widgets.VBox(
            layout=widgets.Layout(
                width='100%',
                margin=container_margin,
                display='flex',
                flex_direction='column',
                align_items='stretch'
            )
        )
        
        # Initialize buttons
        self._init_buttons()
    
    def _init_buttons(self) -> None:
        """Initialize all button widgets."""
        # Primary button (large, centered) with phase support
        self.buttons['primary'] = widgets.Button(
            layout=widgets.Layout(
                width='auto',
                margin='12px auto',
                padding='10px 24px',
                font_weight='bold',
                min_width='200px'
            ),
            disabled=False
        )
        
        # Save/Reset button (no phase support)
        self.buttons['save_reset'] = widgets.ToggleButton(
            value=False,
            description='💾 Save',
            button_style='success',
            layout=widgets.Layout(
                width='auto',
                margin='6px 0',
                align_self='center',
                min_width='120px'
            )
        )
        
        # Action button (no phase support)
        self.buttons['action'] = widgets.Button(
            description='⚙️ Configure',
            layout=widgets.Layout(
                width='auto',
                margin='6px 0',
                align_self='center',
                min_width='120px'
            ),
            button_style='info'
        )
        
        # Add buttons to container
        self.container.children = [
            self.buttons['primary'],
            self.buttons['save_reset'],
            self.buttons['action']
        ]
        
        # Apply initial phase
        self.set_phase('initial')
    
    def set_phases(self, phases: Dict[str, dict]) -> None:
        """Set available phases for the primary button.
        
        Args:
            phases: Dictionary of phase configurations
        """
        self.phases = phases
        
    def set_phase(self, phase_id: str) -> None:
        """Set the current phase of the primary button.
        
        Args:
            phase_id: ID of the phase to activate
        """
        if phase_id not in self.phases:
            raise ValueError(f"Unknown phase: {phase_id}")
            
        self.current_phase = phase_id
        phase = self.phases[phase_id]
        button = self.buttons['primary']
        
        # Update button properties
        button.description = phase.get('text', '')
        button.button_style = phase.get('style', 'primary')
        button.tooltip = phase.get('tooltip', '')
        button.disabled = phase.get('disabled', False)
        
        # Handle icon if present
        if 'icon' in phase:
            button.icon = phase['icon']
    
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
        """
        if icon and not icon.startswith('fa-'):
            icon = f'fa-{icon}'
            
        button = widgets.Button(
            description=f"{text}",
            layout=widgets.Layout(width='auto'),
            disabled=disabled,
            **kwargs
        )
        
        if tooltip:
            button.tooltip = tooltip
            
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
        return self.buttons.get(button_id)
        
    def _update_container(self):
        """Update the container's children based on current buttons."""
        # Filter out None buttons and sort by order
        children = []
        for btn_id, btn in sorted(self.buttons.items(), key=lambda x: getattr(x[1], '_order', 0)):
            if btn is not None:
                children.append(btn)
                
        # Update container children
        self.container.children = children
    
    # Note: Dialog methods have been removed. Use OperationContainer for dialog functionality.
