"""
File: smartcash/ui/components/action_buttons.py
Action Buttons Component

A reusable component for creating flexible action button groups with consistent styling.
Supports primary, secondary, and special action buttons with customizable layouts.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import ipywidgets as widgets
from IPython.display import display

# Type aliases
ButtonConfig = Dict[str, Any]
ButtonList = List[ButtonConfig]
ButtonStyle = Dict[str, str]

# Default button configurations
DEFAULT_BUTTON_CONFIG = {
    'width': '140px',
    'height': '32px',
    'margin': '0 5px 0 0',
    'padding': '0 8px'
}

DEFAULT_BUTTON_STYLES = {
    'primary': {'button_color': '#4CAF50', 'text_color': 'white'},
    'secondary': {'button_color': '#f0f0f0', 'text_color': '#333'},
    'warning': {'button_color': '#ff9800', 'text_color': 'white'},
    'danger': {'button_color': '#f44336', 'text_color': 'white'},
    'info': {'button_color': '#2196F3', 'text_color': 'white'},
}

@dataclass
class ButtonDefinition:
    """Data class for button configuration"""
    id: str
    label: str
    style: str = 'secondary'
    icon: str = ''
    tooltip: str = ''
    visible: bool = True
    disabled: bool = False
    width: str = 'auto'
    order: int = 0

class ActionButtons:
    """A flexible action button group component"""
    
    def __init__(self, container_width: str = '100%', button_spacing: str = '8px'):
        """Initialize the action buttons component"""
        self.buttons: Dict[str, widgets.Widget] = {}
        self.button_definitions: Dict[str, ButtonDefinition] = {}
        self.container = widgets.HBox(
            layout=widgets.Layout(
                width=container_width,
                display='flex',
                flex_flow='row wrap',
                align_items='center',
                justify_content='flex-start',
                margin='5px 0',
                padding='5px',
            )
        )
        self.button_spacing = button_spacing
    
    def add_button(self, button_id: str, label: str, style: str = 'secondary', 
                   icon: str = '', tooltip: str = '', visible: bool = True, 
                   disabled: bool = False, width: str = 'auto', order: int = 0) -> None:
        """Add a new button to the action group"""
        if button_id in self.buttons:
            raise ValueError(f"Button with id '{button_id}' already exists")
            
        button_def = ButtonDefinition(
            id=button_id,
            label=label,
            style=style,
            icon=icon,
            tooltip=tooltip,
            visible=visible,
            disabled=disabled,
            width=width,
            order=order
        )
        
        self.button_definitions[button_id] = button_def
        self._create_button(button_def)
        self._update_layout()
    
    def _create_button(self, button_def: ButtonDefinition) -> None:
        """Create a button widget from definition"""
        style = DEFAULT_BUTTON_STYLES.get(button_def.style.lower(), {})
        
        button = widgets.Button(
            description=f"{button_def.icon} {button_def.label}".strip(),
            tooltip=button_def.tooltip or f"Klik untuk {button_def.label.lower()}",
            disabled=button_def.disabled,
            layout=widgets.Layout(
                width=button_def.width or DEFAULT_BUTTON_CONFIG['width'],
                height=DEFAULT_BUTTON_CONFIG['height'],
                margin=DEFAULT_BUTTON_CONFIG['margin'],
                padding=DEFAULT_BUTTON_CONFIG['padding'],
                display='flex' if button_def.visible else 'none',
                align_items='center',
                justify_content='center'
            )
        )
        
        # Apply button styling
        button.style.button_color = style.get('button_color', '')
        button.style.text_color = style.get('text_color', 'black')
        button.style.font_weight = 'bold'
        
        # Store original values for resetting
        setattr(button, '_original_style', button_def.style)
        setattr(button, '_original_description', button_def.label)
        
        self.buttons[button_def.id] = button
    def _update_layout(self) -> None:
        """Update the container layout with current buttons"""
        # Sort buttons by order then by label
        sorted_buttons = sorted(
            self.button_definitions.values(),
            key=lambda x: (x.order, x.label)
        )
        
        # Create button list in order
        button_widgets = []
        for button_def in sorted_buttons:
            if button_def.id in self.buttons and button_def.visible:
                button_widgets.append(self.buttons[button_def.id])
        
        # Update container children
        self.container.children = button_widgets
    
    def get_button(self, button_id: str) -> Optional[widgets.Widget]:
        """Get a button by its ID"""
        return self.buttons.get(button_id)
    
    def set_visible(self, button_id: str, visible: bool = True) -> None:
        """Show or hide a button"""
        if button_id in self.button_definitions:
            self.button_definitions[button_id].visible = visible
            if button_id in self.buttons:
                self.buttons[button_id].layout.display = 'flex' if visible else 'none'
            self._update_layout()
    
    def set_disabled(self, button_id: str, disabled: bool = True) -> None:
        """Enable or disable a button"""
        if button_id in self.buttons:
            self.buttons[button_id].disabled = disabled
    
    def on_click(self, button_id: str, callback) -> None:
        """Register a click handler for a button"""
        if button_id in self.buttons:
            self.buttons[button_id].on_click(callback)
    
    def display(self) -> None:
        """Display the button group"""
        display(self.container)

def create_action_buttons(
    primary_label: str = "Download",
    primary_icon: str = "download",
    secondary_buttons: Optional[List[Tuple[str, str, str]]] = None,
    button_width: str = '140px',
    container_width: str = '100%',
    primary_style: str = 'primary',
    cleanup_enabled: bool = False,
    cleanup_label: str = "Cleanup",
    cleanup_tooltip: str = "Hapus data yang sudah ada"
) -> Dict[str, Any]:
    """
    Create action buttons with a flexible layout (legacy compatibility function)
    
    Args:
        primary_label: Text for the primary button
        primary_icon: Icon for the primary button
        secondary_buttons: List of (label, icon, style) tuples for secondary buttons
        button_width: Width of the buttons
        container_width: Width of the button container
        primary_style: Style for the primary button
        cleanup_enabled: Whether to show the cleanup button
        cleanup_label: Label for the cleanup button
        cleanup_tooltip: Tooltip for the cleanup button
        
    Returns:
        Dictionary containing button widgets and container
    """
    action_buttons = ActionButtons(container_width=container_width)
    
    # Add primary button
    action_buttons.add_button(
        button_id='primary',
        label=primary_label,
        icon=primary_icon,
        style=primary_style,
        width=button_width,
        order=0
    )
    
    # Add secondary buttons
    secondary_buttons = secondary_buttons or []
    for idx, (label, icon, style) in enumerate(secondary_buttons, 1):
        action_buttons.add_button(
            button_id=f'secondary_{idx}',
            label=label,
            icon=icon,
            style=style,
            width=button_width,
            order=idx
        )
    
    # Add cleanup button if enabled
    if cleanup_enabled:
        action_buttons.add_button(
            button_id='cleanup',
            label=cleanup_label,
            icon='trash',
            style='warning',
            tooltip=cleanup_tooltip,
            width=button_width,
            order=100
        )
    
    # Return a dictionary with all buttons for backward compatibility
    result = {
        'container': action_buttons.container,
        'primary_button': action_buttons.get_button('primary'),
        'secondary_buttons': [action_buttons.get_button(btn_id) 
                            for btn_id in action_buttons.buttons 
                            if btn_id.startswith('secondary_')],
        'cleanup_button': action_buttons.get_button('cleanup'),
        'action_buttons': action_buttons,  # For advanced usage
        'buttons': list(action_buttons.buttons.values())  # For backward compatibility
    }
    
    # Add direct button references for backward compatibility
    if 'primary' in action_buttons.buttons:
        result['download_button'] = action_buttons.get_button('primary')
    if 'secondary_1' in action_buttons.buttons:
        result['check_button'] = action_buttons.get_button('secondary_1')
    
    return result