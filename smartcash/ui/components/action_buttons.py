"""
File: smartcash/ui/components/action_buttons.py
Deskripsi: Simplified action buttons dengan flexible layout dan konfigurasi mudah
"""

from typing import Dict, Any, List, Optional, Union, Literal
import ipywidgets as widgets

def create_action_buttons(
    buttons: List[Dict[str, Any]],
    alignment: Literal['left', 'center', 'right'] = 'left',
    container_width: str = '100%',
    button_spacing: str = '8px',
    container_margin: str = '8px 0'
) -> Dict[str, Any]:
    """Create action buttons with emoji support and duplicate emoji prevention.
    
    Args:
        buttons: List of button configurations
        alignment: Button alignment ('left', 'center', 'right')
        container_width: Width of the button container
        button_spacing: Spacing between buttons
        container_margin: Margin around the container
        
    Returns:
        Dictionary containing container and button widgets
    """
    """
    Create a set of action buttons with flexible layout and configuration.
    
    Args:
        buttons: List of button configurations, each a dictionary with:
            - button_id: Unique identifier for the button (required)
            - text: Button label text (required)
            - icon: Button icon name (optional)
            - style: Button style ('primary', 'success', 'info', 'warning', 'danger', '') (optional)
            - tooltip: Button tooltip text (optional)
            - order: Display order (lower numbers first) (optional)
        alignment: Button container alignment ('left', 'center', 'right')
        container_width: Width of the button container
        button_spacing: Space between buttons
        container_margin: Margin around the button container
        
    Returns:
        Dictionary containing:
            - container: The button container widget
            - buttons: Dictionary of button widgets by button_id
            - count: Number of buttons
    
    Examples:
        >>> buttons = create_action_buttons([
        ...     {'button_id': 'process', 'text': '🚀 Process', 'icon': 'play', 'style': 'primary'},
        ...     {'button_id': 'check', 'text': '🔍 Check', 'style': 'info'},
        ...     {'button_id': 'clean', 'text': '🧹 Clean', 'icon': 'trash', 'style': 'warning'}
        ... ], alignment='center')
        >>> display(buttons['container'])
        
        # Access individual buttons
        >>> process_btn = buttons['buttons']['process']
        >>> process_btn.on_click(lambda b: print('Processing...'))
    """
    # Validate and normalize button configurations
    normalized_buttons = []
    for btn in buttons:
        # Ensure required fields
        if 'button_id' not in btn or 'text' not in btn:
            raise ValueError("Each button must have 'button_id' and 'text' properties")
            
        # Create normalized button config
        normalized_btn = {
            'button_id': btn['button_id'],
            'text': btn['text'],
            'icon': btn.get('icon', ''),
            'style': btn.get('style', ''),
            'tooltip': btn.get('tooltip', f"Click to {btn['text'].lower()}"),
            'order': btn.get('order', 999)  # Default to high number for unspecified order
        }
        normalized_buttons.append(normalized_btn)
    
    # Sort buttons by order
    normalized_buttons.sort(key=lambda b: b['order'])
    
    # Create button widgets
    button_widgets = []
    button_dict = {}
    
    for btn_config in normalized_buttons:
        # Check if text already contains emoji
        import emoji
        text_has_emoji = any(char in emoji.EMOJI_DATA for char in btn_config['text'])
        
        # Only add icon if text doesn't already have an emoji
        if btn_config['icon'] and not text_has_emoji:
            button_text = f"{btn_config['icon']} {btn_config['text']}".strip()
        else:
            button_text = btn_config['text']
        button = widgets.Button(
            description=button_text,
            button_style='',  # Clear default style to use custom styling
            tooltip=btn_config['tooltip'],
            layout=widgets.Layout(
                height='32px',
                margin='0 2px',
                min_width='100px',
                padding='0 12px',
                text_align='center',
                border_radius='4px',
                font_weight='500',
                border='1px solid #ddd'
            ),
            style={
                'button_color': '#f5f5f5',
                'font_weight': '500',
                'text_color': '#333'
            }
        )
        
        # Store button in dictionary by ID
        button_widgets.append(button)
        button_dict[btn_config['button_id']] = button
    
    # Map alignment parameter to justify-content CSS value
    justify_content_map = {
        'left': 'flex-start',
        'center': 'center',
        'right': 'flex-end'
    }
    justify_content = justify_content_map.get(alignment, 'flex-start')  # Default to left alignment
    
    # Create container with flex layout
    container = widgets.HBox(
        button_widgets,
        layout=widgets.Layout(
            width=container_width,
            margin=container_margin,
            display='flex',
            flex_flow='row wrap',
            justify_content=justify_content,
            align_items='center',
            gap=button_spacing
        )
    )
    
    # Return result with container and button references
    result = {
        'container': container,
        'buttons': button_dict,
        'count': len(button_widgets)
    }
    
    return result

# === UTILITY FUNCTIONS ===

def get_button_by_id(action_buttons: Dict[str, Any], button_id: str) -> Optional[widgets.Button]:
    """Get button by its ID"""
    return action_buttons.get('buttons', {}).get(button_id)

def disable_all_buttons(action_buttons: Dict[str, Any]) -> None:
    """Disable all buttons in the action_buttons container"""
    for button in action_buttons.get('buttons', {}).values():
        if hasattr(button, 'disabled'):
            button.disabled = True

def enable_all_buttons(action_buttons: Dict[str, Any]) -> None:
    """Enable all buttons in the action_buttons container"""
    for button in action_buttons.get('buttons', {}).values():
        if hasattr(button, 'disabled'):
            button.disabled = False

def update_button_style(button: widgets.Button, new_style: str, new_text: str = None) -> None:
    """Update button style and text"""
    if hasattr(button, 'button_style'):
        button.button_style = new_style
    
    if new_text and hasattr(button, 'description'):
        button.description = new_text

# === PRESET CONFIGURATIONS ===

def create_preprocessing_buttons() -> Dict[str, Any]:
    """🔧 Preset for preprocessing operations"""
    return create_action_buttons([
        {'button_id': 'start', 'text': '🚀 Mulai Preprocessing', 'icon': 'play', 'style': 'primary', 'order': 1},
        {'button_id': 'check', 'text': '🔍 Check Dataset', 'icon': 'search', 'style': 'info', 'order': 2},
        {'button_id': 'cleanup', 'text': '🧹 Cleanup', 'icon': 'trash', 'style': 'warning', 'order': 3}
    ])

def create_dataset_buttons() -> Dict[str, Any]:
    """📊 Preset for dataset operations"""
    return create_action_buttons([
        {'button_id': 'download', 'text': '📥 Download Dataset', 'icon': 'download', 'style': 'primary', 'order': 1},
        {'button_id': 'analyze', 'text': '📊 Analyze', 'icon': 'search', 'style': 'info', 'order': 2},
        {'button_id': 'clean', 'text': '🧹 Clean', 'icon': 'trash', 'style': 'warning', 'order': 3}
    ])

def create_training_buttons() -> Dict[str, Any]:
    """🚀 Preset for training operations"""
    return create_action_buttons([
        {'button_id': 'start', 'text': '🚀 Start Training', 'icon': 'play', 'style': 'primary', 'order': 1},
        {'button_id': 'pause', 'text': '⏸️ Pause', 'icon': 'pause', 'style': 'warning', 'order': 2},
        {'button_id': 'stop', 'text': '⏹️ Stop', 'icon': 'stop', 'style': 'danger', 'order': 3},
        {'button_id': 'monitor', 'text': '📊 Monitor', 'icon': 'search', 'style': 'info', 'order': 4}
    ])