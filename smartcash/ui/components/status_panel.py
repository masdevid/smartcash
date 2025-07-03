"""
Status panel component for displaying status messages with consistent styling and emoji handling.
"""

import re
from typing import Dict, Any, Optional, Tuple
import ipywidgets as widgets
from .base_component import BaseUIComponent


def _filter_emoji(message: str) -> Tuple[str, str]:
    """Extract first emoji from message if any.
    
    Returns:
        Tuple of (emoji, cleaned_message) - First emoji found and cleaned message
    """
    # Regex to match most Unicode emojis
    emoji_pattern = re.compile(
        r'[\U0001F1E0-\U0001F1FF]|'  # flags
        r'[\U0001F300-\U0001F5FF]|'  # symbols & pictographs
        r'[\U0001F600-\U0001F64F]|'  # emoticons
        r'[\U0001F680-\U0001F6FF]|'  # transport & map symbols
        r'[\U0001F700-\U0001F77F]|'  # alchemical symbols
        r'[\U0001F780-\U0001F7FF]|'  # Geometric Shapes
        r'[\U0001F800-\U0001F8FF]|'  # Supplemental Arrows-C
        r'[\U0001F900-\U0001F9FF]|'  # Supplemental Symbols and Pictographs
        r'[\U0001FA00-\U0001FA6F]|'  # Chess Symbols
        r'[\U0001FA70-\U0001FAFF]|'  # Symbols and Pictographs Extended-A
        r'[\U00002702-\U000027B0]|'  # Dingbats
        r'[\U000024C2-\U0001F251]'    # Enclosed characters
    )
    
    # Find all emojis in the message
    emojis = emoji_pattern.findall(message)
    
    # If emojis found, return first one and clean message
    if emojis:
        first_emoji = emojis[0]
        # Remove all emojis from message
        cleaned = emoji_pattern.sub('', message).strip()
        return first_emoji, cleaned
    
    return "", message.strip()


class StatusPanel(BaseUIComponent):
    """A status panel component for displaying status messages with consistent styling."""
    
    def __init__(self, 
                 component_name: str = "status_panel",
                 initial_message: str = "",
                 status_type: str = "info",
                 **kwargs):
        """Initialize the status panel.
        
        Args:
            component_name: Unique name for this component
            initial_message: Initial status message to display
            status_type: Type of status (info, success, warning, error)
            **kwargs: Additional arguments to pass to BaseUIComponent
        """
        super().__init__(component_name, **kwargs)
        self._initial_message = initial_message
        self._status_type = status_type
        self._style_map = self._get_style_map()
        
    def _get_style_map(self) -> Dict[str, Dict[str, str]]:
        """Get the style mapping for different status types."""
        return {
            'success': {
                'gradient': 'linear-gradient(135deg, #28a745, #34ce57)',
                'color': 'white',
                'icon': '✅'
            },
            'info': {
                'gradient': 'linear-gradient(135deg, #007bff, #17a2b8)',
                'color': 'white',
                'icon': 'ℹ️'
            },
            'warning': {
                'gradient': 'linear-gradient(135deg, #ffc107, #fd7e14)',
                'color': 'black',
                'icon': '⚠️'
            },
            'error': {
                'gradient': 'linear-gradient(135deg, #dc3545, #c82333)',
                'color': 'white',
                'icon': '❌'
            },
            'default': {
                'gradient': 'linear-gradient(135deg, #6c757d, #5a6268)',
                'color': 'white',
                'icon': 'ℹ️'
            }
        }
    
    def _create_ui_components(self) -> None:
        """Create the status panel UI components."""
        # Create the status HTML widget
        self._ui_components['status_html'] = widgets.HTML(
            value=self._get_status_html(self._initial_message, self._status_type),
            layout=widgets.Layout(width='100%')
        )
        
        # Create container
        self._ui_components['container'] = widgets.VBox(
            [self._ui_components['status_html']],
            layout=widgets.Layout(
                width='100%',
                margin='5px 0',
                padding='0'
            )
        )
        
        # Add CSS classes for styling
        self._ui_components['status_html'].add_class('smartcash-status-panel')
        self._ui_components['status_html'].add_class(f'status-{self._status_type}')
    
    def _get_status_html(self, message: str, status_type: str) -> str:
        """Generate HTML for the status message."""
        style = self._style_map.get(status_type, self._style_map['default'])
        emoji, clean_msg = _filter_emoji(message)
        display_emoji = emoji or style.get('icon', 'ℹ️')
        
        return f"""
        <div style="
            padding: 10px; 
            border-radius: 8px; 
            background-color: {style['color']}; 
            color: white; 
            font-weight: bold;
            word-wrap: break-word;
            white-space: normal;
            overflow-wrap: break-word;
            max-width: 100%;
            box-sizing: border-box;
            display: inline-block;
            width: 100%;
        ">
            {display_emoji} {clean_msg}
        </div>
        """
    
    def update_status(self, message: str, status_type: str = None) -> None:
        """Update the status message and optionally change the status type.
        
        Args:
            message: New status message to display
            status_type: Optional new status type (info, success, warning, error)
        """
        if not self._initialized:
            self.initialize()
            
        if status_type is not None:
            self._status_type = status_type
            
        status_html = self._ui_components.get('status_html')
        if status_html:
            # Update status type classes if changed
            if status_type is not None and hasattr(status_html, 'remove_class'):
                for cls in ['status-info', 'status-success', 'status-warning', 'status-error']:
                    status_html.remove_class(cls)
                status_html.add_class(f'status-{status_type}')
            
            # Update the content
            status_html.value = self._get_status_html(message, self._status_type)
    
    def clear(self) -> None:
        """Clear the status message."""
        if self._initialized and 'status_html' in self._ui_components:
            self._ui_components['status_html'].value = ""


# Backward compatibility functions
def create_status_panel(message: str = "", status_type: str = "info", layout: Optional[Dict[str, Any]] = None) -> widgets.HTML:
    """Legacy function to create a status panel (for backward compatibility)."""
    panel = StatusPanel("legacy_status_panel", message, status_type)
    return panel.show()


def update_status_panel(panel: widgets.HTML, message: str, status_type: str = "info") -> None:
    """Legacy function to update a status panel (for backward compatibility)."""
    # This is a simplified version that works with the legacy API
    try:
        style_map = {
            'success': {'gradient': 'linear-gradient(135deg, #28a745, #34ce57)', 'color': 'white', 'icon': '✅'},
            'info': {'gradient': 'linear-gradient(135deg, #007bff, #17a2b8)', 'color': 'white', 'icon': 'ℹ️'},
            'warning': {'gradient': 'linear-gradient(135deg, #ffc107, #fd7e14)', 'color': 'black', 'icon': '⚠️'},
            'error': {'gradient': 'linear-gradient(135deg, #dc3545, #c82333)', 'color': 'white', 'icon': '❌'}
        }
        
        style = style_map.get(status_type, style_map['info'])
        emoji, clean_msg = _filter_emoji(message)
        display_emoji = emoji or style.get('icon', 'ℹ️')
        
        panel.value = f"""
        <div style="
            padding: 8px 12px;
            background: {style['gradient']};
            color: {style['color']};
            border-radius: 4px;
            margin: 5px 0;
            font-weight: 500;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            background-size: 200% 200%;
            animation: gradient 3s ease infinite;
        ">
            {display_emoji} {clean_msg}
            <style>
                @keyframes gradient {{
                    0% {{ background-position: 0% 50%; }}
                    50% {{ background-position: 100% 50%; }}
                    100% {{ background-position: 0% 50%; }}
                }}
            </style>
        </div>"""
        
        # Update class for styling
        for cls in ['status-info', 'status-success', 'status-warning', 'status-error']:
            if hasattr(panel, 'remove_class'):
                panel.remove_class(cls)
        if hasattr(panel, 'add_class'):
            panel.add_class(f'status-{status_type}')
            
    except Exception as e:
        # Fallback to error message if something goes wrong
        if hasattr(panel, 'value'):
            panel.value = f'<div style="color: red;">Error updating status: {str(e)}</div>'
