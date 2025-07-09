"""
Modern and minimal status panel component for displaying status messages.
"""

import re
from typing import Dict, Any, Optional, Tuple
import ipywidgets as widgets
from .base_component import BaseUIComponent


def _extract_emoji(message: str) -> Tuple[str, str]:
    """Extract first emoji from message if any."""
    emoji_pattern = re.compile(
        r'[\U0001F1E0-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]'
    )
    emojis = emoji_pattern.findall(message)
    return (emojis[0], emoji_pattern.sub('', message).strip()) if emojis else ("", message.strip())


class StatusPanel(BaseUIComponent):
    """A modern and minimal status panel for displaying status messages."""
    
    def __init__(self, 
                 component_name: str = "status_panel",
                 initial_message: str = "",
                 status_type: str = "info",
                 **kwargs):
        super().__init__(component_name, **kwargs)
        self._initial_message = initial_message
        self._status_type = status_type
        self._style_map = self._get_style_map()
        
    def _get_style_map(self) -> Dict[str, Dict[str, str]]:
        """Get style configurations for different status types."""
        return {
            'success': {'bg': '#e6f7ed', 'color': '#0d6832', 'icon': '✓'},
            'info': {'bg': '#e6f4ff', 'color': '#0050b3', 'icon': 'i'},
            'warning': {'bg': '#fff8e6', 'color': '#ad6800', 'icon': '!'},
            'error': {'bg': '#fff1f0', 'color': '#cf1322', 'icon': '×'},
            'default': {'bg': '#f5f5f5', 'color': '#595959', 'icon': '•'}
        }
    
    def _create_ui_components(self) -> None:
        """Initialize UI components."""
        self._ui_components['status_html'] = widgets.HTML(
            value=self._get_status_html(self._initial_message, self._status_type),
            layout=widgets.Layout(
                width='100%',
                margin='4px 0',
                padding='0'
            )
        )
        
        self._ui_components['container'] = widgets.VBox(
            [self._ui_components['status_html']],
            layout=widgets.Layout(width='100%')
        )
    
    def _get_status_html(self, message: str, status_type: str) -> str:
        """Generate clean HTML for the status message."""
        style = self._style_map.get(status_type, self._style_map['default'])
        emoji, clean_msg = _extract_emoji(message)
        display_icon = emoji or style['icon']
        
        return f"""
        <div style="
            background: {style['bg']};
            color: {style['color']};
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 14px;
            line-height: 1.5;
            display: flex;
            align-items: center;
            gap: 8px;
            border-left: 3px solid {style['color']};
        ">
            <span style="
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: {style['color']};
                color: white;
                font-size: 12px;
                font-weight: bold;
                flex-shrink: 0;
            ">{display_icon}</span>
            <span>{clean_msg}</span>
        </div>
        """
    
    def update_status(self, message: str, status_type: str = None) -> None:
        """Update status message and optionally change type."""
        if not self._initialized:
            self.initialize()
            
        if status_type is not None:
            self._status_type = status_type
            
        if status_html := self._ui_components.get('status_html'):
            status_html.value = self._get_status_html(message, self._status_type)
    
    def clear(self) -> None:
        """Clear the status message."""
        if self._initialized and 'status_html' in self._ui_components:
            self._ui_components['status_html'].value = ""


# Backward compatibility
def create_status_panel(message: str = "", status_type: str = "info", **_) -> widgets.HTML:
    panel = StatusPanel("legacy_status_panel", message, status_type)
    return panel.show()

def update_status_panel(panel: widgets.HTML, message: str, status_type: str = "info") -> None:
    """Legacy update function for backward compatibility."""
    try:
        if hasattr(panel, 'value'):
            style_map = {
                'success': {'bg': '#e6f7ed', 'color': '#0d6832', 'icon': '✓'},
                'info': {'bg': '#e6f4ff', 'color': '#0050b3', 'icon': 'i'},
                'warning': {'bg': '#fff8e6', 'color': '#ad6800', 'icon': '!'},
                'error': {'bg': '#fff1f0', 'color': '#cf1322', 'icon': '×'}
            }
            style = style_map.get(status_type, style_map['info'])
            emoji, clean_msg = _extract_emoji(message)
            display_icon = emoji or style['icon']
            
            panel.value = f"""
            <div style="
                background: {style['bg']};
                color: {style['color']};
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 14px;
                line-height: 1.5;
                display: flex;
                align-items: center;
                gap: 8px;
                border-left: 3px solid {style['color']};
            ">
                <span style="
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    width: 20px;
                    height: 20px;
                    border-radius: 50%;
                    background: {style['color']};
                    color: white;
                    font-size: 12px;
                    font-weight: bold;
                    flex-shrink: 0;
                ">{display_icon}</span>
                <span>{clean_msg}</span>
            </div>
            """
    except Exception as e:
        # Fallback to error message if something goes wrong
        if hasattr(panel, 'value'):
            panel.value = f'<div style="color: red;">Error updating status: {str(e)}</div>'
