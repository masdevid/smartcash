"""
Alert Components

Reusable alert components for displaying error, warning, success, and info messages.
"""

import ipywidgets as widgets
from typing import Optional, List, Dict, Any
from enum import Enum

class AlertType(Enum):
    """Alert type enum for different alert styles."""
    ERROR = 'error'
    WARNING = 'warning'
    SUCCESS = 'success'
    INFO = 'info'

ALERT_STYLES = {
    AlertType.ERROR: {
        'icon': '❌',
        'bg_color': '#ffebee',
        'border_color': '#f44336',
        'text_color': '#c62828',
        'title': 'Error'
    },
    AlertType.WARNING: {
        'icon': '⚠️',
        'bg_color': '#fff8e1',
        'border_color': '#ffc107',
        'text_color': '#ff8f00',
        'title': 'Warning'
    },
    AlertType.SUCCESS: {
        'icon': '✅',
        'bg_color': '#e8f5e9',
        'border_color': '#4caf50',
        'text_color': '#2e7d32',
        'title': 'Success'
    },
    AlertType.INFO: {
        'icon': 'ℹ️',
        'bg_color': '#e3f2fd',
        'border_color': '#2196f3',
        'text_color': '#1565c0',
        'title': 'Information'
    }
}

def create_alert(
    message: str,
    alert_type: AlertType = AlertType.INFO,
    title: Optional[str] = None,
    solution: Optional[str] = None,
    width: str = '100%',
    margin: str = '0 0 20px 0',
    padding: str = '16px',
    border_radius: str = '8px',
    show_icon: bool = True,
    show_title: bool = True,
    **kwargs
) -> widgets.VBox:
    """
    Create a styled alert box.
    
    Args:
        message: The main message to display
        alert_type: Type of alert (error, warning, success, info)
        title: Optional custom title (defaults to type-based title)
        solution: Optional solution text to display below the message
        width: Width of the alert box
        margin: CSS margin
        padding: CSS padding
        border_radius: Border radius
        show_icon: Whether to show the icon
        show_title: Whether to show the title
        **kwargs: Additional style overrides
        
    Returns:
        VBox containing the alert
    """
    style = ALERT_STYLES.get(alert_type, ALERT_STYLES[AlertType.INFO])
    
    # Prepare title section
    title_section = ''
    if show_title:
        title_text = title or style['title']
        title_section = f"""
        <h3 style="
            margin: 0 0 12px 0;
            color: {style['text_color']};
            font-size: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        ">
            {style['icon'] if show_icon else ''}
            {title_text}
        </h3>
        """
    
    # Prepare solution section
    solution_section = f"""
    <div style="
        margin-top: 12px;
        padding-top: 8px;
        border-top: 1px solid rgba(0, 0, 0, 0.1);
    ">
        <strong>Solusi:</strong> {solution}
    </div>
    """ if solution else ''
    
    # Build the alert HTML
    html = f"""
    <div style="
        background: {bg_color};
        border: 1px solid {border_color};
        border-radius: {border_radius};
        color: {text_color};
        padding: {padding};
        font-size: 14px;
        line-height: 1.5;
    ">
        {title_section}
        <div style="margin: 0;">{message}</div>
        {solution_section}
    </div>
    """.format(
        bg_color=kwargs.get('bg_color', style['bg_color']),
        border_color=kwargs.get('border_color', style['border_color']),
        text_color=kwargs.get('text_color', style['text_color']),
        title_section=title_section,
        solution_solution=solution_section,
        border_radius=border_radius,
        padding=padding
    )
    
    return widgets.VBox([
        widgets.HTML(
            value=html,
            layout=widgets.Layout(width='100%')
        )
    ], layout=widgets.Layout(width=width, margin=margin))

# Convenience functions for common alert types
def error_alert(message: str, **kwargs) -> widgets.VBox:
    """Create an error alert."""
    return create_alert(message, AlertType.ERROR, **kwargs)

def warning_alert(message: str, **kwargs) -> widgets.VBox:
    """Create a warning alert."""
    return create_alert(message, AlertType.WARNING, **kwargs)

def success_alert(message: str, **kwargs) -> widgets.VBox:
    """Create a success alert."""
    return create_alert(message, AlertType.SUCCESS, **kwargs)

def info_alert(message: str, **kwargs) -> widgets.VBox:
    """Create an info alert."""
    return create_alert(message, AlertType.INFO, **kwargs)
