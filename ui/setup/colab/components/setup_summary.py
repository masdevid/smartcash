"""
file_path: smartcash/ui/setup/colab/components/setup_summary.py
Component for displaying environment setup summary with status updates
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Union

# Status type definitions for better type hints
StatusType = str  # 'success', 'warning', 'error', 'info', etc.

# Color mapping for different status types
STATUS_COLORS = {
    'success': '#4caf50',  # Green
    'warning': '#ff9800',  # Orange
    'error': '#f44336',    # Red
    'info': '#2196f3',     # Blue
    'default': '#9e9e9e'   # Grey
}

def create_setup_summary(initial_message: Optional[str] = None) -> widgets.HTML:
    """
    Create a setup summary widget with optional initial message
    
    Args:
        initial_message: Optional message to display initially
        
    Returns:
        HTML widget for displaying setup summary
    """
    return widgets.HTML(
        value=_get_initial_summary_content(initial_message),
        layout=widgets.Layout(
            width='100%',
            padding='15px',
            border='1px solid #e0e0e0',
            border_radius='6px',
            margin='10px 0',
            background='#f9f9f9'
        )
    )

def update_setup_summary(
    summary_widget: widgets.HTML, 
    status_message: str, 
    status_type: StatusType = 'info',
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Update the setup summary with new status information
    
    Args:
        summary_widget: The HTML widget to update
        status_message: Main status message to display
        status_type: Type of status (affects color and icon)
        details: Optional dictionary with additional details to display
    """
    color = STATUS_COLORS.get(status_type, STATUS_COLORS['default'])
    
    # Create the main status line
    icon = _get_status_icon(status_type)
    content = f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">
        <div style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #e0e0e0;">
            <h3 style="margin: 0; color: {color};">{icon} {status_message}</h3>
        </div>
    """
    
    # Add details if provided
    if details:
        content += _format_summary_content(details)
    
    content += "</div>"
    summary_widget.value = content

def _get_initial_summary_content(message: Optional[str] = None) -> str:
    """
    Generate the initial summary content
    
    Args:
        message: Optional custom message to display
        
    Returns:
        HTML content for the initial state
    """
    default_message = """
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">
        <div style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #e0e0e0;">
            <h3 style="margin: 0; color: #9e9e9e;">ℹ️ Waiting for setup to begin...</h3>
        </div>
        <p>Click the "Setup Environment" button to start the configuration process.</p>
    </div>
    """
    
    if message:
        return f"""
        <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">
            <div style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #e0e0e0;">
                <h3 style="margin: 0; color: #9e9e9e;">ℹ️ {message}</h3>
            </div>
            <p>Click the "Setup Environment" button to start the configuration process.</p>
        </div>
        """
    
    return default_message

def _get_status_icon(status_type: StatusType) -> str:
    """Get appropriate icon for status type"""
    icons = {
        'success': '✅',
        'warning': '⚠️',
        'error': '❌',
        'info': 'ℹ️',
        'default': 'ℹ️'
    }
    return icons.get(status_type, icons['default'])

def _format_summary_content(data: Dict) -> str:
    """
    Format summary data for display in the summary widget
    
    Args:
        data: Dictionary containing summary data
        
    Returns:
        Formatted HTML string
    """
    if not isinstance(data, dict):
        return "<p>No summary data available</p>"
    
    content = []
    
    # Format each section
    for section, items in data.items():
        if not items:
            continue
            
        # Add section header
        content.append(f'<h4 style="margin: 15px 0 10px 0; color: #333;">{section}</h4>')
        
        # Add items as a list
        content.append('<ul style="margin: 0 0 10px 0; padding-left: 20px;">')
        
        if isinstance(items, dict):
            for key, value in items.items():
                if value is True:
                    content.append(f'<li><strong>{key}:</strong> ✅</li>')
                elif value is False or value is None:
                    content.append(f'<li><strong>{key}:</strong> ❌</li>')
                else:
                    content.append(f'<li><strong>{key}:</strong> {value}</li>')
        elif isinstance(items, (list, tuple)):
            for item in items:
                content.append(f'<li>{item}</li>')
        else:
            content.append(f'<li>{items}</li>')
        
        content.append('</ul>')
    
    return '\n'.join(content)
