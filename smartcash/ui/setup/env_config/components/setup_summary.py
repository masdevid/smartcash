"""
File: smartcash/ui/setup/env_config/components/setup_summary.py
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
        summary_widget: The widget to update
        status_message: Message to display
        status_type: Type of status ('success', 'warning', 'error', 'info')
        details: Optional dictionary with additional details to display
    """
    try:
        content = _format_summary_content(status_message, status_type, details)
        summary_widget.value = content
    except Exception as e:
        error_html = """
        <div style='color: #f44336; padding: 10px; background: #ffebee; border-radius: 4px;'>
            <strong>❌ Error updating summary:</strong> {error}
        </div>
        """.format(error=str(e))
        summary_widget.value = error_html

def _get_initial_summary_content(message: Optional[str] = None) -> str:
    """
    Generate the initial summary content
    
    Args:
        message: Optional custom message to display
        
    Returns:
        HTML content for the initial state
    """
    default_message = message or "Waiting for setup to complete..."
    return f"""
    <div style="
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        color: #424242;
    ">
        <div style="
            display: flex;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e0e0e0;
        ">
            <span style="
                font-size: 1.2em;
                font-weight: 600;
                color: #2196f3;
            ">📋 Setup Summary</span>
        </div>
        <div style="
            padding: 12px;
            background: #ffffff;
            border-radius: 4px;
            border-left: 4px solid #2196f3;
            margin: 8px 0;
        ">
            <p style="margin: 0; color: #757575; font-style: italic;">
                {default_message}
            </p>
        </div>
    </div>
    """

def _format_summary_content(
    status_message: str,
    status_type: StatusType = 'info',
    details: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format the summary content with status information
    
    Args:
        status_message: The main status message to display
        status_type: Type of status ('success', 'warning', 'error', 'info')
        details: Optional dictionary with additional details
        
    Returns:
        Formatted HTML content for the summary
    """
    # Validate status type
    status_type = status_type.lower() if status_type else 'info'
    if status_type not in STATUS_COLORS:
        status_type = 'info'
    
    # Get status styling
    status_color = STATUS_COLORS.get(status_type, STATUS_COLORS['default'])
    status_icon = {
        'success': '✅',
        'warning': '⚠️',
        'error': '❌',
        'info': 'ℹ️'
    }.get(status_type, 'ℹ️')
    
    # Format details if provided
    details_html = ""
    if details and isinstance(details, dict):
        details_items = []
        for key, value in details.items():
            if isinstance(value, dict):
                # Handle nested dictionaries
                nested_items = ", ".join([f"{k}: {v}" for k, v in value.items()])
                details_items.append(f"<strong>{key}:</strong> {nested_items}")
            elif isinstance(value, (list, tuple)):
                # Handle lists/tuples
                list_items = ", ".join([str(v) for v in value])
                details_items.append(f"<strong>{key}:</strong> {list_items}")
            else:
                details_items.append(f"<strong>{key}:</strong> {value}")
        
        if details_items:
            details_html = """
            <div style="
                margin-top: 12px;
                padding: 10px;
                background: #ffffff;
                border-radius: 4px;
                border: 1px solid #e0e0e0;
                font-size: 0.9em;
                line-height: 1.5;
            ">
                <div style="
                    font-weight: 600;
                    margin-bottom: 8px;
                    color: #616161;
                ">
                    Details:
                </div>
                <div style="padding-left: 10px;">
                    {details_list}
                </div>
            </div>
            """.format(details_list="<br>".join(details_items))
    
    # Format the timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return f"""
    <div style="
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        color: #424242;
    ">
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e0e0e0;
        ">
            <span style="
                font-size: 1.2em;
                font-weight: 600;
                color: #2196f3;
            ">📋 Setup Summary</span>
            <span style="
                font-size: 0.85em;
                color: #9e9e9e;
            ">{timestamp}</span>
        </div>
        
        <div style="
            padding: 14px;
            background: #ffffff;
            border-radius: 6px;
            border-left: 4px solid {status_color};
            margin-bottom: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        ">
            <div style="display: flex; align-items: flex-start;">
                <span style="
                    font-size: 1.4em;
                    margin-right: 12px;
                    line-height: 1;
                    margin-top: 2px;
                ">{status_icon}</span>
                <div style="flex: 1;">
                    <div style="
                        font-weight: 500;
                        color: {status_color};
                        margin-bottom: 4px;
                        line-height: 1.3;
                    ">{status_message}</div>
                    
                    {details_html}
                </div>
                <p style="margin: 0; font-size: 14px;"><strong>{symlinks_created}</strong> symlinks</p>
            </div>
            
            <div style="background: #f8f9fa; padding: 12px; border-radius: 4px; border-left: 4px solid #9c27b0;">
                <h5 style="margin: 0 0 8px 0; color: #333;">📁 Folders Created</h5>
                <p style="margin: 0; font-size: 14px;"><strong>{folders_created}</strong> directories</p>
            </div>
        </div>
    </div>
    """