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
        # Convert the details to the format expected by _format_summary_content
        if details is None:
            details = {}
            
        # Add status message to details if not present
        if 'status_message' not in details:
            details['status_message'] = status_message
            
        content = _format_summary_content(details)
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

def _format_summary_content(data: Dict) -> str:
    """📊 Format summary data untuk drive mount, config sync, symlinks, dan folders"""
    drive_mounted = data.get('drive_mounted', False)
    mount_path = data.get('mount_path', 'N/A')
    configs_synced = data.get('configs_synced', 0)
    symlinks_created = data.get('symlinks_created', 0)
    folders_created = data.get('folders_created', 0)
    
    drive_status = "✅ Mounted" if drive_mounted else "❌ Not mounted"
    mount_info = f"at {mount_path}" if drive_mounted else ""
    
    return f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">
        <h4 style="color: #2196f3; margin-top: 0; margin-bottom: 15px;">📋 Setup Summary</h4>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
            <div style="background: #f8f9fa; padding: 12px; border-radius: 4px; border-left: 4px solid #2196f3;">
                <h5 style="margin: 0 0 8px 0; color: #333;">💾 Drive Status</h5>
                <p style="margin: 0; font-size: 14px;"><strong>{drive_status}</strong> {mount_info}</p>
            </div>
            
            <div style="background: #f8f9fa; padding: 12px; border-radius: 4px; border-left: 4px solid #4caf50;">
                <h5 style="margin: 0 0 8px 0; color: #333;">⚙️ Configs Synced</h5>
                <p style="margin: 0; font-size: 14px;"><strong>{configs_synced}</strong> config files</p>
            </div>
            
            <div style="background: #f8f9fa; padding: 12px; border-radius: 4px; border-left: 4px solid #ff9800;">
                <h5 style="margin: 0 0 8px 0; color: #333;">🔗 Symlinks Created</h5>
                <p style="margin: 0; font-size: 14px;"><strong>{symlinks_created}</strong> symlinks</p>
            </div>
            
            <div style="background: #f8f9fa; padding: 12px; border-radius: 4px; border-left: 4px solid #9c27b0;">
                <h5 style="margin: 0 0 8px 0; color: #333;">📁 Folders Created</h5>
                <p style="margin: 0; font-size: 14px;"><strong>{folders_created}</strong> directories</p>
            </div>
        </div>
    </div>
    """