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
            
        # Create a copy of details to avoid modifying the original
        summary_details = details.copy()
        
        # Set the status message (this will be displayed in the status area)
        summary_details['status_message'] = status_message
            
        # Ensure we have the mount path if drive is mounted
        if summary_details.get('drive_mounted') and 'mount_path' not in summary_details:
            if 'drive_mount_path' in summary_details and summary_details['drive_mount_path']:
                summary_details['mount_path'] = summary_details['drive_mount_path']
        
        # Generate the content with the updated details
        content = _format_summary_content(summary_details)
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
    from smartcash.ui.setup.env_config.constants import REQUIRED_FOLDERS, SYMLINK_MAP
    
    # Get verification data
    verified_folders = data.get('verified_folders', [])
    missing_folders = data.get('missing_folders', [])
    verified_symlinks = data.get('verified_symlinks', [])
    missing_symlinks = data.get('missing_symlinks', [])
    
    # Calculate counts
    folders_verified = len(verified_folders)
    folders_required = len(REQUIRED_FOLDERS)
    symlinks_verified = len(verified_symlinks)
    symlinks_required = len(SYMLINK_MAP)
    configs_synced = data.get('configs_synced', 0)
    
    # Format status indicators
    folders_status = "✅" if not missing_folders else "⚠️"
    symlinks_status = "✅" if not missing_symlinks else "⚠️"
    configs_status = "✅" if configs_synced > 0 else "⚠️"
    drive_status_icon = "✅" if data.get('drive_mounted') else "❌"
    
    # Format info with verification results
    folders_info = f"{folders_status} {folders_verified}/{folders_required} folders verified"
    if missing_folders:
        folders_info += f"<br><small style='color:#d32f2f;'>{len(missing_folders)} missing</small>"
    
    symlinks_info = f"{symlinks_status} {symlinks_verified}/{symlinks_required} symlinks verified"
    if missing_symlinks:
        symlinks_info += f"<br><small style='color:#d32f2f;'>{len(missing_symlinks)} missing</small>"
    
    configs_info = f"{configs_status} {configs_synced} configs synced"
    
    # Format status message
    status_message = data.get('status_message', 'Setup completed')
    
    # Determine overall status
    all_verified = (
        not missing_folders and
        not missing_symlinks and
        configs_synced > 0 and
        data.get('drive_mounted', False)
    )
    overall_status = "✅ All checks passed" if all_verified else "⚠️ Some checks failed"
    
    # Create details sections
    details_sections = []
    
    # Add missing folders section if any
    if missing_folders:
        missing_folders_list = "\n".join(f"<li>{folder}</li>" for folder in missing_folders)
        details_sections.append(f"""
        <div style="margin-top: 10px; background: #fff3e0; padding: 10px; border-radius: 4px;">
            <h5 style="margin: 0 0 5px 0; color: #e65100;">⚠️ Missing Folders</h5>
            <ul style="margin: 5px 0 0 0; padding-left: 20px; font-size: 0.9em;">
                {missing_folders_list}
            </ul>
        </div>
        """)
    
    # Add missing symlinks section if any
    if missing_symlinks:
        missing_symlinks_list = "\n".join(f"<li>{src} → {dst}</li>" for src, dst in missing_symlinks)
        details_sections.append(f"""
        <div style="margin-top: 10px; background: #fff3e0; padding: 10px; border-radius: 4px;">
            <h5 style="margin: 0 0 5px 0; color: #e65100;">⚠️ Missing Symlinks</h5>
            <ul style="margin: 5px 0 0 0; padding-left: 20px; font-size: 0.9em;">
                {missing_symlinks_list}
            </ul>
        </div>
        """)
    
    # Combine all sections
    details_html = "\n".join(details_sections)
    
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
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 16px;
        ">
            <div style="
                background: #ffffff;
                border-radius: 6px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                padding: 12px;
                border-left: 4px solid #2196f3;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 500; color: #333;">Google Drive</span>
                    <span style="font-size: 0.9em; color: #666;">
                        {drive_status_icon} {data.get('mount_path', 'N/A') if data.get('drive_mounted') else 'Not mounted'}
                    </span>
                </div>
                <div style="font-size: 0.9em; color: #666;">
                    {overall_status}
                </div>
            </div>
            
            <div style="
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 8px;
                margin-top: 4px;
            ">
                <div style="
                    background: #ffffff;
                    border-radius: 6px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    padding: 12px;
                    border-left: 4px solid #4caf50;
                ">
                    <div style="font-weight: 500; color: #333; margin-bottom: 4px;">Folders</div>
                    <div style="font-size: 0.9em; color: #666;">{folders_info}</div>
                </div>
                
                <div style="
                    background: #ffffff;
                    border-radius: 6px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    padding: 12px;
                    border-left: 4px solid #ff9800;
                ">
                    <div style="font-weight: 500; color: #333; margin-bottom: 4px;">Symlinks</div>
                    <div style="font-size: 0.9em; color: #666;">{symlinks_info}</div>
                </div>
                
                <div style="
                    background: #ffffff;
                    border-radius: 6px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    padding: 12px;
                    grid-column: span 2;
                    border-left: 4px solid #9c27b0;
                ">
                    <div style="font-weight: 500; color: #333; margin-bottom: 4px;">Configurations</div>
                    <div style="font-size: 0.9em; color: #666;">{configs_info}</div>
                </div>
            </div>
        </div>
        
        {details_html}
        
        <div style="
            margin-top: 12px;
            background: #f0f7ff;
            padding: 12px;
            border-radius: 6px;
            border-left: 4px solid #2196f3;
        ">
            <p style="margin: 0; font-size: 0.9em; color: #0d47a1; font-style: italic;">
                {status_message}
            </p>
        </div>
    </div>
    """