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
        content += _format_enhanced_summary_content(details)
    
    content += "</div>"
    summary_widget.value = content

def update_setup_summary_with_verification(
    summary_widget: widgets.HTML,
    verification_results: Dict[str, Any],
    system_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Update setup summary with comprehensive verification results.
    
    Args:
        summary_widget: The HTML widget to update
        verification_results: Results from verification operation
        system_info: Optional system information from env_detector
    """
    success = verification_results.get('success', False)
    issues = verification_results.get('issues', [])
    verification = verification_results.get('verification', {})
    
    # Determine overall status
    status_type = 'success' if success else 'error'
    main_message = "Environment Setup Complete" if success else f"Setup Issues Found ({len(issues)} issues)"
    
    # Build content with verification details
    color = STATUS_COLORS.get(status_type, STATUS_COLORS['default'])
    icon = _get_status_icon(status_type)
    
    content = f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6;">
        <div style="margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #e0e0e0;">
            <h3 style="margin: 0; color: {color};">{icon} {main_message}</h3>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
            <div>
                <h4 style="color: #333; margin: 0 0 10px 0; padding-bottom: 5px; border-bottom: 2px solid #2196f3;">
                    🔧 Setup Components
                </h4>
                {_format_verification_status(verification)}
            </div>
            
            <div>
                <h4 style="color: #333; margin: 0 0 10px 0; padding-bottom: 5px; border-bottom: 2px solid #4caf50;">
                    💻 System Status
                </h4>
                {_format_system_status(system_info)}
            </div>
        </div>
        
        {_format_issues_section(issues) if issues else ""}
    </div>
    """
    
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

def _format_enhanced_summary_content(data: Dict) -> str:
    """
    Format enhanced summary data for display in the summary widget
    
    Args:
        data: Dictionary containing summary data
        
    Returns:
        Formatted HTML string
    """
    if not isinstance(data, dict):
        return "<p>No summary data available</p>"
    
    content = []
    
    # Format each section with enhanced styling
    for section, items in data.items():
        if not items:
            continue
            
        # Add section header with color coding
        section_color = _get_section_color(section)
        content.append(f'<h4 style="margin: 15px 0 10px 0; color: {section_color}; border-left: 3px solid {section_color}; padding-left: 10px;">{section}</h4>')
        
        # Add items as a styled list
        content.append('<div style="margin: 0 0 15px 15px;">')
        
        if isinstance(items, dict):
            for key, value in items.items():
                status_icon, status_color = _get_value_status(value)
                content.append(f'<p style="margin: 5px 0; color: {status_color};"><strong>{key}:</strong> {status_icon} {value}</p>')
        elif isinstance(items, (list, tuple)):
            for item in items:
                content.append(f'<p style="margin: 5px 0;">• {item}</p>')
        else:
            content.append(f'<p style="margin: 5px 0;">{items}</p>')
        
        content.append('</div>')
    
    return '\n'.join(content)

def _format_verification_status(verification: Dict[str, Any]) -> str:
    """Format verification status for display.
    
    Args:
        verification: Verification results dictionary
        
    Returns:
        Formatted HTML string
    """
    content = []
    
    # Drive mount status
    drive_mount = verification.get('drive_mount', {})
    drive_status = "✅ Mounted" if drive_mount.get('mounted', False) else "❌ Not mounted"
    if drive_mount.get('write_access', False):
        drive_status += " (writable)"
    content.append(f'<p><strong>Drive:</strong> {drive_status}</p>')
    
    # Symlinks status
    symlinks = verification.get('symlinks', {})
    valid_count = symlinks.get('valid_count', 0)
    total_count = symlinks.get('total_count', 0)
    symlink_status = f"✅ {valid_count}/{total_count}" if valid_count == total_count else f"⚠️ {valid_count}/{total_count}"
    content.append(f'<p><strong>Symlinks:</strong> {symlink_status}</p>')
    
    # Folders status
    folders = verification.get('folders', {})
    existing_count = folders.get('existing_count', 0)
    total_folders = folders.get('total_count', 0)
    folder_status = f"✅ {existing_count}/{total_folders}" if existing_count == total_folders else f"⚠️ {existing_count}/{total_folders}"
    content.append(f'<p><strong>Folders:</strong> {folder_status}</p>')
    
    # Environment variables status
    env_vars = verification.get('env_vars', {})
    valid_env_count = env_vars.get('valid_count', 0)
    total_env_count = env_vars.get('total_count', 0)
    env_status = f"✅ {valid_env_count}/{total_env_count}" if valid_env_count == total_env_count else f"⚠️ {valid_env_count}/{total_env_count}"
    content.append(f'<p><strong>Environment:</strong> {env_status}</p>')
    
    return '\n'.join(content)

def _format_system_status(system_info: Optional[Dict[str, Any]]) -> str:
    """Format system status for display.
    
    Args:
        system_info: System information dictionary
        
    Returns:
        Formatted HTML string
    """
    if not system_info:
        return '<p>System information not available</p>'
    
    content = []
    
    # Environment type
    env_type = system_info.get('environment', {}).get('type', 'unknown')
    content.append(f'<p><strong>Runtime:</strong> {env_type.capitalize()}</p>')
    
    # Hardware info
    hardware = system_info.get('hardware', {})
    cpu_cores = hardware.get('cpu_cores', 'N/A')
    ram_gb = hardware.get('total_ram_gb', 0)
    content.append(f'<p><strong>CPU:</strong> {cpu_cores} cores</p>')
    content.append(f'<p><strong>RAM:</strong> {ram_gb:.1f}GB</p>')
    
    # GPU info
    gpu_info = hardware.get('gpu_info', 'No GPU')
    gpu_status = "✅" if "available" not in gpu_info.lower() or "no gpu" not in gpu_info.lower() else "❌"
    content.append(f'<p><strong>GPU:</strong> {gpu_status} {gpu_info}</p>')
    
    # Storage info
    storage = system_info.get('storage', {})
    if storage:
        free_gb = storage.get('free_gb', 0)
        total_gb = storage.get('total_gb', 0)
        if total_gb > 0:
            content.append(f'<p><strong>Storage:</strong> {free_gb:.1f}GB free of {total_gb:.1f}GB</p>')
    
    return '\n'.join(content)

def _format_issues_section(issues: list) -> str:
    """Format issues section for display.
    
    Args:
        issues: List of issue strings
        
    Returns:
        Formatted HTML string for issues section
    """
    if not issues:
        return ""
    
    content = f"""
    <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 6px;">
        <h4 style="color: #856404; margin: 0 0 10px 0;">⚠️ Issues Found ({len(issues)})</h4>
        <ul style="margin: 0; padding-left: 20px; color: #856404;">
    """
    
    for issue in issues:
        content += f'<li style="margin: 5px 0;">{issue}</li>'
    
    content += """
        </ul>
    </div>
    """
    
    return content

def _get_section_color(section: str) -> str:
    """Get color for section headers.
    
    Args:
        section: Section name
        
    Returns:
        Color code for the section
    """
    color_map = {
        'system': '#2196f3',
        'resources': '#4caf50', 
        'verification': '#ff9800',
        'setup': '#9c27b0',
        'environment': '#607d8b'
    }
    
    section_lower = section.lower()
    for key, color in color_map.items():
        if key in section_lower:
            return color
    
    return '#333'

def _get_value_status(value) -> tuple:
    """Get status icon and color for a value.
    
    Args:
        value: Value to evaluate
        
    Returns:
        Tuple of (icon, color)
    """
    if value is True:
        return ('✅', '#4caf50')
    elif value is False or value is None:
        return ('❌', '#f44336')
    elif isinstance(value, str) and ('error' in value.lower() or 'failed' in value.lower()):
        return ('❌', '#f44336')
    elif isinstance(value, str) and ('warning' in value.lower() or 'partial' in value.lower()):
        return ('⚠️', '#ff9800')
    elif isinstance(value, str) and ('success' in value.lower() or 'complete' in value.lower()):
        return ('✅', '#4caf50')
    else:
        return ('ℹ️', '#333')

def _format_summary_content(data: Dict) -> str:
    """
    Format summary data for display in the summary widget (legacy function)
    
    Args:
        data: Dictionary containing summary data
        
    Returns:
        Formatted HTML string
    """
    # Delegate to enhanced version for better formatting
    return _format_enhanced_summary_content(data)
