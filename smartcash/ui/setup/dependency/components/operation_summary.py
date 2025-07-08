"""
file_path: smartcash/ui/setup/dependency/components/operation_summary.py
Component for displaying dependency operation summary with detailed results
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List
from datetime import datetime

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

# Operation icons
OPERATION_ICONS = {
    'install': '⬇️',
    'uninstall': '🗑️',
    'update': '🔄',
    'check_status': '🔍'
}

def create_operation_summary(initial_message: Optional[str] = None) -> widgets.HTML:
    """
    Create an operation summary widget with optional initial message
    
    Args:
        initial_message: Optional message to display initially
        
    Returns:
        HTML widget for displaying operation summary
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

def update_operation_summary(
    summary_widget: widgets.HTML,
    operation_type: str,
    result: Dict[str, Any],
    status_type: StatusType = 'info'
) -> None:
    """
    Update the operation summary with operation results
    
    Args:
        summary_widget: HTML widget to update
        operation_type: Type of operation (install, uninstall, update, check_status)
        result: Operation result dictionary
        status_type: Type of status for color coding
    """
    try:
        # Generate summary content based on operation type and results
        content = _generate_summary_content(operation_type, result, status_type)
        summary_widget.value = content
        
    except Exception as e:
        summary_widget.value = _get_error_content(f"Error updating summary: {str(e)}")

def _get_initial_summary_content(message: Optional[str] = None) -> str:
    """Generate initial summary content"""
    default_message = message or "🔧 Dependency operations ready..."
    
    return f"""
    <div style="text-align: center; padding: 20px;">
        <div style="font-size: 18px; color: {STATUS_COLORS['info']}; margin-bottom: 10px;">
            {default_message}
        </div>
        <div style="font-size: 14px; color: #666;">
            Select packages and choose an operation to begin
        </div>
    </div>
    """

def _generate_summary_content(operation_type: str, result: Dict[str, Any], status_type: str) -> str:
    """Generate detailed summary content based on operation results"""
    
    # Get operation icon and title
    icon = OPERATION_ICONS.get(operation_type, '🔧')
    title = operation_type.replace('_', ' ').title()
    
    # Determine overall status
    success = result.get('success', False)
    cancelled = result.get('cancelled', False)
    
    if cancelled:
        status_color = STATUS_COLORS['warning']
        status_text = "Cancelled"
    elif success:
        status_color = STATUS_COLORS['success'] 
        status_text = "Completed"
    else:
        status_color = STATUS_COLORS['error']
        status_text = "Failed"
    
    # Generate operation-specific statistics
    stats_html = _generate_operation_stats(operation_type, result)
    
    # Generate duration info
    duration = result.get('duration', 0)
    duration_text = f"⏱️ Duration: {duration:.1f}s" if duration > 0 else ""
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    return f"""
    <div style="padding: 15px;">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <span style="font-size: 24px; margin-right: 10px;">{icon}</span>
            <div>
                <div style="font-size: 18px; font-weight: bold; color: {status_color};">
                    {title} {status_text}
                </div>
                <div style="font-size: 12px; color: #666;">
                    {timestamp} | {duration_text}
                </div>
            </div>
        </div>
        
        {stats_html}
        
        {_generate_error_details(result) if not success and not cancelled else ""}
    </div>
    """

def _generate_operation_stats(operation_type: str, result: Dict[str, Any]) -> str:
    """Generate operation-specific statistics"""
    
    if operation_type == 'install':
        installed = result.get('installed', 0)
        total = result.get('total', 0)
        return f"""
        <div style="background: #f0f8f0; padding: 10px; border-radius: 4px; margin-bottom: 10px;">
            <div style="font-weight: bold; color: #2e7d32;">📦 Installation Results:</div>
            <div>✅ Successfully installed: {installed}/{total} packages</div>
        </div>
        """
        
    elif operation_type == 'uninstall':
        uninstalled = result.get('uninstalled', 0)
        total = result.get('total', 0)
        return f"""
        <div style="background: #fff3e0; padding: 10px; border-radius: 4px; margin-bottom: 10px;">
            <div style="font-weight: bold; color: #f57c00;">🗑️ Uninstallation Results:</div>
            <div>✅ Successfully uninstalled: {uninstalled}/{total} packages</div>
        </div>
        """
        
    elif operation_type == 'update':
        updated = result.get('updated', 0)
        total = result.get('total', 0)
        return f"""
        <div style="background: #e3f2fd; padding: 10px; border-radius: 4px; margin-bottom: 10px;">
            <div style="font-weight: bold; color: #1976d2;">🔄 Update Results:</div>
            <div>✅ Successfully updated: {updated}/{total} packages</div>
        </div>
        """
        
    elif operation_type == 'check_status':
        checked = result.get('checked', 0)
        installed = result.get('installed', 0)
        outdated = result.get('outdated', 0)
        missing = result.get('missing', 0)
        
        return f"""
        <div style="background: #f3e5f5; padding: 10px; border-radius: 4px; margin-bottom: 10px;">
            <div style="font-weight: bold; color: #7b1fa2;">🔍 Status Check Results:</div>
            <div>📊 Total checked: {checked} packages</div>
            <div>✅ Installed: {installed}</div>
            <div>⚠️ Outdated: {outdated}</div>
            <div>❌ Missing: {missing}</div>
        </div>
        """
    
    return ""

def _generate_error_details(result: Dict[str, Any]) -> str:
    """Generate error details if operation failed"""
    error = result.get('error', '')
    if not error:
        return ""
        
    return f"""
    <div style="background: #ffebee; padding: 10px; border-radius: 4px; border-left: 4px solid #f44336;">
        <div style="font-weight: bold; color: #c62828;">❌ Error Details:</div>
        <div style="margin-top: 5px; font-family: monospace; font-size: 12px;">
            {error}
        </div>
    </div>
    """

def _get_error_content(error_message: str) -> str:
    """Generate error content for display issues"""
    return f"""
    <div style="text-align: center; padding: 20px; background: #ffebee; border-radius: 4px;">
        <div style="font-size: 18px; color: {STATUS_COLORS['error']}; margin-bottom: 10px;">
            ❌ Summary Error
        </div>
        <div style="font-size: 14px; color: #666;">
            {error_message}
        </div>
    </div>
    """

# Convenience functions for specific operations
def update_install_summary(summary_widget: widgets.HTML, result: Dict[str, Any]) -> None:
    """Update summary for install operation"""
    status_type = 'success' if result.get('success') else 'error'
    update_operation_summary(summary_widget, 'install', result, status_type)

def update_uninstall_summary(summary_widget: widgets.HTML, result: Dict[str, Any]) -> None:
    """Update summary for uninstall operation"""
    status_type = 'success' if result.get('success') else 'error'
    update_operation_summary(summary_widget, 'uninstall', result, status_type)

def update_update_summary(summary_widget: widgets.HTML, result: Dict[str, Any]) -> None:
    """Update summary for update operation"""
    status_type = 'success' if result.get('success') else 'error'
    update_operation_summary(summary_widget, 'update', result, status_type)

def update_status_summary(summary_widget: widgets.HTML, result: Dict[str, Any]) -> None:
    """Update summary for status check operation"""
    status_type = 'success' if result.get('success') else 'error'
    update_operation_summary(summary_widget, 'check_status', result, status_type)