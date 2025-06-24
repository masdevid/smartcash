"""
Modern Log Accordion Component

A flexible, modern log display component with smooth scrolling and rich formatting.
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, List, Tuple
from IPython.display import display, HTML
from datetime import datetime
from enum import Enum

class LogLevel(Enum):
    DEBUG = 'debug'
    INFO = 'info'
    SUCCESS = 'success'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'

LOG_LEVEL_STYLES = {
    LogLevel.DEBUG: {'color': '#6c757d', 'bg': '#f8f9fa', 'icon': '🔍'},
    LogLevel.INFO: {'color': '#0d6efd', 'bg': '#e7f1ff', 'icon': 'ℹ️'},
    LogLevel.SUCCESS: {'color': '#198754', 'bg': '#e7f8f0', 'icon': '✅'},
    LogLevel.WARNING: {'color': '#ffc107', 'bg': '#fff8e6', 'icon': '⚠️'},
    LogLevel.ERROR: {'color': '#dc3545', 'bg': '#fdf0f2', 'icon': '❌'},
    LogLevel.CRITICAL: {'color': '#ffffff', 'bg': '#dc3545', 'icon': '🔥'}
}

def create_log_accordion(
    module_name: str = 'Process',
    height: str = '300px',
    width: str = '100%',
    max_logs: int = 1000,
    show_timestamps: bool = True,
    show_level_icons: bool = True,
    auto_scroll: bool = True
) -> Dict[str, widgets.Widget]:
    """
    Create a modern log accordion with rich formatting and smooth scrolling.
    
    Args:
        module_name: Name to display in the accordion header
        height: Height of the log container
        width: Width of the component
        max_logs: Maximum number of log entries to keep in memory
        show_timestamps: Whether to show timestamps
        show_level_icons: Whether to show level icons
        auto_scroll: Whether to automatically scroll to bottom on new messages
        
    Returns:
        Dictionary containing 'log_output' and 'log_accordion' widgets
    """
    # Create a container for log messages
    log_container = widgets.Box(
        layout=widgets.Layout(
            width='100%',
            height=height,
            overflow_y='auto',
            padding='10px',
            border='1px solid #e9ecef',
            border_radius='8px',
            margin='5px 0',
            display='flex',
            flex_direction='column-reverse',
            gap='4px'
        )
    )
    
    # Store logs in memory
    log_entries: List[Dict[str, Any]] = []
    
    def append_log(
        message: str,
        level: LogLevel = LogLevel.INFO,
        namespace: str = None,
        module: str = None,
        timestamp: datetime = None
    ) -> None:
        """
        Append a log message to the container.
        
        Args:
            message: The log message
            level: Log level (DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
            namespace: Optional namespace for categorization
            module: Optional module name
            timestamp: Optional custom timestamp
        """
        nonlocal log_entries
        
        try:
            # Validate input
            if not message:
                return
                
            # Ensure level is a LogLevel enum
            if not isinstance(level, LogLevel):
                try:
                    level = LogLevel(level.lower() if isinstance(level, str) else 'info')
                except (ValueError, AttributeError):
                    level = LogLevel.INFO
            
            # Create log entry
            entry = {
                'id': len(log_entries) + 1,
                'timestamp': timestamp or datetime.now(),
                'level': level,
                'namespace': namespace,
                'module': module,
                'message': str(message)
            }
            
            # Add to log entries and maintain max size
            log_entries.append(entry)
            if len(log_entries) > max_logs:
                log_entries.pop(0)
            
            # Update the display
            _update_log_display()
            
        except Exception as e:
            print(f"[ERROR] Failed to append log: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _update_log_display():
        """Update the log container with current entries."""
        try:
            # Update the children directly without using context manager
            log_container.children = tuple(_create_log_entry(entry) for entry in log_entries)
        except Exception as e:
            print(f"[ERROR] Failed to update log display: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _create_log_entry(entry: Dict[str, Any]) -> widgets.HTML:
        """Create a styled log entry widget."""
        style = LOG_LEVEL_STYLES.get(entry['level'], LOG_LEVEL_STYLES[LogLevel.INFO])
        
        # Format timestamp if needed
        timestamp = entry['timestamp'].strftime('%H:%M:%S.%f')[:-3] if show_timestamps else ''
        timestamp_html = f"<span style='font-size: 11px; opacity: 0.7;'>{timestamp}</span>" if timestamp else ''
        
        # Get level icon if needed
        level_icon = style['icon'] if show_level_icons else ''
        
        # Create namespace/module prefix if available
        ns = entry.get('namespace') or entry.get('module')
        ns_display = f"<span style='color: #6f42c1; font-weight: 500;'>[{ns.split('.')[-1]}]</span> " if ns else ""
        
        # Build the HTML string with direct variable interpolation
        bg_color = style['bg']
        text_color = style['color']
        
        html = f"""
        <div style='
            padding: 6px 12px;
            margin: 2px 0;
            border-radius: 6px;
            background: {bg_color};
            color: {text_color};
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 13px;
            line-height: 1.4;
            transition: all 0.2s ease;'>
            <div style='display: flex; align-items: flex-start; gap: 8px;'>
                {level_icon}
                <span style='flex: 1;'>{ns_display}{entry['message']}</span>
                {timestamp_html}
            </div>
        </div>
        """
        
        return widgets.HTML(html)
    
    # Add methods to the container for external use
    log_container.append_log = append_log
    log_container.clear_logs = lambda: log_entries.clear() or _update_log_display()
    
    # Create the accordion
    accordion = widgets.Accordion(
        children=[log_container],
        layout=widgets.Layout(width=width, margin='10px 0')
    )
    accordion.set_title(0, f"📋 {module_name} Logs")
    
    return {
        'log_output': log_container,
        'log_accordion': accordion
    }

def update_log(
    ui_components: Dict[str, Any],
    message: str,
    level: LogLevel = LogLevel.INFO,
    namespace: str = None,
    module: str = None,
    expand: bool = False,
    clear: bool = False
) -> None:
    """
    Update log with a single method call.
    
    Args:
        ui_components: Dictionary containing 'log_output' and 'log_accordion'
        message: The log message
        level: Log level (default: INFO)
        namespace: Optional namespace
        module: Optional module name
        expand: Whether to expand the accordion
        clear: Whether to clear previous logs
    """
    if 'log_output' not in ui_components:
        return
    
    log_output = ui_components['log_output']
    
    if clear and hasattr(log_output, 'clear_logs'):
        log_output.clear_logs()
    
    if hasattr(log_output, 'append_log'):
        log_output.append_log(
            message=message,
            level=level,
            namespace=namespace,
            module=module
        )
    
    if expand and 'log_accordion' in ui_components:
        ui_components['log_accordion'].selected_index = 0