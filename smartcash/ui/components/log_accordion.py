"""
Modern Log Accordion Component with Deduplication

A flexible, modern log display component with smooth scrolling, rich formatting,
and smart message deduplication.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, HTML, Javascript
import time
import uuid
import pytz
from threading import Timer

class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = 'debug'
    INFO = 'info'
    SUCCESS = 'success'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'

# Define log level styles
LOG_LEVEL_STYLES = {
    LogLevel.DEBUG: {'color': '#6c757d', 'bg': '#f8f9fa', 'icon': '🔍'},
    LogLevel.INFO: {'color': '#0d6efd', 'bg': '#e7f1ff', 'icon': 'ℹ️'},
    LogLevel.SUCCESS: {'color': '#198754', 'bg': '#e7f8f0', 'icon': '✅'},
    LogLevel.WARNING: {'color': '#ffc107', 'bg': '#fff8e6', 'icon': '⚠️'},
    LogLevel.ERROR: {'color': '#dc3545', 'bg': '#fdf0f2', 'icon': '❌'},
    LogLevel.CRITICAL: {'color': '#dc3545', 'bg': '#fdf0f2', 'icon': '🔥'}
}

# Constants for deduplication
DUPLICATE_WINDOW_MS = 1000  # 1 second window for considering messages as duplicates
MAX_DUPLICATE_COUNT = 2   # Maximum number of duplicates to show

def create_log_accordion(
    module_name: str = 'Process',
    height: str = '300px',
    width: str = '100%',
    max_logs: int = 1000,
    show_timestamps: bool = True,
    show_level_icons: bool = True,
    auto_scroll: bool = True,
    enable_deduplication: bool = True
) -> Dict[str, Any]:
    """
    Create a modern log accordion with rich formatting and deduplication.
    
    Args:
        module_name: Name to display in the accordion header
        height: Height of the log container
        width: Width of the component
        max_logs: Maximum number of log entries to keep in memory
        show_timestamps: Whether to show timestamps
        show_level_icons: Whether to show level icons
        auto_scroll: Whether to automatically scroll to bottom on new messages
        enable_deduplication: Whether to enable message deduplication
        
    Returns:
        Dictionary containing 'log_output' and 'log_accordion' widgets
    """
    # Create main container
    log_container = widgets.Output(layout={
        'border': '1px solid #e0e0e0',
        'border_radius': '8px',
        'overflow': 'hidden',
        'width': width,
        'height': height,
        'margin': '5px 0',
        'display': 'flex',
        'flex_direction': 'column'
    })
    
    # Add custom CSS
    log_container.add_class('smartcash-log-container')
    log_container.log_id = f'log-container-{uuid.uuid4().hex}'
    
    # Create entries container
    entries_container = widgets.VBox(layout={
        'overflow_y': 'auto',
        'height': '100%',
        'padding': '8px',
        'flex': '1',
        'min_height': '0'  # Important for flexbox scrolling
    })
    
    # Add entries container to log container
    with log_container:
        display(entries_container)
    
    # Add custom CSS for the log container
    display(HTML(f"""
    <style>
        .{log_container.log_id} {{
            display: flex;
            flex-direction: column;
            height: 100%;
        }}
        
        .{log_container.log_id} .jp-OutputArea-output {{
            height: 100%;
            display: flex;
            flex-direction: column;
        }}
        
        .{log_container.log_id} .lm-Widget {{
            overflow-y: auto !important;
        }}
        
        @keyframes pulse {{
            0% {{ transform: scale(0.95); opacity: 0.7; }}
            50% {{ transform: scale(1.1); opacity: 1; }}
            100% {{ transform: scale(0.95); opacity: 0.7; }}
        }}
        
        .duplicate-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            background-color: #ffc107;
            border-radius: 50%;
            margin-right: 6px;
            vertical-align: middle;
            animation: pulse 1.5s infinite;
        }}
        
        .duplicate-count {{
            background-color: #ffc107;
            border-radius: 8px;
            padding: 0 6px;
            font-size: 0.85em;
            transition: background 0.3s ease;
            margin-left: 4px;
            color: #000;
            font-weight: bold;
        }}
    </style>
    """))
    
    # Store log entries
    log_entries: List[Dict[str, Any]] = []
    last_message = None
    duplicate_count = 0
    last_timestamp = None
    
    def append_log(
        message: str,
        level: LogLevel = LogLevel.INFO,
        namespace: str = None,
        module: str = None,
        timestamp: datetime = None
    ) -> None:
        """Append a log message with deduplication support."""
        nonlocal log_entries, last_message, duplicate_count, last_timestamp
        
        try:
            if not message:
                return
                
            # Ensure level is a LogLevel enum
            if not isinstance(level, LogLevel):
                try:
                    level = LogLevel(level.lower() if isinstance(level, str) else 'info')
                except (ValueError, AttributeError):
                    level = LogLevel.INFO
            
            current_timestamp = timestamp or datetime.now()
            
            # Check for duplicates if enabled
            if enable_deduplication:
                is_duplicate = (
                    last_message is not None and
                    last_message.get('message') == str(message) and
                    last_message.get('level') == level and
                    last_message.get('namespace') == namespace and
                    last_message.get('module') == module and
                    (current_timestamp - last_message['timestamp']).total_seconds() * 1000 < DUPLICATE_WINDOW_MS
                )
                
                if is_duplicate:
                    duplicate_count = min(duplicate_count + 1, MAX_DUPLICATE_COUNT)
                    last_message['count'] = duplicate_count
                    last_message['last_timestamp'] = current_timestamp
                    last_message['show_duplicate_indicator'] = True
                    
                    # Update the display
                    _update_duplicate_count(
                        entry_id=last_message['id'],
                        count=duplicate_count,
                        show_indicator=True
                    )
                    
                    # Schedule indicator removal
                    def remove_indicator():
                        if last_message and last_message['id'] == entry_id:
                            last_message['show_duplicate_indicator'] = False
                            _update_log_display()
                    
                    entry_id = last_message['id']
                    Timer(2.0, remove_indicator).start()
                    return
            
            # Reset duplicate tracking for new message
            if duplicate_count > 0:
                _update_duplicate_count(
                    entry_id=last_message['id'],
                    count=duplicate_count,
                    show_indicator=False
                )
            
            duplicate_count = 0
            
            # Create new log entry
            entry = {
                'id': len(log_entries) + 1,
                'timestamp': current_timestamp,
                'level': level,
                'namespace': namespace,
                'module': module,
                'message': str(message),
                'count': 0,
                'show_duplicate_indicator': False
            }
            
            # Update last message tracking
            last_message = entry
            last_timestamp = current_timestamp
            
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
    
    def _update_duplicate_count(entry_id: int, count: int, show_indicator: bool = False) -> None:
        """Update the duplicate count for an existing log entry."""
        nonlocal log_entries
        
        for entry in reversed(log_entries):
            if entry['id'] == entry_id:
                entry['count'] = count
                if show_indicator:
                    entry['show_duplicate_indicator'] = True
                break
        
        _update_log_display()
    
    def _scroll_to_bottom():
        """Scroll the log container to the bottom."""
        display(Javascript(f"""
        (function() {{
            const container = document.querySelector('.{log_container.log_id}');
            if (container) {{
                container.scrollTop = container.scrollHeight;
            }}
        }})();
        """))
    
    def _update_log_display():
        """Update the log container with current entries."""
        try:
            current_count = len(entries_container.children)
            new_entries = log_entries[current_count:]
            
            if not new_entries:
                return
                
            # Create widgets for new entries
            new_widgets = [_create_log_entry(entry) for entry in new_entries]
            entries_container.children = list(entries_container.children) + new_widgets
            
            # Auto-scroll if enabled
            if auto_scroll:
                _scroll_to_bottom()
                
        except Exception as e:
            print(f"[ERROR] Failed to update log display: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _create_log_entry(entry: Dict[str, Any]) -> widgets.HTML:
        """Create a styled log entry widget."""
        style = LOG_LEVEL_STYLES.get(entry['level'], LOG_LEVEL_STYLES[LogLevel.INFO])
        
        # Format timestamp
        timestamp_html = ''
        if show_timestamps and 'timestamp' in entry and entry['timestamp']:
            try:
                ts = entry['timestamp']
                if ts.tzinfo is None:
                    ts = pytz.utc.localize(ts)
                
                local_tz = pytz.timezone('Asia/Jakarta')
                local_ts = ts.astimezone(local_tz)
                timestamp = local_ts.strftime('%H:%M:%S.%f')[:-3]
                timezone_str = local_ts.strftime('%Z')
                
                # Add duplicate count if present
                count_html = ""
                if entry.get('count', 0) > 0:
                    count_bg = "#ffc107" if entry.get('show_duplicate_indicator', False) else "rgba(0,0,0,0.1)"
                    count_html = f" <span class='duplicate-count' style='background: {count_bg};'>{entry['count']+1}x</span>"
                
                timestamp_html = f"<span style='font-size: 11px; opacity: 0.7;'>{timestamp} {timezone_str}{count_html}</span>"
                
            except Exception as e:
                timestamp = entry['timestamp'].strftime('%H:%M:%S.%f')[:-3]
                count_html = f" <span class='duplicate-count'>{entry['count']+1}x</span>" if entry.get('count', 0) > 0 else ""
                timestamp_html = f"<span style='font-size: 11px; opacity: 0.7;'>{timestamp}{count_html}</span>"
        
        # Create namespace/module prefix
        ns = entry.get('namespace') or entry.get('module')
        ns_display = f"<span style='color: #6f42c1; font-weight: 500;'>[{ns.split('.')[-1]}]</span> " if ns else ""
        
        # Add duplicate indicator if needed
        indicator_html = "<span class='duplicate-indicator'></span>" if entry.get('show_duplicate_indicator', False) else ""
        
        # Create the HTML
        html = f"""
        <div style='
            padding: 6px 12px;
            margin: 2px 0;
            border-radius: 6px;
            background: {style['bg']};
            color: {style['color']};
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 13px;
            line-height: 1.4;
            transition: all 0.2s ease;'>
            <div style='display: flex; align-items: flex-start; gap: 8px;'>
                {indicator_html}{style['icon'] if show_level_icons else ''}
                <span style='flex: 1;'>{ns_display}{entry['message']}</span>
                {timestamp_html}
            </div>
        </div>
        """
        
        return widgets.HTML(html)
    
    def clear_logs():
        """Clear all log entries."""
        nonlocal log_entries, last_message, duplicate_count
        log_entries = []
        last_message = None
        duplicate_count = 0
        entries_container.children = ()
    
    # Expose methods
    log_container.append_log = append_log
    log_container.clear_logs = clear_logs
    entries_container.clear_logs = clear_logs
    
    # Create the accordion
    accordion = widgets.Accordion(children=[log_container])
    accordion.set_title(0, f"{module_name} Logs")
    accordion.selected_index = None  # Start collapsed
    
    return {
        'log_output': log_container,
        'log_accordion': accordion,
        'entries_container': entries_container
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