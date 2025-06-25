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
    log_container = widgets.Box(
        layout=widgets.Layout(
            width=width,
            height=height,
            border='1px solid #e0e0e0',
            border_radius='8px',
            overflow='hidden',
            display='flex',
            flex_flow='column',
            align_items='stretch'
        )
    )
    
    # Add custom class for JavaScript targeting
    log_container.add_class('smartcash-log-container')
    log_container.log_id = f'log-container-{uuid.uuid4().hex}'
    
    # Create entries container
    entries_container = widgets.VBox(
        layout=widgets.Layout(
            width='100%',
            display='flex',
            flex_flow='column',
            align_items='stretch',
            gap='4px',
            margin='0',
            padding='8px',
            overflow_y='auto',
            overflow_x='hidden'
        )
    )
    
    # Add entries container to log container
    log_container.children = [entries_container]
    
    # Add CSS for the log container
    display(HTML(f"""
    <style>
        .{log_container.log_id} {{
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
            width: 100% !important;
            max-width: 100% !important;
            overflow: hidden !important;
        }}
        
        .{log_container.log_id} > .p-Widget.panel-widgets-box {{
            overflow-y: auto !important;
            overflow-x: hidden !important;
            max-height: 100% !important;
            width: 100% !important;
            padding: 8px !important;
            margin: 0 !important;
        }}
        
        /* Ensure log entries are visible and properly formatted */
        .{log_container.log_id} .p-Widget.panel-widgets-box > div {{
            margin-bottom: 4px;
            width: 100%;
            box-sizing: border-box;
        }}
        
        /* Style for log entries */
        .log-entry {{
            width: 100% !important;
            max-width: 100% !important;
            overflow: visible !important;
            word-wrap: break-word !important;
            white-space: pre-wrap !important;
        }}
        
        /* Scrollbar styling */
        .{log_container.log_id}::-webkit-scrollbar {{
            width: 6px !important;
            height: 6px !important;
        }}
        
        .{log_container.log_id}::-webkit-scrollbar-thumb {{
            background-color: rgba(0, 0, 0, 0.2);
            border-radius: 3px;
        }}
        
        .{log_container.log_id}::-webkit-scrollbar-track {{
            background: transparent;
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
        """Scroll the log container to the bottom using JavaScript."""
        from IPython.display import display, Javascript
        
        js_code = f"""
        (function() {{
            // Try to find our specific log container first
            let logContainer = document.querySelector('.{log_container.log_id}');
            
            // If not found, try to find any scrollable container
            if (!logContainer) {{
                const selectors = [
                    '.smartcash-log-container',
                    '.jp-OutputArea-output',
                    '.output_scroll',
                    '.output_subarea',
                    '.output'
                ];
                
                for (const selector of selectors) {{
                    const elements = document.querySelectorAll(selector);
                    for (const el of elements) {{
                        if (el.scrollHeight > el.clientHeight) {{
                            logContainer = el;
                            break;
                        }}
                    }}
                    if (logContainer) break;
                }}
            }}
            
            if (!logContainer) return;
            
            // Check if we're already at the bottom (or close)
            const isNearBottom = logContainer.scrollHeight - logContainer.clientHeight - logContainer.scrollTop < 50;
            
            // Only auto-scroll if we're near the bottom or if forced
            if (isNearBottom || {auto_scroll}) {{
                // Smooth scroll to bottom
                function scrollToBottom() {{
                    if (!logContainer) return;
                    
                    const start = logContainer.scrollTop;
                    const end = logContainer.scrollHeight - logContainer.clientHeight;
                    const duration = 150; // ms
                    
                    // Skip animation if we're already at the bottom
                    if (Math.abs(start - end) < 1) return;
                    
                    const startTime = performance.now();
                    
                    function step(currentTime) {{
                        const elapsed = currentTime - startTime;
                        const progress = Math.min(elapsed / duration, 1);
                        
                        // Ease in-out function
                        const easeInOut = t => t < 0.5 
                            ? 2 * t * t 
                            : -1 + (4 - 2 * t) * t;
                        
                        logContainer.scrollTop = start + (end - start) * easeInOut(progress);
                        
                        if (progress < 1) {{
                            window.requestAnimationFrame(step);
                        }}
                    }}
                    
                    window.requestAnimationFrame(step);
                }}
                
                // Small delay to ensure the new content is rendered
                setTimeout(scrollToBottom, 10);
            }}
        }})();
        """.format(log_container=log_container.log_id, auto_scroll='true' if auto_scroll else 'false')
        
        display(Javascript(js_code))
    
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
        """Create a compact log entry widget with proper styling."""
        # Format timestamp
        timestamp_html = ''
        if show_timestamps and 'timestamp' in entry and entry['timestamp']:
            try:
                ts = entry['timestamp']
                if not isinstance(ts, datetime):
                    if isinstance(ts, (int, float)):
                        ts = datetime.fromtimestamp(ts)
                    else:
                        ts = datetime.fromisoformat(str(ts))
                
                # Ensure timezone-aware
                if ts.tzinfo is None:
                    ts = pytz.utc.localize(ts)
                
                # Convert to local timezone
                local_ts = ts.astimezone()
                timestamp = local_ts.strftime('%H:%M:%S %Z')
                timestamp_html = f"<span style='color:#6c757d;font-size:10px;font-family:monospace;margin-left:4px;white-space:nowrap;'>{timestamp}</span>"
                
            except Exception as e:
                # Fallback to current time if timestamp is invalid
                timestamp = datetime.now().astimezone().strftime('%H:%M:%S %Z')
                timestamp_html = f"<span style='color:#6c757d;font-size:10px;font-family:monospace;margin-left:4px;white-space:nowrap;'>{timestamp}</span>"
                print(f"[WARN] Failed to parse timestamp {entry.get('timestamp')}: {str(e)}")
        
        # Create namespace badge if available
        ns_badge = ''
        try:
            ns = entry.get('namespace') or entry.get('module')
            if ns:
                # Try to get namespace from KNOWN_NAMESPACES first
                from smartcash.ui.utils.ui_logger_namespace import KNOWN_NAMESPACES
                ns_display = KNOWN_NAMESPACES.get(ns, ns.split('.')[-1])
                ns_badge = (
                    f'<span style="display:inline-block;padding:1px 4px;margin:1px 4px 0 0;align-self:flex-start;'
                    f'background-color:#f1f3f5;color:#5f3dc4;border-radius:2px;'
                    f'font-size:10px;font-weight:500;line-height:1.2;white-space:nowrap;">'
                    f'{ns_display}</span>'
                )
        except (ImportError, AttributeError):
            pass
        
        # Get style for the log level
        style = LOG_LEVEL_STYLES.get(entry['level'], LOG_LEVEL_STYLES[LogLevel.INFO])
        
        # Add border for duplicate messages
        border_style = '2px solid #e9ecef' if entry.get('show_duplicate_indicator', False) else 'none'
        
        # Build the HTML for the log entry with row layout
        html_parts = [
            f'<div style="margin:0 0 1px 0;padding:2px 8px 2px 6px;border-radius:2px;'
            f'background-color:{style["bg"]};border-left:2px solid {style["color"]};'
            f'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;'
            f'font-size:12px;line-height:1.5;word-break:break-word;white-space:pre-wrap;'
            f'overflow-wrap:break-word;display:flex;flex-direction:row;align-items:flex-start;gap:6px;'
            f'border-right:{border_style};border-left:{border_style};">',
            # Icon container
            f'<div style="display:flex;align-items:center;flex-shrink:0;font-size:12px;line-height:1.5;">',
            f'{style["icon"]}',
            '</div>',
            # Message and namespace in a row
            f'<div style="flex:1;display:flex;flex-direction:row;align-items:flex-start;gap:4px;line-height:1.5;min-width:0;">',
            f'<div style="color:{style["color"]};flex:1;display:flex;flex-direction:row;align-items:flex-start;gap:4px;">',
            f'{ns_badge if ns_badge else ""}',
            f'<span style="flex:1;">{entry["message"]}</span>',
            '</div>',  # End of message content
            f'<div style="color:#6c757d;font-size:10px;font-family:monospace;white-space:nowrap;margin-left:4px;line-height:1.5;">',
            f'{timestamp}',
            '</div>',  # End of timestamp
            '</div>',  # End of message row
            '</div>'  # End of log entry
        ]
        
        html = ''.join(html_parts)
        
        return widgets.HTML(html)
    
    duplicate_count = 0
    
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