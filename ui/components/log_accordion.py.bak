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
    # Create a container for log messages with a custom class for JavaScript targeting
    log_container = widgets.VBox(
        layout=widgets.Layout(
            width='100%',
            height=height,
            overflow_y='auto',
            padding='10px',
            border='1px solid #e9ecef',
            border_radius='8px',
            margin='5px 0',
            display='flex',
            flex_flow='column-reverse',
            align_items='stretch',
            overflow='hidden'
        )
    )
    # Add a custom class for JavaScript targeting
    log_container.add_class('smartcash-log-container')
    
    # Create a container to hold all log entries
    entries_container = widgets.VBox(
        layout=widgets.Layout(
            width='100%',
            display='flex',
            flex_flow='column',
            align_items='stretch',
            gap='4px',
            margin='0',
            padding='0'
        )
    )
    
    # Add the entries container to the main container
    log_container.children = [entries_container]
    
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
    
    def _scroll_to_bottom():
        """Scroll the log container to the bottom using JavaScript."""
        from IPython.display import display, Javascript
        
        # Generate a unique ID for our log container if it doesn't have one
        if not hasattr(log_container, 'log_id'):
            import uuid
            log_container.log_id = f'log-container-{uuid.uuid4().hex}'
            log_container.add_class(log_container.log_id)
        
        # Add a data attribute to track if we should auto-scroll
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
            if (isNearBottom || {force_scroll}) {{
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
        """.format(log_container=log_container.log_id, force_scroll='true' if auto_scroll else 'false')
        
        display(Javascript(js_code))
    
    def _update_log_display():
        """Update the log container with current entries."""
        try:
            # Only update if there are new entries
            current_count = len(entries_container.children)
            new_entries = log_entries[current_count:]
            
            if not new_entries:
                return
                
            # Store the current scroll position and container reference
            scroll_container = log_container
            
            # Add only new entries to the container
            new_widgets = [_create_log_entry(entry) for entry in new_entries]
            entries_container.children = list(entries_container.children) + new_widgets
            
            # Auto-scroll to bottom if enabled and user hasn't scrolled up
            if auto_scroll:
                _scroll_to_bottom()
                
        except Exception as e:
            print(f"[ERROR] Failed to update log display: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _create_log_entry(entry: Dict[str, Any]) -> widgets.HTML:
        """Create a styled log entry widget with local timezone support."""
        style = LOG_LEVEL_STYLES.get(entry['level'], LOG_LEVEL_STYLES[LogLevel.INFO])
        
        # Format timestamp in local timezone if needed
        timestamp_html = ''
        if show_timestamps and 'timestamp' in entry and entry['timestamp']:
            try:
                # Convert to local timezone
                import pytz
                from datetime import datetime
                
                # Get the timestamp (assume UTC if timezone-naive)
                ts = entry['timestamp']
                if ts.tzinfo is None:
                    ts = pytz.utc.localize(ts)
                
                # Convert to local timezone
                local_tz = pytz.timezone('Asia/Jakarta')  # Default to WIB
                local_ts = ts.astimezone(local_tz)
                
                # Format with milliseconds and timezone
                timestamp = local_ts.strftime('%H:%M:%S.%f')[:-3]  # HH:MM:SS.mmm
                timezone_str = local_ts.strftime('%Z')  # Timezone abbreviation
                
                timestamp_html = f"<span style='font-size: 11px; opacity: 0.7;'>{timestamp} {timezone_str}</span>"
                
            except Exception as e:
                # Fallback to simple timestamp if any error occurs
                timestamp = entry['timestamp'].strftime('%H:%M:%S.%f')[:-3]
                timestamp_html = f"<span style='font-size: 11px; opacity: 0.7;'>{timestamp}</span>"
                print(f"[WARNING] Failed to format local time: {str(e)}")
        
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
    
    # Create the accordion
    accordion = widgets.Accordion(children=[log_container])
    accordion.set_title(0, f"{module_name} Logs")
    accordion.selected_index = None  # Start collapsed
    
    def clear_logs():
        """Clear all log entries."""
        nonlocal log_entries
        log_entries = []
        entries_container.children = ()
    
    # Expose the append_log and clear_logs methods
    log_container.append_log = append_log
    log_container.clear_logs = clear_logs
    
    # Add clear_logs to the entries container as well for backward compatibility
    entries_container.clear_logs = clear_logs
    
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