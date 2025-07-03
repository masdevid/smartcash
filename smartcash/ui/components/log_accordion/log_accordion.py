"""
Main LogAccordion implementation.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, HTML, Javascript
import uuid
import pytz
from threading import Timer

from smartcash.ui.components.base_component import BaseUIComponent
from .log_level import LogLevel, get_log_level_style
from .log_entry import LogEntry


class LogAccordion(BaseUIComponent):
    """A modern log accordion component with rich formatting and deduplication."""
    
    # Constants for deduplication
    DEFAULT_DUPLICATE_WINDOW_MS = 1000  # 1 second window for considering messages as duplicates
    DEFAULT_MAX_DUPLICATE_COUNT = 2     # Maximum number of duplicates to show
    
    def __init__(
        self,
        component_name: str = "log_accordion",
        module_name: str = 'Process',
        height: str = '300px',
        width: str = '100%',
        max_logs: int = 1000,
        show_timestamps: bool = True,
        show_level_icons: bool = True,
        auto_scroll: bool = True,
        enable_deduplication: bool = True,
        duplicate_window_ms: Optional[int] = None,
        max_duplicate_count: Optional[int] = None
    ) -> None:
        """Initialize the LogAccordion.
        
        Args:
            component_name: Unique name for this component instance
            module_name: Name to display in the accordion header
            height: Height of the log container
            width: Width of the component
            max_logs: Maximum number of log entries to keep in memory
            show_timestamps: Whether to show timestamps
            show_level_icons: Whether to show level icons
            auto_scroll: Whether to automatically scroll to bottom on new messages
            enable_deduplication: Whether to enable message deduplication
            duplicate_window_ms: Time window in ms to consider messages as duplicates
            max_duplicate_count: Maximum number of duplicates to show
        """
        super().__init__(component_name)
        self.module_name = module_name
        self.height = height
        self.width = width
        self.max_logs = max_logs
        self.show_timestamps = show_timestamps
        self.show_level_icons = show_level_icons
        self.auto_scroll = auto_scroll
        self.enable_deduplication = enable_deduplication
        self.duplicate_window_ms = duplicate_window_ms or self.DEFAULT_DUPLICATE_WINDOW_MS
        self.max_duplicate_count = max_duplicate_count or self.DEFAULT_MAX_DUPLICATE_COUNT
        
        # Initialize state
        self.log_entries: List[LogEntry] = []
        self.last_entry: Optional[LogEntry] = None
        self.duplicate_count: int = 0
        self.log_id = f'log-container-{uuid.uuid4().hex}'
        
        # Initialize UI components
        self._ui_components: Dict[str, Any] = {}
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the component and create UI elements."""
        if self._initialized:
            return
            
        self._create_ui_components()
        self._initialized = True
    
    def _create_ui_components(self) -> None:
        """Create and initialize UI components."""
        # Create main container
        self._ui_components['log_container'] = widgets.Box(
            layout=widgets.Layout(
                width=self.width,
                height=self.height,
                border='1px solid #e0e0e0',
                border_radius='8px',
                overflow='hidden',
                display='flex',
                flex_flow='column',
                align_items='stretch'
            )
        )
        
        # Add custom class for JavaScript targeting
        self._ui_components['log_container'].add_class('smartcash-log-container')
        
        # Create entries container
        self._ui_components['entries_container'] = widgets.VBox(
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
        self._ui_components['log_container'].children = [self._ui_components['entries_container']]
        
        # Create the accordion
        self._ui_components['accordion'] = widgets.Accordion(
            children=[self._ui_components['log_container']]
        )
        self._ui_components['accordion'].set_title(0, f"{self.module_name.upper()} LOGS")
        self._ui_components['accordion'].selected_index = None  # Start collapsed
        
        # Add CSS for the log container
        self._add_css_styles()
    
    def _add_css_styles(self) -> None:
        """Add CSS styles for the log container."""
        css = f"""
        <style>
            .{self.log_id} {{
                display: flex !important;
                flex-direction: column !important;
                height: 100% !important;
                width: 100% !important;
                max-width: 100% !important;
                overflow: hidden !important;
            }}
            
            .{self.log_id} > .p-Widget.panel-widgets-box {{
                overflow-y: auto !important;
                overflow-x: hidden !important;
                max-height: 100% !important;
                width: 100% !important;
                padding: 8px !important;
                margin: 0 !important;
            }}
            
            /* Ensure log entries are visible and properly formatted */
            .{self.log_id} .p-Widget.panel-widgets-box > div {{
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
            .{self.log_id}::-webkit-scrollbar {{
                width: 6px !important;
                height: 6px !important;
            }}
            
            .{self.log_id}::-webkit-scrollbar-thumb {{
                background-color: rgba(0, 0, 0, 0.2);
                border-radius: 3px;
            }}
            
            .{self.log_id}::-webkit-scrollbar-track {{
                background: transparent;
            }}
        </style>
        
        <script>
            function scrollToBottom(logId) {{
                const element = document.querySelector(`.${{logId}}`);
                if (element) {{
                    element.scrollTop = element.scrollHeight;
                }}
            }}
        </script>
        """
        display(HTML(css))
    
    def log(
        self,
        message: str,
        level: Union[LogLevel, str] = LogLevel.INFO,
        namespace: Optional[str] = None,
        module: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Append a log message with deduplication support.
        
        Args:
            message: The log message
            level: Log level (default: INFO)
            namespace: Optional namespace for the log
            module: Optional module name
            timestamp: Optional timestamp (default: current time)
        """
        if not self._initialized:
            self.initialize()
            
        if not message:
            return
            
        # Ensure level is a LogLevel enum
        if not isinstance(level, LogLevel):
            try:
                level = LogLevel(level.lower() if isinstance(level, str) else 'info')
            except (ValueError, AttributeError):
                level = LogLevel.INFO
        
        # Create new log entry
        entry = LogEntry(
            message=str(message),
            level=level,
            namespace=namespace,
            module=module,
            timestamp=timestamp or datetime.now()
        )
        
        # Handle deduplication if enabled
        if self.enable_deduplication and self.last_entry and entry.is_duplicate_of(self.last_entry, self.duplicate_window_ms):
            self.last_entry.increment_duplicate_count(self.max_duplicate_count)
            self._update_log_display()
            return
        
        # Add new entry
        self.log_entries.append(entry)
        self.last_entry = entry
        
        # Trim old entries if needed
        if len(self.log_entries) > self.max_logs:
            self.log_entries.pop(0)
        
        # Update the display
        self._update_log_display()
    
    def _update_log_display(self) -> None:
        """Update the log display with current entries."""
        if not self._initialized:
            return
            
        entries_container = self._ui_components['entries_container']
        current_count = len(entries_container.children)
        new_entries = self.log_entries[current_count:]
        
        if not new_entries:
            return
            
        # Create widgets for new entries
        new_widgets = [self._create_log_widget(entry) for entry in new_entries]
        entries_container.children = list(entries_container.children) + new_widgets
        
        # Auto-scroll if enabled
        if self.auto_scroll:
            self._scroll_to_bottom()
    
    def _create_log_widget(self, entry: LogEntry) -> widgets.HTML:
        """Create an HTML widget for a log entry."""
        # Get style for the log level
        style = get_log_level_style(entry.level)
        
        # Format timestamp
        timestamp_html = ''
        if self.show_timestamps and entry.timestamp:
            try:
                ts = entry.timestamp
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
                timestamp_str = local_ts.strftime('%H:%M:%S %Z')
                timestamp_html = f"<span style='color:#6c757d;font-size:10px;font-family:monospace;margin-left:4px;white-space:nowrap;'>{timestamp_str}</span>"
                
            except Exception as e:
                # Fallback to current time if timestamp is invalid
                timestamp_str = datetime.now().astimezone().strftime('%H:%M:%S %Z')
                timestamp_html = f"<span style='color:#6c757d;font-size:10px;font-family:monospace;margin-left:4px;white-space:nowrap;'>{timestamp_str}</span>"
        
        # Create namespace badge if available
        ns_badge = ''
        try:
            ns = entry.namespace or entry.module
            if ns:
                # Try to get namespace from KNOWN_NAMESPACES first
                from smartcash.ui.utils.ui_logger_namespace import KNOWN_NAMESPACES
                ns_display = KNOWN_NAMESPACES.get(ns, ns.split('.')[-1])
                ns_badge = (
                    f'<span style="display:inline-block;padding:1px 4px;margin:1px 4px 0 0;align-self:flex-start;'
                    f'background-color:#f1f3f5;color:#5f3dc4;border-radius:2px;'
                    f'font-size:10px;font-weight:500;line-height:1.2;white-space:nowrap;'
                    f'"'  # Close the style attribute
                    f'>{ns_display}</span>'
                )
        except (ImportError, AttributeError):
            pass
        
        # Add duplicate count if applicable
        count_badge = ''
        if entry.count > 1:
            count_badge = f'<span style="margin-left:4px;color:#868e96;font-size:0.8em;">(x{entry.count})</span>'
        
        # Add border for duplicate messages
        border_style = '2px solid #e9ecef' if entry.show_duplicate_indicator else 'none'
        
        # Build the HTML for the log entry with row layout
        level_icon = f'<span style="margin-right:4px;">{style["icon"]}</span>' if self.show_level_icons else ''
        bg_color = style['bg']
        color = style['color']
        message = entry.message
        
        # Build the HTML string with all variables defined
        html = f"""
        <div class="log-entry" style="
            margin:0 0 1px 0;
            padding:4px 8px;
            border-radius:2px;
            display:flex;
            background-color:{bg_color};
            border-left:2px solid {color};
            font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
            font-size:12px;
            line-height:1.5;
            word-break:break-word;
            white-space:pre-wrap;
            overflow-wrap:break-word;
            border-right:{border_style};
            border-left:{border_style};
        ">
            <div style="display:flex;flex-direction:column;width:100%;">
                <div style="display:flex;align-items:flex-start;">
                    {level_icon}
                    {ns_badge}
                    <div style="flex:1;min-width:0;">
                        <span style="color:{color};font-weight:500;">{message}</span>
                        {count_badge}
                    </div>
                    {timestamp_html}
                </div>
            </div>
        </div>
        """.format(
            color=color,
            bg_color=bg_color,
            level_icon=level_icon,
            ns_badge=ns_badge,
            message=message,
            count_badge=count_badge,
            timestamp=timestamp_html,
            border_style=border_style
        )
        
        return widgets.HTML(html, layout=widgets.Layout(width='100%', margin='0', padding='0'))
    
    def _scroll_to_bottom(self) -> None:
        """Scroll the log container to the bottom using JavaScript."""
        if not self.auto_scroll:
            return
            
        # Use direct f-string interpolation with self.log_id
        js_code = f"""
        (function() {{
            const logContainer = document.querySelector('.{self.log_id}');
            if (logContainer) {{
                logContainer.scrollTop = logContainer.scrollHeight;
            }}
        }})();
        """
        
        display(Javascript(js_code))
    
    def clear(self) -> None:
        """Clear all log entries."""
        self.log_entries = []
        self.last_entry = None
        self.duplicate_count = 0
        
        if self._initialized:
            self._ui_components['entries_container'].children = []
    
    def display(self) -> widgets.Accordion:
        """Display the log accordion."""
        if not self._initialized:
            self.initialize()
        accordion = self._ui_components['accordion']
        display(accordion)
        return accordion
    
    def _ipython_display_(self):
        """IPython display integration."""
        return self.display()._ipython_display_()
