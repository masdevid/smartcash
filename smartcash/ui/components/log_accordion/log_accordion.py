"""
Main LogAccordion implementation.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, HTML, Javascript
import uuid
import pytz
import time

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
        max_duplicate_count: Optional[int] = None,
        namespace_filter: Optional[Union[str, List[str]]] = None,
        log_entry_style: str = 'compact'
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
            log_entry_style: Style of log entries ('compact' or 'default')
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
        self.log_entry_style = log_entry_style.lower() if log_entry_style else 'compact'
        
        # Namespace filtering
        self.namespace_filter = namespace_filter
        
        # Initialize state
        self.log_entries: List[LogEntry] = []
        self.last_entry: Optional[LogEntry] = None
        self.duplicate_count: int = 0
        
        # Generate log ID with fallback
        try:
            self.log_id = f'log-container-{uuid.uuid4().hex}'
        except Exception:
            # Fallback to timestamp-based ID if UUID fails
            self.log_id = f'log-container-{int(time.time() * 1000)}'
        
        # Cache for filtered entries to improve performance
        self._filtered_entries: List[LogEntry] = []
        
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
        
        # Create entries container with better spacing and scrolling
        self._ui_components['entries_container'] = widgets.VBox(
            layout=widgets.Layout(
                width='100%',
                display='flex',
                flex_flow='column',
                align_items='stretch',
                gap='2px',
                margin='0',
                padding='4px 8px',
                overflow_y='auto',
                overflow_x='hidden',
                max_height=self.height,
                min_height='100px'
            )
        )
        
        # Add entries container to log container
        self._ui_components['log_container'].children = [self._ui_components['entries_container']]
        
        # Create the accordion
        self._ui_components['accordion'] = widgets.Accordion(
            children=[self._ui_components['log_container']],
            selected_index=0  # Start expanded
        )
        self._ui_components['accordion'].set_title(0, f"{self.module_name.upper()} LOGS")
        
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
                overflow-x: auto !important;
                max-height: 100% !important;
                width: 100% !important;
                padding: 4px 0 !important;
                margin: 0 !important;
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace !important;
                font-size: 12.5px !important;
                line-height: 1.4 !important;
                word-wrap: break-word !important;
                word-break: break-word !important;
                white-space: pre-wrap !important;
                overflow-wrap: anywhere !important;
            }}
            
            /* Log entry styling - original format */
            .log-entry {{
                width: 100% !important;
                max-width: 100% !important;
                overflow: visible !important;
                word-wrap: break-word !important;
                word-break: break-word !important;
                white-space: pre-wrap !important;
                padding: 4px 8px !important;
                margin: 1px 0 !important;
                border-radius: 3px !important;
                border-left: 3px solid transparent !important;
                transition: all 0.15s ease !important;
                cursor: default !important;
                overflow-wrap: anywhere !important;
                hyphens: auto !important;
            }}
            
            /* Compact log entry styling - new format */
            .log-entry-compact {{
                width: 100% !important;
                max-width: 100% !important;
                overflow: visible !important;
                word-wrap: break-word !important;
                word-break: break-word !important;
                white-space: pre-wrap !important;
                padding: 2px 8px !important;
                margin: 1px 0 !important;
                border-radius: 3px !important;
                transition: all 0.15s ease !important;
                cursor: default !important;
                overflow-wrap: anywhere !important;
                hyphens: auto !important;
            }}
            
            /* Hover effect for both log entry types */
            .log-entry:hover, .log-entry-compact:hover {{
                transform: translateX(2px) !important;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
            }}
            
            /* Log entry content */
            .log-content {{
                display: flex !important;
                align-items: flex-start !important;
                gap: 6px !important;
            }}
            
            /* Timestamp */
            .log-timestamp {{
                color: #6c757d !important;
                font-size: 11px !important;
                white-space: nowrap !important;
                opacity: 0.8 !important;
                flex-shrink: 0 !important;
                padding-top: 1px !important;
            }}
            
            /* Log level badge */
            .log-level {{
                font-weight: 600 !important;
                font-size: 11px !important;
                padding: 1px 4px !important;
                border-radius: 3px !important;
                margin-right: 2px !important;
                text-transform: uppercase !important;
                letter-spacing: 0.5px !important;
                flex-shrink: 0 !important;
                line-height: 1.3 !important;
            }}
            
            /* Namespace badge */
            .log-namespace {{
                color: #6c757d !important;
                background: rgba(108, 117, 125, 0.1) !important;
                border-radius: 3px !important;
                padding: 0 4px !important;
                font-size: 11px !important;
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace !important;
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
                max-width: 200px !important;
                display: inline-block !important;
                vertical-align: middle !important;
                line-height: 1.4 !important;
                margin-right: 4px !important;
            }}
            
            /* Log message */
            .log-message {{
                flex: 1 !important;
                min-width: 0 !important;
                word-break: break-word !important;
                white-space: pre-wrap !important;
            }}
            
            /* Error traceback */
            .error-traceback {{
                margin-top: 4px !important;
                margin-left: 16px !important;
                padding-left: 8px !important;
                border-left: 2px solid rgba(220, 53, 69, 0.2) !important;
                display: none !important;
            }}
            
            .show-traceback .error-traceback {{
                display: block !important;
            }}
            
            .toggle-traceback {{
                color: #6c757d !important;
                cursor: pointer !important;
                font-size: 11px !important;
                margin-top: 2px !important;
                display: inline-block !important;
                user-select: none !important;
            }}
            
            .toggle-traceback:hover {{
                text-decoration: underline !important;
            }}
            
            /* Scrollbar styling */
            .{self.log_id}::-webkit-scrollbar {{
                width: 6px !important;
                height: 6px !important;
            }}
            
            .{self.log_id}::-webkit-scrollbar-thumb {{
                background-color: rgba(0, 0, 0, 0.2) !important;
                border-radius: 3px !important;
            }}
            
            .{self.log_id}::-webkit-scrollbar-track {{
                background: transparent !important;
            }}
            
            /* Duplicate counter */
            .duplicate-counter {{
                background: #6c757d !important;
                color: white !important;
                border-radius: 8px !important;
                font-size: 9px !important;
                padding: 1px 4px !important;
                margin-left: 4px !important;
                display: inline-block !important;
                vertical-align: middle !important;
                line-height: 1.2 !important;
                font-weight: 600 !important;
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
    
    def _get_filtered_entries(self) -> List[LogEntry]:
        """Get log entries filtered by allowed namespaces.
        
        Always includes core namespaces regardless of the current filter:
        - smartcash.ui.core
        - smartcash.common
        - smartcash.dataset
        """
        # If no filter, return all entries
        if not self.namespace_filter:
            return self.log_entries
            
        # Core namespaces that should always be included
        core_namespaces = {
            'smartcash.ui.core',
            'smartcash.common',
            'smartcash.dataset',
            'smartcash.model'
        }
        
        # Add current cell's namespace filter
        current_filter = self.namespace_filter.lower() if isinstance(self.namespace_filter, str) else ''
        if current_filter and current_filter not in {ns.lower() for ns in core_namespaces}:
            core_namespaces.add(current_filter)
            
        # Filter entries by allowed namespaces (case-insensitive)
        return [entry for entry in self.log_entries 
                if entry.namespace and any(
                    entry.namespace.lower().startswith(ns.lower())
                    for ns in core_namespaces
                )]
    
    def _update_log_display(self) -> None:
        """Update the log display with current entries."""
        if not self._initialized:
            self.initialize()
            
        # Get filtered entries
        filtered_entries = self._get_filtered_entries()
        
        # Only update if entries changed
        if hasattr(self, '_last_filtered_count') and \
           len(filtered_entries) == self._last_filtered_count:
            return
            
        self._last_filtered_count = len(filtered_entries)
        
        # Clear existing entries
        entries = []
        
        # Add log entries
        for entry in filtered_entries:
            entries.append(self._create_log_widget(entry))
        
        # Update the display
        self._ui_components['entries_container'].children = entries
        
        # Auto-scroll if enabled
        if self.auto_scroll:
            self._scroll_to_bottom()
    
    def _shorten_namespace(self, namespace: str) -> str:
        """Shorten long namespace paths for better readability.
        
        Args:
            namespace: The full namespace to shorten
            
        Returns:
            A shortened, more readable version of the namespace
        """
        if not namespace or not isinstance(namespace, str):
            return ""
            
        # Common prefixes to shorten
        replacements = [
            ('smartcash.', ''),
            ('smartcash.common.', 'common.'),
            ('smartcash.dataset.', 'dataset.'),
            ('smartcash.model.', 'model.'),
            ('smartcash.ui.', 'ui.'),
            ('smartcash.ui.core.', 'core.'),
            ('smartcash.ui.core.shared.', 'core.'),
            ('smartcash.ui.components.', 'components.'),
            ('smartcash.ui.setup.', 'setup.'),
            ('smartcash.ui.setup.colab.', 'colab.'),
            ('smartcash.ui.setup.dependency', 'dependency.'),
            ('smartcash.ui.dataset.', 'dataset.'),
            ('smartcash.ui.dataset.downloader.', 'downloader.'),
            ('smartcash.ui.dataset.preprocessing.', 'preprocessing.'),
            ('smartcash.ui.dataset.augmentation.', 'augmentation.'),
            ('smartcash.ui.dataset.split.', 'split.'),
            ('smartcash.ui.dataset.visualization.', 'visualization.'),
            ('smartcash.ui.model.', 'model.'),
            ('smartcash.ui.model.pretrained.', 'pretrained.'),
            ('smartcash.ui.model.backbone.', 'backbone.'),
            ('smartcash.ui.model.train.', 'train.'),
            ('smartcash.ui.model.training.', 'training.'),
            ('smartcash.ui.model.evaluate.', 'evaluate.'),
            ('smartcash.ui.model.evaluation.', 'evaluation.'),
        ]
        
        # Apply replacements
        short_ns = namespace
        for old, new in replacements:
            if short_ns.startswith(old):
                short_ns = new + short_ns[len(old):]
                # Only apply the first matching replacement
                break
                
        # If no replacements were made, try to get the last part of the path
        if short_ns == namespace and '.' in short_ns:
            short_ns = short_ns.split('.')[-1]
                
        return short_ns

    def _create_log_widget(self, entry: LogEntry) -> widgets.HTML:
        """Create an HTML widget for a log entry in a single row format."""
        try:
            style = get_log_level_style(entry.level)
            
            # Format timestamp in GMT+7
            timestamp = self._format_timestamp(entry.timestamp)
            
            # Get level emoji/icon
            level_emoji = style['icon'] if self.show_level_icons else ""
            
            # Add duplicate counter if needed
            duplicate_counter = f" <span class='duplicate-counter'>{entry.count}</span>" if entry.show_duplicate_indicator else ""
            
            # Get and format namespace
            full_namespace = self._shorten_namespace(entry.namespace) if entry.namespace else ''
                
            # Handle multi-line messages (e.g., tracebacks)
            message_lines = entry.message.split('\n')
            main_message = message_lines[0]  # First line only for compact view
            has_more_lines = len(message_lines) > 1
            
            # Create expandable content for multi-line messages
            if has_more_lines:
                traceback_id = f"traceback-{id(entry)}"
                additional_content = '\n'.join(message_lines[1:]).strip()
                
                expandable_html = f"""
                <div class='toggle-traceback' onclick='document.getElementById("{traceback_id}").classList.toggle("show-traceback")'>
                    ▼ Show details
                </div>
                <div class='error-traceback' id='{traceback_id}'>
                    <pre style='margin:0; white-space:pre-wrap; font-size: 11px;'>{additional_content}</pre>
                </div>
                """
            else:
                expandable_html = ""
            
            # Determine which CSS class to use based on log_entry_style
            entry_class = 'log-entry-compact' if self.log_entry_style == 'compact' else 'log-entry'
            
            # Create single row format HTML with dynamic class and styles
            html = f"""
            <div class='{entry_class}' style='
                padding: {'2px 8px' if self.log_entry_style == 'compact' else '4px 8px'}; 
                margin: 2px 0; 
                border-left: 3px solid {style['color']}; 
                background: {style['bg']}; 
                border-radius: 3px;
                transition: all 0.15s ease;
                display: flex;
                flex-wrap: wrap;
                align-items: flex-start;
                gap: 6px;
                line-height: 1.4;
                font-size: 13px;
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;'>
                <!-- Namespace -->
                <span class='log-namespace' title='{entry.namespace or ""}' style='
                    color: #6c757d;
                    font-size: 11px;
                    display: {'inline-block' if full_namespace else 'none'};
                    max-width: {'200px' if self.log_entry_style == 'compact' else 'none'};
                    overflow: hidden;
                    text-overflow: ellipsis;
                    white-space: nowrap;
                    margin-right: 8px;
                    vertical-align: middle;
                    line-height: 1.4;'>
                    {full_namespace}
                </span>
                <!-- Icon -->
                <span style='
                    font-size: 14px;
                    flex-shrink: 0;
                    width: 20px;
                    text-align: center;'>
                    {level_emoji}
                </span>
                
                <!-- Message -->
                <span style='
                    flex: 1;
                    color: {style['text_color']};
                    word-break: break-word;
                    overflow-wrap: break-word;
                    white-space: pre-wrap;
                    line-height: 1.4;
                    display: inline-block;
                    min-width: 0;'>
                    {main_message}{duplicate_counter}
                </span>
                
                <!-- Timestamp -->
                <span style='
                    color: #6c757d;
                    font-size: 11px;
                    flex-shrink: 0;
                    margin-left: 8px;'>
                    {timestamp}
                </span>
                
                {expandable_html}
            </div>
            """
            
            return widgets.HTML(html)
            
        except Exception as e:
            # Use the inherited logger from BaseUIComponent
            try:
                self.logger.error(f"Error creating log widget: {str(e)}")
            except (TypeError, AttributeError):
                # Fallback if logger is not properly initialized
                import logging
                logging.getLogger(__name__).error(f"Error creating log widget: {str(e)}")
            return widgets.HTML(f"<div class='log-entry' style='color: #dc3545; padding: 4px 8px;'>Error displaying log: {str(e)}</div>")

    def _format_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp with error handling and timezone awareness (GMT+7)."""
        try:
            # Convert to GMT+7 (Asia/Jakarta timezone)
            gmt_plus_7 = pytz.timezone('Asia/Jakarta')
            if timestamp.tzinfo is None:
                # If naive datetime, assume it's local time first
                timestamp = timestamp.astimezone()
            # Convert to GMT+7
            timestamp_gmt7 = timestamp.astimezone(gmt_plus_7)
            return timestamp_gmt7.strftime("%H:%M:%S")
        except Exception:
            return datetime.now().strftime("%H:%M:%S")

    def _create_namespace_badge(self, namespace: Optional[str]) -> str:
        """Create namespace badge with error handling.
        
        Note: This method is kept for backward compatibility but the actual
        namespace badge is now created in _create_log_widget for better styling.
        """
        return ""
    
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
    
    def show(self) -> widgets.Accordion:
        """Show the log accordion (alias for display for backward compatibility)."""
        if not self._initialized:
            self.initialize()
        return self._ui_components['accordion']
    
    def _ipython_display_(self):
        """IPython display integration."""
        return self.display()._ipython_display_()
