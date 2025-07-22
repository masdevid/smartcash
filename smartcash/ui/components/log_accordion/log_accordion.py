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
    
    # Constants for enhanced deduplication
    DEFAULT_DUPLICATE_WINDOW_MS = 2000  # 2 second window for considering messages as duplicates
    DEFAULT_MAX_DUPLICATE_COUNT = 5     # Maximum number of duplicates to show before hiding
    DEFAULT_MESSAGE_CACHE_SIZE = 100    # Cache size for recent message hashes
    
    def __init__(
        self,
        component_name: str = "log_accordion",
        module_name: str = 'Process',
        height: str = '150px',
        width: str = '100%',
        max_logs: int = 1000,
        show_timestamps: bool = True,
        show_level_icons: bool = True,
        enable_deduplication: bool = True,
        duplicate_window_ms: Optional[int] = None,
        max_duplicate_count: Optional[int] = None,
        namespace_filter: Optional[Union[str, List[str]]] = None
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

        self.enable_deduplication = enable_deduplication
        self.duplicate_window_ms = duplicate_window_ms or self.DEFAULT_DUPLICATE_WINDOW_MS
        self.max_duplicate_count = max_duplicate_count or self.DEFAULT_MAX_DUPLICATE_COUNT
        # Namespace filtering
        self.namespace_filter = namespace_filter
        
        # Initialize state with enhanced deduplication
        self.log_entries: List[LogEntry] = []
        self.last_entry: Optional[LogEntry] = None
        self.duplicate_count: int = 0
        
        # Enhanced deduplication cache
        self._message_cache: Dict[str, int] = {}  # message_hash -> count
        self._last_cleanup_time: float = time.time()
        
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
            
            /* Compact log entry styling with text wrapping */
            .log-entry {{
                width: 100% !important;
                max-width: 100% !important;
                overflow: visible !important;
                word-wrap: break-word !important;
                word-break: break-word !important;
                white-space: pre-wrap !important;
                padding: 2px 8px !important;
                margin: 0 !important;
                border-radius: 2px !important;
                border-left: 2px solid transparent !important;
                cursor: default !important;
                overflow-wrap: break-word !important;
                hyphens: auto !important;
                line-height: 1.3 !important;
                font-size: 11.5px !important;
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace !important;
            }}
            
            /* Simplified hover effect for performance */
            .log-entry:hover {{
                background: rgba(0, 0, 0, 0.02) !important;
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
        
        # Enhanced deduplication handling
        if self.enable_deduplication:
            # Create message hash for efficient duplicate detection
            message_hash = self._create_message_hash(entry.message, entry.level, entry.namespace)
            
            # Check if this message has been seen recently
            if self._is_duplicate_message(message_hash):
                self._increment_duplicate_cache(message_hash)
                return  # Skip adding duplicate message
            
            # Traditional deduplication for consecutive messages
            if self.last_entry and entry.is_duplicate_of(self.last_entry, self.duplicate_window_ms):
                self.last_entry.increment_duplicate_count(self.max_duplicate_count)
                self._update_log_display()
                return
            
            # Add to cache for future duplicate detection
            self._message_cache[message_hash] = 1
        
        # Add new entry
        self.log_entries.append(entry)
        self.last_entry = entry
        
        # Trim old entries if needed
        if len(self.log_entries) > self.max_logs:
            self.log_entries.pop(0)
        
        # Periodic cache cleanup for performance
        self._cleanup_message_cache()
        
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
        """Optimized log display update with minimal DOM operations."""
        if not self._initialized:
            self.initialize()
            
        # Get filtered entries
        filtered_entries = self._get_filtered_entries()
        
        # Enhanced change detection - check content hash instead of just count
        current_hash = self._calculate_entries_hash(filtered_entries)
        if hasattr(self, '_last_entries_hash') and current_hash == self._last_entries_hash:
            return  # No changes, skip update
            
        self._last_entries_hash = current_hash
        
        # Efficient entry widget creation
        entries = [self._create_log_widget(entry) for entry in filtered_entries[-50:]]  # Limit to last 50 for performance
        
        # Update the display
        self._ui_components['entries_container'].children = entries
        
        # Auto-scroll functionality has been removed
    
    def _calculate_entries_hash(self, entries: List[LogEntry]) -> str:
        """Calculate a hash of current entries for change detection."""
        if not entries:
            return "empty"
        
        # Use last few entries and their counts for efficient comparison
        last_entries = entries[-10:] if len(entries) > 10 else entries
        content = ''.join([
            f"{entry.message[:50]}:{entry.level.value}:{getattr(entry, 'count', 1)}"
            for entry in last_entries
        ])
        
        import hashlib
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
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
            ('smartcash.common.', 'sc.common.'),
            ('smartcash.dataset.', 'sc.dataset.'),
            ('smartcash.model.', 'sc.model.'),
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
            
            # Use the full namespace
            full_namespace = entry.namespace or ''
                
            # Handle multi-line messages (e.g., tracebacks)
            message_lines = entry.message.split('\n')
            main_message = message_lines[0]  # First line only for compact view
            has_more_lines = len(message_lines) > 1
            
            # Create expandable content for multi-line messages
            if has_more_lines:
                traceback_id = f"traceback-{id(entry)}"
                additional_content = '\n'.join(message_lines[1:]).strip()
                
                expandable_html = f"""
                <div style='cursor: pointer; color: #6c757d; font-size: 0.85em; padding: 2px 0 2px 22px;'
                     onclick='this.nextElementSibling.classList.toggle("show-traceback"); this.textContent = this.textContent.includes("Show") ? "▲ Hide details" : "▼ Show details"'>
                    ▼ Show details
                </div>
                <div class='error-traceback' id='{traceback_id}' style='display: none; margin-top: 4px; padding-left: 22px;'>
                    <pre style='margin: 0; padding: 8px; background: rgba(0,0,0,0.05); border-radius: 4px; white-space: pre-wrap; font-size: 11px; font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;'>{additional_content}</pre>
                </div>
                """
            else:
                expandable_html = ""
            
            # Always use compact style for log entries
            entry_class = 'log-entry-compact'
            
            # Create compact flex layout
            html = f"""
            <div class='{entry_class}' style='
                padding: 2px 8px;
                margin: 1px 0;
                border-left: 3px solid {style['color']};
                background: {style['bg']};
                border-radius: 3px;
                transition: all 0.15s ease;
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;'>
                
                <div style='display: flex; gap: 6px; align-items: flex-start; line-height: 1.3;'>
                    <!-- Icon -->
                    <div style='flex: 0 0 16px; margin-top: 1px;'>{level_emoji}</div>
                    
                    <!-- Message Content -->
                    <div style='flex: 1; min-width: 0;'>
                        <div style='display: flex; flex-wrap: wrap; gap: 4px; align-items: baseline;'>
                            <!-- Namespace Badge -->
                            {f'''
                            <span style="
                                display: inline-block;
                                background: #e9ecef;
                                color: #495057;
                                font-size: 0.75em;
                                font-weight: 500;
                                padding: 1px 6px;
                                border-radius: 10px;
                                margin-right: 6px;
                                white-space: nowrap;
                                overflow: hidden;
                                text-overflow: ellipsis;
                                max-width: 200px;
                                border: 1px solid #dee2e6;
                                line-height: 1.3;"
                                title="{full_namespace}">
                                {full_namespace.split('.')[-1]}
                            </span>
                            ''' if full_namespace else ''}
                            
                            <!-- Message -->
                            <div style='color: {style['text_color']}; word-break: break-word; flex: 1; min-width: 100px;'>{main_message}{duplicate_counter}</div>
                            
                            <!-- Timestamp -->
                            <span style='color: #6c757d; font-size: 0.85em; flex-shrink: 0; white-space: nowrap;'>{timestamp}</span>
                        </div>
                        {expandable_html}
                    </div>
                </div>
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
    
    # Removed _scroll_to_bottom method as auto-scroll is disabled
    
    def clear(self) -> None:
        """Clear all log entries and deduplication cache."""
        self.log_entries = []
        self.last_entry = None
        self.duplicate_count = 0
        
        # Clear enhanced deduplication cache
        self._message_cache.clear()
        self._last_cleanup_time = time.time()
        
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
        from IPython.display import display
        display(self.display())
    
    # Enhanced deduplication helper methods
    
    def _create_message_hash(self, message: str, level: LogLevel, namespace: Optional[str] = None) -> str:
        """Create a hash for message deduplication."""
        import hashlib
        content = f"{message}:{level.value}:{namespace or ''}"
        return hashlib.md5(content.encode()).hexdigest()[:16]  # Short hash for performance
    
    def _is_duplicate_message(self, message_hash: str) -> bool:
        """Check if message hash already exists in cache."""
        return message_hash in self._message_cache and self._message_cache[message_hash] >= self.max_duplicate_count
    
    def _increment_duplicate_cache(self, message_hash: str) -> None:
        """Increment duplicate count in cache."""
        if message_hash in self._message_cache:
            self._message_cache[message_hash] += 1
    
    def _cleanup_message_cache(self) -> None:
        """Periodic cleanup of message cache to prevent memory bloat."""
        current_time = time.time()
        
        # Cleanup every 30 seconds
        if current_time - self._last_cleanup_time > 30:
            # Keep only recent entries, limit cache size
            if len(self._message_cache) > self.DEFAULT_MESSAGE_CACHE_SIZE:
                # Remove oldest half of entries (simple LRU approximation)
                items = list(self._message_cache.items())
                keep_count = self.DEFAULT_MESSAGE_CACHE_SIZE // 2
                self._message_cache = dict(items[-keep_count:])
            
            self._last_cleanup_time = current_time
