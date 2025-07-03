"""
Legacy compatibility layer for the LogAccordion component.

This module provides backward compatibility with the old procedural API.
"""

from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, HTML

from .log_accordion import LogAccordion
from .log_level import LogLevel

# Global registry of log accordion instances
_log_accordions: Dict[str, LogAccordion] = {}


def create_log_accordion(
    name: str = "log_accordion",
    module_name: str = "Process",
    height: str = "300px",
    width: str = "100%",
    max_logs: int = 1000,
    show_timestamps: bool = True,
    show_level_icons: bool = True,
    auto_scroll: bool = True,
    enable_deduplication: bool = True
) -> Dict[str, Any]:
    """Create a new log accordion (legacy API).
    
    Args:
        name: Unique name for this log accordion
        module_name: Name to display in the accordion header
        height: Height of the log container
        width: Width of the component
        max_logs: Maximum number of log entries to keep in memory
        show_timestamps: Whether to show timestamps
        show_level_icons: Whether to show level icons
        auto_scroll: Whether to automatically scroll to bottom on new messages
        enable_deduplication: Whether to enable message deduplication
        
    Returns:
        Dictionary containing the log accordion UI components
    """
    # Create new LogAccordion instance
    log_accordion = LogAccordion(
        component_name=name,
        module_name=module_name,
        height=height,
        width=width,
        max_logs=max_logs,
        show_timestamps=show_timestamps,
        show_level_icons=show_level_icons,
        auto_scroll=auto_scroll,
        enable_deduplication=enable_deduplication
    )
    
    # Store the instance in the global registry
    _log_accordions[name] = log_accordion
    
    # Initialize and get the UI components
    log_accordion.initialize()
    ui_components = {
        'accordion': log_accordion.display(),
        'log_container': log_accordion._ui_components['log_container'],
        'entries_container': log_accordion._ui_components['entries_container']
    }
    
    # Add legacy methods to the UI components dict
    ui_components['append_log'] = lambda *args, **kwargs: log_accordion.log(*args, **kwargs)
    ui_components['clear'] = log_accordion.clear
    
    return ui_components


def get_log_accordion(name: str = "log_accordion") -> Optional[Dict[str, Any]]:
    """Get an existing log accordion by name (legacy API).
    
    Args:
        name: Name of the log accordion to retrieve
        
    Returns:
        Dictionary containing the log accordion UI components, or None if not found
    """
    log_accordion = _log_accordions.get(name)
    if not log_accordion:
        return None
        
    return {
        'accordion': log_accordion.display(),
        'log_container': log_accordion._ui_components['log_container'],
        'entries_container': log_accordion._ui_components['entries_container'],
        'append_log': lambda *args, **kwargs: log_accordion.log(*args, **kwargs),
        'clear': log_accordion.clear
    }


def update_log(
    log_accordion_name: str = "log_accordion",
    message: Optional[str] = None,
    level: Union[LogLevel, str] = LogLevel.INFO,
    namespace: Optional[str] = None,
    module: Optional[str] = None,
    timestamp: Optional[datetime] = None
) -> None:
    """Update the log display (legacy API).
    
    Args:
        log_accordion_name: Name of the log accordion to update
        message: Optional message to log
        level: Log level (default: INFO)
        namespace: Optional namespace for the log
        module: Optional module name
        timestamp: Optional timestamp (default: current time)
    """
    if message is not None:
        log(
            message=message,
            level=level,
            namespace=namespace,
            module=module,
            timestamp=timestamp,
            log_accordion_name=log_accordion_name
        )

def log(
    message: str,
    level: Union[LogLevel, str] = LogLevel.INFO,
    namespace: Optional[str] = None,
    module: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    log_accordion_name: str = "log_accordion"
) -> None:
    """Append a log message to the specified log accordion (legacy API).
    
    Args:
        message: The log message
        level: Log level (default: INFO)
        namespace: Optional namespace for the log
        module: Optional module name
        timestamp: Optional timestamp (default: current time)
        log_accordion_name: Name of the log accordion to append to
    """
    log_accordion = _log_accordions.get(log_accordion_name)
    if not log_accordion:
        # Create a new log accordion if it doesn't exist
        log_accordion = LogAccordion(component_name=log_accordion_name)
        _log_accordions[log_accordion_name] = log_accordion
        log_accordion.initialize()
    
    log_accordion.log(
        message=message,
        level=level,
        namespace=namespace,
        module=module,
        timestamp=timestamp
    )


# Add legacy functions to the module for backward compatibility
__all__ = [
    'create_log_accordion',
    'get_log_accordion',
    'log',
    'LogLevel'
]
