"""
LogAccordion - A modern, extensible log display component for Jupyter notebooks.

This component provides a rich, interactive log display with features like:
- Multiple log levels with color coding
- Message deduplication
- Smooth scrolling
- Namespace support
- Customizable appearance
"""

from .log_accordion import LogAccordion
from .log_level import LogLevel, get_log_level_style
from .log_entry import LogEntry
from .legacy import create_log_accordion

__all__ = ['LogAccordion', 'LogLevel', 'LogEntry', 'get_log_level_style' 'create_log_accordion']
