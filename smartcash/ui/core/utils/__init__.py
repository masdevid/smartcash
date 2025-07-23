"""
File: smartcash/ui/core/utils/__init__.py
Core Utilities Module.

This module provides utility functions for the SmartCash UI core functionality.
"""

from .factory_utils import create_ui_factory_method, create_display_function
from .markdown_formatter import (
    MarkdownHTMLFormatter,
    get_markdown_formatter,
    format_summary_to_html
)

__all__ = [
    'create_ui_factory_method',
    'create_display_function',
    'MarkdownHTMLFormatter',
    'get_markdown_formatter', 
    'format_summary_to_html'
]
