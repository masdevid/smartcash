"""
Info components for displaying information and notifications.

This module provides UI components for displaying information in various formats
like accordions, styled boxes, and tabs, built on top of the BaseUIComponent.
"""

# Import new component classes
from .info_component import (
    InfoBox,
    InfoAccordion,
    TabbedInfo,
    # Legacy functions for backward compatibility
    create_info_accordion,
    style_info_content,
    create_tabbed_info
)

__all__ = [
    # New component classes
    'InfoBox',
    'InfoAccordion',
    'TabbedInfo',
    # Legacy functions for backward compatibility
    'create_info_accordion',
    'style_info_content',
    'create_tabbed_info'
]
