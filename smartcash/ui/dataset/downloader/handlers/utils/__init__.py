"""
UI utilities for downloader handlers.
"""

from .ui_utils import (
    display_check_results,
    show_confirmation_area,
    hide_confirmation_area,
    map_step_to_current_progress,
    is_milestone_step,
    format_file_count,
    format_duration,
    get_log_emoji,
    safe_get_widget_value
)
from .progress_utils import create_progress_callback

__all__ = [
    'display_check_results',
    'show_confirmation_area',
    'hide_confirmation_area',
    'map_step_to_current_progress',
    'is_milestone_step',
    'format_file_count',
    'format_duration',
    'get_log_emoji',
    'safe_get_widget_value',
    'create_progress_callback'
]