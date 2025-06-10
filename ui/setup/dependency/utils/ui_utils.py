"""
File: smartcash/ui/setup/dependency/utils/ui_utils.py
Deskripsi: Utilitas tampilan UI untuk dependency installer dengan absolute imports
"""

from typing import Dict, Any, Optional, Callable, List
import ipywidgets as widgets
from smartcash.ui.utils.constants import COLORS

def log_to_ui_safe(ui_components: Dict[str, Any], message: str, level: str = 'info') -> None:
    """Log ke UI dengan penanganan error yang aman"""
    try:
        log_output = ui_components.get('log_output')
        if log_output and hasattr(log_output, 'append_log'):
            log_output.append_log(message, level)
        elif log_output and hasattr(log_output, 'append'):
            # Fallback untuk log_output lama
            color_map = {
                'info': COLORS.get('TEXT', 'black'),
                'success': COLORS.get('SUCCESS', 'green'),
                'warning': COLORS.get('WARNING', 'orange'),
                'error': COLORS.get('ERROR', 'red'),
                'debug': COLORS.get('MUTED', 'gray')
            }
            color = color_map.get(level, COLORS.get('TEXT', 'black'))
            log_output.append(f"<span style='color: {color};'>{message}</span>")
    except Exception as e:
        # Silent fail untuk mencegah loop error
        pass

def update_status_panel(ui_components: Dict[str, Any], message: str, level: str = 'info') -> None:
    """Update status panel dengan penanganan error yang aman"""
    try:
        status_panel = ui_components.get('status_panel')
        if status_panel and hasattr(status_panel, 'update'):
            status_panel.update(message, level)
    except Exception as e:
        # Silent fail
        pass

def clear_ui_outputs(ui_components: Dict[str, Any], output_keys: Optional[List[str]] = None) -> None:
    """Clear multiple UI outputs dengan one-liner pattern"""
    output_keys = output_keys or ['log_output', 'status', 'confirmation_area']
    [widget.clear_output(wait=True) for key in output_keys 
     if (widget := ui_components.get(key)) and hasattr(widget, 'clear_output')]

def reset_ui_logger(ui_components: Dict[str, Any]) -> None:
    """Reset UI logger output untuk clear previous logs"""
    log_output = ui_components.get('log_output')
    if log_output and hasattr(log_output, 'clear_output'):
        log_output.clear_output(wait=True)

def update_progress_step(ui_components: Dict[str, Any], progress_type: str, value: int, 
                         message: str = "", color: str = None) -> None:
    """Update progress dengan API progress tracker baru dan safe error handling"""
    try:
        # Gunakan fungsi update_progress dari ui_components jika tersedia (backward compatibility wrapper)
        update_progress = ui_components.get('update_progress')
        if update_progress and callable(update_progress):
            update_progress(progress_type, value, message, color)
            return
        
        # Jika tidak ada update_progress, gunakan progress_tracker langsung
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            # Gunakan metode yang benar sesuai API progress tracker baru
            if progress_type == 'overall' or progress_type == 'level1':
                if hasattr(progress_tracker, 'update_overall'):
                    progress_tracker.update_overall(value, message, color)
            elif progress_type == 'step' or progress_type == 'level2':
                if hasattr(progress_tracker, 'update_current'):
                    progress_tracker.update_current(value, message, color)
            elif progress_type == 'step_progress' or progress_type == 'level3':
                # Untuk triple level progress tracker
                if hasattr(progress_tracker, 'update_step_progress'):
                    progress_tracker.update_step_progress(value, message, color)
                elif hasattr(progress_tracker, 'update_current'):
                    # Fallback ke current jika tidak ada step_progress
                    progress_tracker.update_current(value, message, color)
    except Exception as e:
        # Silent fail untuk compatibility
        logger = ui_components.get('logger')
        if logger:
            logger.debug(f"🔄 Progress update error (non-critical): {str(e)}")
