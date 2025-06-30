"""
File: smartcash/ui/dataset/downloader/utils/ui_utils.py
Deskripsi: Optimized UI utilities for download operations
"""
from typing import Dict, Any

def log_to_accordion(ui_components: Dict[str, Any], message: str, level: str = 'info') -> None:
    """Log message using the logger_bridge if available.
    
    Args:
        ui_components: Dictionary containing UI components
        message: The message to log
        level: Log level (info, warning, error)
    """
    logger_bridge = ui_components.get('logger_bridge')
    if logger_bridge:
        # Map level to logger_bridge method
        log_methods = {
            'info': logger_bridge.info,
            'success': logger_bridge.info,  # Fallback to info if success not available
            'warning': logger_bridge.warning,
            'error': logger_bridge.error
        }
        log_method = log_methods.get(level, logger_bridge.info)
        log_method(message)
    
    # Auto-expand for errors/warnings
    if level in ['error', 'warning'] and 'log_accordion' in ui_components:
        if hasattr(ui_components['log_accordion'], 'selected_index'):
            ui_components['log_accordion'].selected_index = 0

def map_step_to_current_progress(step: str, overall_progress: int) -> int:
    """Map step progress to current operation progress bar.
    
    Args:
        step: Current step in the download process
        overall_progress: Overall progress percentage (0-100)
        
    Returns:
        Mapped progress percentage for the current step
    """
    step_mapping = {
        'init': (0, 10), 'metadata': (10, 20), 'backup': (20, 25),
        'download': (25, 70), 'extract': (70, 80), 'organize': (80, 90),
        'uuid_rename': (90, 95), 'validate': (95, 98), 'cleanup': (98, 100)
    }
    
    step_key = step.lower().split('_')[0]
    if step_key in step_mapping:
        start, end = step_mapping[step_key]
        range_size = end - start
        step_progress = start + (overall_progress * range_size / 100)
        return min(100, max(0, int(step_progress)))
    return overall_progress

def is_milestone_step(step: str, progress: int) -> bool:
    """Check if the current step is a milestone that should be logged.
    
    Args:
        step: Current step name
        progress: Current progress percentage
        
    Returns:
        True if this is a milestone step, False otherwise
    """
    milestone_steps = ['init', 'metadata', 'backup', 'extract', 'organize', 'validate', 'complete']
    return (step.lower() in milestone_steps or 
            progress in [0, 25, 50, 75, 100] or 
            progress % 25 == 0)