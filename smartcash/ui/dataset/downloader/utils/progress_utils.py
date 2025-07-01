"""
File: smartcash/ui/dataset/downloader/utils/progress_utils.py
Deskripsi: Progress callback utilities untuk download operations
"""

from typing import Dict, Any, Callable, Optional
from smartcash.ui.utils.ui_logger import UILogger
from .ui_utils import is_milestone_step, map_step_to_current_progress

def create_progress_callback(ui_components: Dict[str, Any]) -> Callable[[str, int, int, str], None]:
    """Create a progress callback function for download operations.
    
    Args:
        ui_components: Dictionary containing UI components including logger_bridge
        
    Returns:
        A callback function that can be used to report progress
    """
    """Create minimal progress callback untuk download operations"""
    def progress_callback(step: str, current: int, total: int, message: str):
        try:
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker:
                # Update both overall and current operation progress
                if hasattr(progress_tracker, 'update_overall'):
                    progress_tracker.update_overall(current, message)
                if hasattr(progress_tracker, 'update_current'):
                    # Map step to current operation progress
                    step_progress = map_step_to_current_progress(step, current)
                    progress_tracker.update_current(step_progress, f"Step: {step}")
            
            # Only log important milestones, not every progress update
            if is_milestone_step(step, current):
                logger_bridge = ui_components.get('logger_bridge')
                if logger_bridge:
                    logger_bridge.info(message)
        except Exception:
            pass  # Silent fail to prevent blocking
    
    return progress_callback