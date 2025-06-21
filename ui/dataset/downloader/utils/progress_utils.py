"""
File: smartcash/ui/dataset/downloader/utils/progress_utils.py
Deskripsi: Progress callback utilities untuk download operations
"""

from typing import Dict, Any, Callable
from .ui_utils import is_milestone_step, map_step_to_current_progress

def create_progress_callback(ui_components: Dict[str, Any]) -> Callable[[str, int, int, str], None]:
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
                logger = ui_components.get('logger')
                if logger:
                    logger.info(message)
        except Exception:
            pass  # Silent fail to prevent blocking
    
    return progress_callback