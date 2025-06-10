"""
File: smartcash/ui/dataset/preprocessing/utils/progress_utils.py
Deskripsi: Simplified progress utilities tanpa redundant complexity
"""

from typing import Dict, Any, Callable, Optional
from smartcash.common.logger import get_logger

def create_dual_progress_callback(ui_components: Dict[str, Any]) -> Callable[[str, int, int, str], None]:
    """Create dual progress callback yang sederhana"""
    def progress_callback(level: str, current: int, total: int, message: str):
        try:
            progress_tracker = ui_components.get('progress_tracker')
            if not progress_tracker:
                return
            
            # Map level ke tracker method
            progress_pct = int((current / max(total, 1)) * 100)
            
            if level in ['overall', 'step']:
                if hasattr(progress_tracker, 'update_overall'):
                    progress_tracker.update_overall(progress_pct, message)
            else:  # current, batch, file
                if hasattr(progress_tracker, 'update_current'):
                    progress_tracker.update_current(progress_pct, message)
            
            # Log milestone progress
            if _is_milestone(current, total):
                logger = get_logger('progress_utils')
                logger.info(f"🔄 {message} ({current}/{total})")
                    
        except Exception:
            pass  # Silent fail
    
    return progress_callback

def setup_progress_tracking(ui_components: Dict[str, Any], operation_name: str = "Processing"):
    """Setup progress tracking untuk operation"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'show'):
            progress_tracker.show(operation_name)
            progress_tracker.update_overall(0, f"🚀 Memulai {operation_name.lower()}")
    except Exception:
        pass

def complete_progress_tracking(ui_components: Dict[str, Any], message: str):
    """Complete progress tracking"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'complete'):
            progress_tracker.complete(message)
    except Exception:
        pass

def error_progress_tracking(ui_components: Dict[str, Any], error_msg: str):
    """Set error state pada progress tracking"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'error'):
            progress_tracker.error(error_msg)
    except Exception:
        pass

def hide_progress_tracking(ui_components: Dict[str, Any]):
    """Hide progress tracking"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'hide'):
            progress_tracker.hide()
    except Exception:
        pass

def _is_milestone(current: int, total: int) -> bool:
    """Check if progress adalah milestone"""
    if total <= 10:
        return True
    
    milestones = [0, 10, 25, 50, 75, 90, 100]
    progress_pct = (current / total) * 100
    return any(abs(progress_pct - milestone) < 1 for milestone in milestones) or current == total

# One-liner utilities
create_progress_callback = lambda ui_components: create_dual_progress_callback(ui_components)
setup_progress = lambda ui_components, name="Processing": setup_progress_tracking(ui_components, name)
complete_progress = lambda ui_components, msg="Completed": complete_progress_tracking(ui_components, msg)
error_progress = lambda ui_components, msg="Error": error_progress_tracking(ui_components, msg)
hide_progress = lambda ui_components: hide_progress_tracking(ui_components)