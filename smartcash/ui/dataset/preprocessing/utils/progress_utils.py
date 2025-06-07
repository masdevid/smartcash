"""
File: smartcash/ui/dataset/preprocessing/utils/progress_utils.py
Deskripsi: Progress callback utilities untuk preprocessing operations
"""

from typing import Dict, Any, Callable
from .ui_utils import is_milestone_step

def create_progress_callback(ui_components: Dict[str, Any]) -> Callable[[str, int, int, str], None]:
    """Create progress callback untuk preprocessing operations dengan dual tracker"""
    def progress_callback(step: str, current: int, total: int, message: str):
        try:
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker:
                # Update overall progress
                overall_progress = min(100, (current / max(total, 1)) * 100)
                progress_tracker.update_overall(int(overall_progress), message)
                
                # Update step progress berdasarkan current step
                step_progress = map_step_to_progress(step, current, total)
                progress_tracker.update_step(step_progress, f"Step: {step}")
            
            # Only log important milestones
            if is_milestone_step(step, current):
                logger = ui_components.get('logger')
                if logger:
                    logger.info(message)
        except Exception:
            pass  # Silent fail to prevent blocking
    
    return progress_callback

def map_step_to_progress(step: str, current: int, total: int) -> int:
    """Map preprocessing step to progress percentage"""
    step_ranges = {
        'validate': (0, 20),
        'analyze': (20, 30),
        'normalize': (30, 70),
        'resize': (70, 90),
        'save': (90, 98),
        'finalize': (98, 100)
    }
    
    normalized_step = step.lower().split('_')[0]
    if normalized_step in step_ranges:
        start, end = step_ranges[normalized_step]
        if total > 0:
            step_progress = start + ((current / total) * (end - start))
            return min(100, max(0, int(step_progress)))
    
    return min(100, (current / max(total, 1)) * 100)

def setup_progress_tracker(ui_components: Dict[str, Any], operation_name: str = "Dataset Preprocessing"):
    """Setup progress tracker untuk preprocessing operation"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        # Define preprocessing steps
        steps = ["validate", "analyze", "normalize", "save"]
        step_weights = {"validate": 20, "analyze": 10, "normalize": 60, "save": 10}
        
        # Show with steps
        progress_tracker.show(operation_name, steps, step_weights)
        progress_tracker.update_overall(0, f"🚀 Memulai {operation_name.lower()}...")
    
    logger = ui_components.get('logger')
    if logger:
        logger.info(f"🚀 Memulai {operation_name.lower()}")

def complete_progress_tracker(ui_components: Dict[str, Any], message: str):
    """Complete progress tracker dengan success message"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker and hasattr(progress_tracker, 'complete'):
        progress_tracker.complete(message)

def error_progress_tracker(ui_components: Dict[str, Any], error_msg: str):
    """Set error state pada progress tracker"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker and hasattr(progress_tracker, 'error'):
        progress_tracker.error(error_msg)

def reset_progress_tracker(ui_components: Dict[str, Any]):
    """Reset progress tracker"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker and hasattr(progress_tracker, 'reset'):
        progress_tracker.reset()