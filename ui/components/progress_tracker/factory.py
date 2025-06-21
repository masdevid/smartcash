"""
File: smartcash/ui/components/progress_tracker/factory.py
Deskripsi: Updated factory functions dengan auto hide 1 jam dan tanpa step info
"""

from typing import Dict, List, Any
from smartcash.ui.components.progress_tracker.progress_config import ProgressConfig, ProgressLevel
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker

def create_single_progress_tracker(operation: str = "Process", auto_hide: bool = False) -> ProgressTracker:
    """Create single-level progress tracker dengan optional auto hide"""
    config = ProgressConfig(level=ProgressLevel.SINGLE, operation=operation, auto_hide=auto_hide)
    return ProgressTracker(config)

def create_dual_progress_tracker(operation: str = "Process", auto_hide: bool = False) -> ProgressTracker:
    """Create dual-level progress tracker dengan optional auto hide"""
    config = ProgressConfig(level=ProgressLevel.DUAL, operation=operation, auto_hide=auto_hide)
    return ProgressTracker(config)

def create_triple_progress_tracker(operation: str = "Process", 
                                 steps: List[str] = None,
                                 step_weights: Dict[str, int] = None,
                                 auto_hide: bool = False) -> ProgressTracker:
    """Create triple-level progress tracker tanpa step info display"""
    steps = steps or ["Initialization", "Processing", "Completion"]
    
    config = ProgressConfig(
        level=ProgressLevel.TRIPLE, operation=operation,
        steps=steps, step_weights=step_weights or {},
        auto_hide=auto_hide, show_step_info=False
    )
    return ProgressTracker(config)

def create_flexible_tracker(config: ProgressConfig) -> ProgressTracker:
    """Create tracker dengan custom configuration"""
    return ProgressTracker(config)

def create_three_progress_tracker(auto_hide: bool = False) -> Dict[str, Any]:
    """Backward compatibility tanpa step info widget"""
    tracker = create_triple_progress_tracker(auto_hide=auto_hide)
    return {
        'container': tracker.container,
        'progress_container': tracker.container,
        'status_widget': tracker.status_widget,
        'step_info_widget': None,  # Always None tanpa step info
        'tracker': tracker,
        'show_container': tracker.show,
        'hide_container': tracker.hide,
        'show_for_operation': tracker.show,
        'update_overall': tracker.update_overall,
        'update_step': tracker.update_step,
        'update_current': tracker.update_current,
        'update_progress': tracker.update,
        'complete_operation': tracker.complete,
        'error_operation': tracker.error,
        'reset_all': tracker.reset
    }