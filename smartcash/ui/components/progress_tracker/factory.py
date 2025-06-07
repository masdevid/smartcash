"""
File: smartcash/ui/components/progress_tracker/factory.py
Deskripsi: Factory functions untuk membuat progress tracker instances
"""

from typing import Dict, List, Any
from smartcash.ui.components.progress_tracker.progress_config import ProgressConfig, ProgressLevel
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker

def create_single_progress_tracker(operation: str = "Process") -> ProgressTracker:
    """Create single-level progress tracker"""
    config = ProgressConfig(level=ProgressLevel.SINGLE, operation=operation)
    return ProgressTracker(config)

def create_dual_progress_tracker(operation: str = "Process") -> ProgressTracker:
    """Create dual-level progress tracker"""
    config = ProgressConfig(level=ProgressLevel.DUAL, operation=operation)
    return ProgressTracker(config)

def create_triple_progress_tracker(operation: str = "Process", 
                                 steps: List[str] = None,
                                 step_weights: Dict[str, int] = None) -> ProgressTracker:
    """Create triple-level progress tracker"""
    steps = steps or ["Initialization", "Processing", "Completion"]
    
    config = ProgressConfig(
        level=ProgressLevel.TRIPLE, operation=operation,
        steps=steps, step_weights=step_weights or {},
        auto_hide_delay=0.0  # Disable auto hide to avoid threading
    )
    return ProgressTracker(config)

def create_flexible_tracker(config: ProgressConfig) -> ProgressTracker:
    """Create tracker dengan custom configuration"""
    # Disable auto hide untuk avoid threading
    config.auto_hide_delay = 0.0
    return ProgressTracker(config)

def create_three_progress_tracker() -> Dict[str, Any]:
    """Backward compatibility untuk existing code"""
    tracker = create_triple_progress_tracker()
    return {
        'container': tracker.container,
        'progress_container': tracker.container,
        'status_widget': tracker.status_widget,
        'step_info_widget': tracker.step_info_widget,
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