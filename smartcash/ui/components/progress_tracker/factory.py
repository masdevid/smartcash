"""
File: smartcash/ui/components/progress_tracker/factory.py
Deskripsi: Updated factory functions dengan auto hide 1 jam dan tanpa step info
"""

from typing import Dict, List, Any
from smartcash.ui.components.progress_tracker.types import ProgressConfig, ProgressLevel
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker

def create_single_progress_tracker(operation: str = "Process", auto_hide: bool = False) -> ProgressTracker:
    """Create single-level progress tracker dengan optional auto hide"""
    config = ProgressConfig(
        level=ProgressLevel.SINGLE, 
        operation=operation, 
        steps=["Progress"],
        auto_hide=auto_hide
    )
    return ProgressTracker("single_progress_tracker", config)

def create_dual_progress_tracker(operation: str = "Process", auto_hide: bool = False) -> ProgressTracker:
    """Create dual-level progress tracker dengan optional auto hide"""
    config = ProgressConfig(
        level=ProgressLevel.DUAL,
        operation=operation,
        steps=["Overall", "Current"],
        auto_hide=auto_hide
    )
    return ProgressTracker("dual_progress_tracker", config)

def create_triple_progress_tracker(operation: str = "Process", 
                                 steps: List[str] = None,
                                 step_weights: Dict[str, int] = None,
                                 auto_hide: bool = False) -> ProgressTracker:
    """Create triple-level progress tracker tanpa step info display"""
    steps = steps or ["Initialization", "Processing", "Completion"]
    
    config = ProgressConfig(
        level=ProgressLevel.TRIPLE, 
        operation=operation,
        steps=steps, 
        step_weights=step_weights or {},
        auto_hide=auto_hide
    )
    return ProgressTracker("triple_progress_tracker", config)

def create_flexible_tracker(config: ProgressConfig) -> ProgressTracker:
    """Create tracker dengan custom configuration"""
    return ProgressTracker("flexible_tracker", config)
