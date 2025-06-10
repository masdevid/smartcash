"""
File: smartcash/ui/components/progress_tracker/factory.py
Deskripsi: Fixed factory functions dengan visible layout dan proper initialization
"""

from typing import Dict, List, Any
from smartcash.ui.components.progress_tracker.progress_config import ProgressConfig, ProgressLevel
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker

def create_single_progress_tracker(operation: str = "Process", auto_hide: bool = False) -> ProgressTracker:
    """Create single-level progress tracker dengan visible layout"""
    config = ProgressConfig(level=ProgressLevel.SINGLE, operation=operation, auto_hide=auto_hide)
    tracker = ProgressTracker(config)
    _ensure_visible_layout(tracker)
    return tracker

def create_dual_progress_tracker(operation: str = "Process", auto_hide: bool = False) -> ProgressTracker:
    """Create dual-level progress tracker dengan visible layout"""
    config = ProgressConfig(level=ProgressLevel.DUAL, operation=operation, auto_hide=auto_hide)
    tracker = ProgressTracker(config)
    _ensure_visible_layout(tracker)
    return tracker

def create_triple_progress_tracker(operation: str = "Process", 
                                 steps: List[str] = None,
                                 step_weights: Dict[str, int] = None,
                                 auto_hide: bool = False) -> ProgressTracker:
    """Create triple-level progress tracker dengan visible layout"""
    steps = steps or ["Initialization", "Processing", "Completion"]
    
    config = ProgressConfig(
        level=ProgressLevel.TRIPLE, operation=operation,
        steps=steps, step_weights=step_weights or {},
        auto_hide=auto_hide, show_step_info=False
    )
    tracker = ProgressTracker(config)
    _ensure_visible_layout(tracker)
    return tracker

def create_flexible_tracker(config: ProgressConfig) -> ProgressTracker:
    """Create tracker dengan custom configuration dan visible layout"""
    tracker = ProgressTracker(config)
    _ensure_visible_layout(tracker)
    return tracker

def create_three_progress_tracker(auto_hide: bool = False) -> Dict[str, Any]:
    """🔑 KEY FIX: Backward compatibility dengan forced visible layout"""
    tracker = create_triple_progress_tracker(auto_hide=auto_hide)
    
    # 🎯 CRITICAL: Ensure container is visible dari awal
    if hasattr(tracker, 'container') and hasattr(tracker.container, 'layout'):
        tracker.container.layout.visibility = 'visible'
        tracker.container.layout.display = 'flex'
        tracker.container.layout.height = 'auto'
        tracker.container.layout.width = '100%'
    
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

def _ensure_visible_layout(tracker: ProgressTracker):
    """🔑 KEY: Ensure tracker container is visible dari initialization"""
    try:
        if hasattr(tracker, 'container') and hasattr(tracker.container, 'layout'):
            layout = tracker.container.layout
            layout.visibility = 'visible'
            layout.display = 'flex'
            layout.height = 'auto'
            layout.width = '100%'
            layout.margin = '10px 0'
            layout.padding = '10px'
            
        # Also ensure all child progress bars are visible
        if hasattr(tracker, 'overall_progress') and hasattr(tracker.overall_progress, 'layout'):
            tracker.overall_progress.layout.visibility = 'visible'
            tracker.overall_progress.layout.display = 'flex'
            
        if hasattr(tracker, 'current_progress') and hasattr(tracker.current_progress, 'layout'):
            tracker.current_progress.layout.visibility = 'visible'
            tracker.current_progress.layout.display = 'flex'
            
        if hasattr(tracker, 'step_progress') and hasattr(tracker.step_progress, 'layout'):
            tracker.step_progress.layout.visibility = 'visible'
            tracker.step_progress.layout.display = 'flex'
            
    except Exception as e:
        # Silent fail to prevent initialization issues
        print(f"Warning: Could not ensure visible layout: {str(e)}")

# Enhanced factory dengan explicit visibility
def create_visible_dual_progress_tracker(operation: str = "Dataset Processing") -> Dict[str, Any]:
    """🔑 KEY: Create dual progress tracker yang pasti visible"""
    tracker = create_dual_progress_tracker(operation=operation, auto_hide=False)
    
    # Force visibility untuk semua components
    try:
        # Container visibility
        if hasattr(tracker, 'container'):
            container = tracker.container
            if hasattr(container, 'layout'):
                container.layout.visibility = 'visible'
                container.layout.display = 'flex'
                container.layout.flex_flow = 'column'
                container.layout.width = '100%'
                container.layout.height = 'auto'
                container.layout.margin = '15px 0'
                container.layout.padding = '15px'
                container.layout.border = '1px solid #e0e0e0'
                container.layout.border_radius = '8px'
                container.layout.background_color = '#f8f9fa'
        
        # Show immediately
        if hasattr(tracker, 'show'):
            tracker.show()
            
        # Update dengan initial message
        if hasattr(tracker, 'update_overall'):
            tracker.update_overall(0, f"📋 {operation} ready to start")
            
    except Exception as e:
        print(f"Warning: Error setting up visible tracker: {str(e)}")
    
    return {
        'tracker': tracker,
        'container': tracker.container,
        'show': tracker.show,
        'hide': tracker.hide,
        'update_overall': tracker.update_overall,
        'update_current': tracker.update_current,
        'complete': tracker.complete,
        'error': tracker.error,
        'reset': tracker.reset
    }