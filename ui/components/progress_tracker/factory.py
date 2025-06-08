"""
File: smartcash/ui/components/progress_tracker/factory.py
Deskripsi: Fixed factory dengan proper dict return dan error handling
"""

from typing import Dict, List, Any
from smartcash.ui.components.progress_tracker.progress_config import ProgressConfig, ProgressLevel
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker

def create_triple_progress_tracker(operation: str = "Process", 
                                 steps: List[str] = None,
                                 step_weights: Dict[str, int] = None,
                                 auto_hide: bool = False) -> Dict[str, Any]:
    """FIXED: Always return dict dengan proper widget access"""
    try:
        steps = steps or ["Initialization", "Processing", "Completion"]
        
        config = ProgressConfig(
            level=ProgressLevel.TRIPLE, 
            operation=operation,
            steps=steps, 
            step_weights=step_weights or {},
            auto_hide=auto_hide, 
            show_step_info=False
        )
        
        tracker = ProgressTracker(config)
        
        # FIXED: Always return dict structure
        return {
            'container': tracker.container,
            'progress_container': tracker.container,
            'status_widget': tracker.status_widget,
            'step_info_widget': None,  # Always None tanpa step info
            'tracker': tracker,
            'show_container': tracker.show,
            'hide_container': tracker.hide,
            'show_for_operation': tracker.show,
            'update_overall': getattr(tracker, 'update_overall', None),
            'update_step': getattr(tracker, 'update_step', None),
            'update_current': getattr(tracker, 'update_current', None),
            'update_progress': getattr(tracker, 'update', None),
            'complete_operation': getattr(tracker, 'complete', None),
            'error_operation': getattr(tracker, 'error', None),
            'reset_all': getattr(tracker, 'reset', None)
        }
        
    except Exception as e:
        # CRITICAL: Fallback dict jika tracker creation gagal
        import ipywidgets as widgets
        
        fallback_widget = widgets.HTML(f"""
        <div style="padding: 10px; background: #f8d7da; border: 1px solid #dc3545; 
                    border-radius: 4px; color: #721c24; margin: 5px 0;">
            ⚠️ Progress tracker error: {str(e)}
        </div>
        """)
        
        # Return minimal working dict
        return {
            'container': fallback_widget,
            'progress_container': fallback_widget,
            'status_widget': fallback_widget,
            'step_info_widget': None,
            'tracker': None,
            'show_container': lambda *args: None,
            'hide_container': lambda *args: None,
            'show_for_operation': lambda *args: None,
            'update_overall': lambda *args: None,
            'update_step': lambda *args: None,
            'update_current': lambda *args: None,
            'update_progress': lambda *args: None,
            'complete_operation': lambda *args: None,
            'error_operation': lambda *args: None,
            'reset_all': lambda *args: None,
            'fallback_mode': True,
            'error': str(e)
        }

def create_single_progress_tracker(operation: str = "Process", auto_hide: bool = False) -> Dict[str, Any]:
    """Create single-level progress tracker - always return dict"""
    try:
        config = ProgressConfig(level=ProgressLevel.SINGLE, operation=operation, auto_hide=auto_hide)
        tracker = ProgressTracker(config)
        
        return {
            'container': tracker.container,
            'tracker': tracker,
            'update': getattr(tracker, 'update', None),
            'complete': getattr(tracker, 'complete', None),
            'error': getattr(tracker, 'error', None),
            'reset': getattr(tracker, 'reset', None)
        }
    except Exception as e:
        return _create_fallback_tracker(str(e))

def create_dual_progress_tracker(operation: str = "Process", auto_hide: bool = False) -> Dict[str, Any]:
    """Create dual-level progress tracker - always return dict"""
    try:
        config = ProgressConfig(level=ProgressLevel.DUAL, operation=operation, auto_hide=auto_hide)
        tracker = ProgressTracker(config)
        
        return {
            'container': tracker.container,
            'tracker': tracker,
            'update_overall': getattr(tracker, 'update_overall', None),
            'update_current': getattr(tracker, 'update_current', None),
            'complete': getattr(tracker, 'complete', None),
            'error': getattr(tracker, 'error', None),
            'reset': getattr(tracker, 'reset', None)
        }
    except Exception as e:
        return _create_fallback_tracker(str(e))

def create_flexible_tracker(config: ProgressConfig) -> Dict[str, Any]:
    """Create tracker dengan custom configuration - always return dict"""
    try:
        tracker = ProgressTracker(config)
        return {'container': tracker.container, 'tracker': tracker}
    except Exception as e:
        return _create_fallback_tracker(str(e))

def _create_fallback_tracker(error_msg: str) -> Dict[str, Any]:
    """Create fallback tracker dict jika terjadi error"""
    import ipywidgets as widgets
    
    error_widget = widgets.HTML(f"""
    <div style="padding: 8px; background: #fff3cd; border: 1px solid #ffc107; 
                border-radius: 4px; color: #856404; margin: 5px 0; font-size: 12px;">
        ⚠️ Progress tracker fallback: {error_msg}
    </div>
    """)
    
    return {
        'container': error_widget,
        'tracker': None,
        'update_overall': lambda *args: None,
        'update_step': lambda *args: None,
        'update_current': lambda *args: None,
        'update': lambda *args: None,
        'complete': lambda *args: None,
        'error': lambda *args: None,
        'reset': lambda *args: None,
        'fallback_mode': True
    }

# Backward compatibility
def create_three_progress_tracker(auto_hide: bool = False) -> Dict[str, Any]:
    """Backward compatibility - always return dict"""
    return create_triple_progress_tracker("Process", auto_hide=auto_hide)