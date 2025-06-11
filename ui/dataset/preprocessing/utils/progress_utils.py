"""
File: smartcash/ui/dataset/preprocessing/utils/progress_utils.py
Deskripsi: Complete progress utilities dengan proper UI integration tanpa double logging
"""

from typing import Dict, Any, Callable, Optional
from smartcash.common.logger import get_logger

def create_dual_progress_callback(ui_components: Dict[str, Any]) -> Callable[[str, int, int, str], None]:
    """🔑 KEY: Create progress callback yang terintegrasi dengan UI tracker TANPA console logging"""
    def progress_callback(level: str, current: int, total: int, message: str):
        try:
            progress_tracker = ui_components.get('progress_tracker')
            if not progress_tracker:
                return
            
            # Map level ke tracker method dengan percentage calculation
            if level in ['overall', 'step']:
                if hasattr(progress_tracker, 'update_overall'):
                    progress_tracker.update_overall(current, message)
            elif level in ['current', 'batch', 'file']:
                if hasattr(progress_tracker, 'update_current'):
                    progress_tracker.update_current(current, message)
            
            # 🎯 CRITICAL: Log milestone ke UI TANPA console logging untuk avoid double log
            if _is_milestone(current, total):
                from smartcash.ui.dataset.preprocessing.utils.ui_utils import log_to_accordion
                log_to_accordion(ui_components, f"🔄 {message} ({current}/{total})", "info")
                    
        except Exception:
            pass  # Silent fail to prevent breaking the process
    
    return progress_callback

def setup_progress_tracking(ui_components: Dict[str, Any], operation_name: str = "Processing"):
    """Setup progress tracking untuk operation dengan proper UI integration"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'show'):
            progress_tracker.show()
            if hasattr(progress_tracker, 'update_overall'):
                progress_tracker.update_overall(0, f"🚀 Memulai {operation_name.lower()}")
    except Exception:
        pass

def complete_progress_tracking(ui_components: Dict[str, Any], message: str):
    """Complete progress tracking dengan success state"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            if hasattr(progress_tracker, 'complete'):
                progress_tracker.complete(message)
            elif hasattr(progress_tracker, 'update_overall'):
                progress_tracker.update_overall(100, f"✅ {message}")
    except Exception:
        pass

def error_progress_tracking(ui_components: Dict[str, Any], error_msg: str):
    """Set error state pada progress tracking"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            if hasattr(progress_tracker, 'error'):
                progress_tracker.error(error_msg)
            elif hasattr(progress_tracker, 'update_overall'):
                progress_tracker.update_overall(0, f"❌ {error_msg}")
    except Exception:
        pass

def hide_progress_tracking(ui_components: Dict[str, Any]):
    """Hide progress tracking setelah operation selesai"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'hide'):
            progress_tracker.hide()
    except Exception:
        pass

def reset_progress_tracking(ui_components: Dict[str, Any]):
    """Reset progress tracking ke state awal"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'reset'):
            progress_tracker.reset()
    except Exception:
        pass

def update_progress_manual(ui_components: Dict[str, Any], level: str, progress: int, message: str):
    """Manual progress update untuk operations yang tidak memiliki callback"""
    try:
        progress_tracker = ui_components.get('progress_tracker')
        if not progress_tracker:
            return
        
        if level == 'overall' and hasattr(progress_tracker, 'update_overall'):
            progress_tracker.update_overall(progress, message)
        elif level == 'current' and hasattr(progress_tracker, 'update_current'):
            progress_tracker.update_current(progress, message)
        elif level == 'step' and hasattr(progress_tracker, 'update_step'):
            progress_tracker.update_step(progress, message)
    except Exception:
        pass

def _is_milestone(current: int, total: int) -> bool:
    """Check if progress adalah milestone yang perlu di-log"""
    if total <= 10:
        return True
    
    milestones = [0, 10, 25, 50, 75, 90, 100]
    progress_pct = (current / total) * 100 if total > 0 else 0
    return any(abs(progress_pct - milestone) < 1 for milestone in milestones) or current == total

def create_progress_manager(ui_components: Dict[str, Any]) -> Dict[str, Callable]:
    """Create progress manager dengan semua utility functions"""
    return {
        'setup': lambda name: setup_progress_tracking(ui_components, name),
        'complete': lambda msg: complete_progress_tracking(ui_components, msg),
        'error': lambda msg: error_progress_tracking(ui_components, msg),
        'hide': lambda: hide_progress_tracking(ui_components),
        'reset': lambda: reset_progress_tracking(ui_components),
        'update': lambda level, progress, msg: update_progress_manual(ui_components, level, progress, msg),
        'callback': lambda: create_dual_progress_callback(ui_components)
    }

# One-liner utilities
create_progress_callback = lambda ui_components: create_dual_progress_callback(ui_components)
setup_progress = lambda ui_components, name="Processing": setup_progress_tracking(ui_components, name)
complete_progress = lambda ui_components, msg="Completed": complete_progress_tracking(ui_components, msg)
error_progress = lambda ui_components, msg="Error": error_progress_tracking(ui_components, msg)
hide_progress = lambda ui_components: hide_progress_tracking(ui_components)
reset_progress = lambda ui_components: reset_progress_tracking(ui_components)

# Compatibility aliases untuk backward compatibility
def show_progress_for_operation(ui_components: Dict[str, Any], operation_name: str):
    """Compatibility alias untuk setup_progress_tracking"""
    setup_progress_tracking(ui_components, operation_name)

def update_progress_step(ui_components: Dict[str, Any], step: int, total_steps: int, message: str):
    """Update progress step dengan proper percentage calculation"""
    progress_pct = int((step / total_steps) * 100) if total_steps > 0 else 0
    update_progress_manual(ui_components, 'step', progress_pct, message)

def update_progress_current(ui_components: Dict[str, Any], current: int, total: int, message: str):
    """Update current operation progress dengan proper percentage calculation"""
    progress_pct = int((current / total) * 100) if total > 0 else 0
    update_progress_manual(ui_components, 'current', progress_pct, message)