"""
File: smartcash/ui/dataset/preprocessing/utils/progress_utils.py
Deskripsi: Simplified progress utilities untuk dual progress tracker
"""

from typing import Dict, Any, Callable

def create_dual_progress_callback(ui_components: Dict[str, Any]) -> Callable[[str, int, int, str], None]:
    """Create dual progress callback"""
    def progress_callback(step: str, current: int, total: int, message: str):
        try:
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker:
                # Overall progress
                overall_progress = min(100, (current / max(total, 1)) * 100)
                progress_tracker.update_overall(int(overall_progress), message)
                
                # Current operation progress
                current_progress = min(100, (current / max(total, 1)) * 100)
                progress_tracker.update_current(int(current_progress), f"Processing: {current}/{total}")
            
            # Log milestone saja
            if current % max(1, total // 10) == 0 or current == total:
                logger = ui_components.get('logger')
                if logger:
                    logger.info(f"🔄 {message} ({current}/{total})")
                    
        except Exception:
            pass
    
    return progress_callback

def setup_dual_progress_tracker(ui_components: Dict[str, Any], operation_name: str = "Dataset Preprocessing"):
    """Setup dual progress tracker"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        progress_tracker.show(operation_name)
        progress_tracker.update_overall(0, f"🚀 Memulai {operation_name.lower()}...")
    
    logger = ui_components.get('logger')
    if logger:
        logger.info(f"🚀 Memulai {operation_name.lower()}")

def complete_progress_tracker(ui_components: Dict[str, Any], message: str):
    """Complete dual progress tracker"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker and hasattr(progress_tracker, 'complete'):
        progress_tracker.complete(message)
    
    logger = ui_components.get('logger')
    if logger:
        logger.success(f"✅ {message}")

def error_progress_tracker(ui_components: Dict[str, Any], error_msg: str):
    """Set error state pada dual progress tracker"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker and hasattr(progress_tracker, 'error'):
        progress_tracker.error(error_msg)
    
    logger = ui_components.get('logger')
    if logger:
        logger.error(f"❌ {error_msg}")

def reset_progress_tracker(ui_components: Dict[str, Any]):
    """Reset dual progress tracker"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker and hasattr(progress_tracker, 'reset'):
        progress_tracker.reset()
    
    logger = ui_components.get('logger')
    if logger:
        logger.info("🔄 Progress tracker direset")