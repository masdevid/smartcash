"""
File: smartcash/ui/dataset/preprocessing/utils/progress_utils.py
Deskripsi: Fixed progress utilities dengan proper UI integration tanpa double logging
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