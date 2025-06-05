"""
File: smartcash/ui/setup/env_config/components/progress_tracking.py
Deskripsi: Progress tracking adapter untuk menggunakan progress_tracker.py baru

[DEPRECATED] File ini dipertahankan untuk backward compatibility.
Gunakan smartcash.ui.components.progress_tracker secara langsung untuk implementasi baru.
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

# Import progress tracker baru
from smartcash.ui.components.progress_tracker import create_single_progress_tracker

def reset_progress(ui_components: Dict[str, Any], message: str = "") -> None:
    """Reset progress bar ke 0"""
    progress_tracker = ui_components.get('progress_tracker')
    
    if progress_tracker:
        # Gunakan progress tracker baru
        progress_tracker.reset()
    else:
        # Fallback ke implementasi lama
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].description = "0%"
        
        if 'progress_message' in ui_components:
            ui_components['progress_message'].value = message or ""
        
        # Show progress container pada reset
        if 'progress_container' in ui_components:
            ui_components['progress_container'].layout.visibility = 'visible'

def update_progress(ui_components: Dict[str, Any], current: int, total: int, message: str = "") -> None:
    """Update progress bar dengan nilai baru"""
    progress_tracker = ui_components.get('progress_tracker')
    
    if progress_tracker:
        # Gunakan progress tracker baru
        percentage = min(100, max(0, int((current / total) * 100)))
        progress_tracker.update('level1', percentage, message)
    else:
        # Fallback ke implementasi lama
        if 'progress_bar' in ui_components:
            percentage = min(100, max(0, int((current / total) * 100)))
            ui_components['progress_bar'].value = percentage
            ui_components['progress_bar'].description = f"{percentage}%"
        
        if 'progress_message' in ui_components and message:
            ui_components['progress_message'].value = message

def hide_progress(ui_components: Dict[str, Any]) -> None:
    """Hide progress container setelah selesai"""
    progress_tracker = ui_components.get('progress_tracker')
    
    if progress_tracker:
        # Gunakan progress tracker baru
        progress_tracker.reset()
    else:
        # Fallback ke implementasi lama
        if 'progress_container' in ui_components:
            ui_components['progress_container'].layout.visibility = 'hidden'

def create_progress_tracking(module_name: str = "progress", width: str = "100%") -> Dict[str, Any]:
    """Create progress tracking components menggunakan progress tracker baru"""
    # Gunakan progress tracker baru dengan single level
    progress_components = create_single_progress_tracker()
    
    # Tambahkan referensi ke progress_tracker untuk digunakan oleh fungsi lain
    result = {
        'progress_tracker': progress_components.get('progress_tracker'),
        'progress_container': progress_components.get('container'),
        'reset_progress': progress_components.get('reset_all'),
        'update_progress': progress_components.get('update_progress'),
        'complete_progress': progress_components.get('complete_operation'),
        'error_progress': progress_components.get('error_operation')
    }
    
    # Untuk backward compatibility
    result['progress_bar'] = progress_components.get('progress_bars', {}).get('level1')
    result['progress_message'] = progress_components.get('messages', {}).get('level1')
    
    return result