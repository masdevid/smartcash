"""
File: smartcash/dataset/utils/progress_wrapper.py
Deskripsi: Wrapper untuk progress tracking dari common untuk memastikan backward compatibility dan DRY
"""

from typing import Dict, Any, List, Optional, Callable, Tuple

from smartcash.common.progress import ProgressTracker, get_progress_tracker
from smartcash.common.progress import (
    ProgressObserver, ProgressEventEmitter, create_progress_tracker_observer,
    update_progress as common_update_progress
)

# Re-export fungsi dan kelas dari common untuk backward compatibility
__all__ = [
    'ProgressTracker',
    'ProgressObserver', 
    'ProgressEventEmitter',
    'get_tracker',
    'create_tracker_with_observer',
    'update_progress'
]

def get_tracker(
    name: str, 
    total: int = 0, 
    desc: str = "", 
    display: bool = True
) -> ProgressTracker:
    """
    Wrapper untuk get_progress_tracker.
    
    Args:
        name: Nama untuk tracker
        total: Total nilai progres
        desc: Deskripsi progres
        display: Apakah menampilkan progress bar
        
    Returns:
        Instance ProgressTracker
    """
    return get_progress_tracker(name, total, desc, display=display)

def create_tracker_with_observer(
    name: str,
    total: int = 100,
    desc: str = "Progress",
    display: bool = True
) -> Tuple[ProgressTracker, ProgressObserver]:
    """
    Wrapper untuk create_progress_tracker_observer.
    
    Args:
        name: Nama untuk tracker
        total: Total nilai progres
        desc: Deskripsi progres
        display: Apakah menampilkan progress bar
        
    Returns:
        Tuple (ProgressTracker, ProgressObserver)
    """
    return create_progress_tracker_observer(name, total, desc, display)

def update_progress(
    callback: Callable, 
    current: int, 
    total: int, 
    message: Optional[str] = None, 
    status: str = 'info', 
    **kwargs
) -> None:
    """
    Wrapper untuk update_progress.
    
    Args:
        callback: Callback function untuk progress reporting
        current: Nilai progress saat ini
        total: Nilai total progress
        message: Pesan progress
        status: Status progress ('info', 'success', 'warning', 'error')
        **kwargs: Parameter tambahan untuk callback
    """
    common_update_progress(callback, current, total, message, status, **kwargs)