"""
File: smartcash/model/utils/progress_utils.py
Deskripsi: Utilitas untuk tracking progress operasi model
"""

from typing import Dict, Any, Optional, Callable, Union

class ProgressTracker:
    """Kelas untuk tracking progress operasi dengan callback support"""
    
    def __init__(self, callback: Optional[Callable] = None):
        """Initialize progress tracker dengan optional callback"""
        self._progress = {'current': 0, 'total': 0, 'message': ''}
        self._callback = callback
    
    def update(self, current: int, total: int, message: str) -> None:
        """Update progress tracking dan trigger callback"""
        self._progress = {'current': current, 'total': total, 'message': message}
        self._notify_callback()
    
    def _notify_callback(self) -> None:
        """Notify callback dengan progress saat ini"""
        if not self._callback: return
        try:
            self._callback(self._progress['current'], self._progress['total'], self._progress['message'])
        except Exception:
            pass  # Silent fail untuk callback
    
    def set_callback(self, callback: Callable) -> None:
        """Set atau update callback function"""
        self._callback = callback
        # Trigger callback dengan current progress
        self._notify_callback()
    
    def reset(self) -> None:
        """Reset progress ke nilai awal"""
        self._progress = {'current': 0, 'total': 0, 'message': ''}
    
    @property
    def progress(self) -> Dict[str, Any]:
        """Get current progress state"""
        return self._progress
    
    @property
    def percentage(self) -> float:
        """Calculate percentage completion"""
        return 0.0 if self._progress['total'] == 0 else (self._progress['current'] / self._progress['total']) * 100
    
    @property
    def is_complete(self) -> bool:
        """Check if operation is complete"""
        return self._progress['current'] >= self._progress['total'] and self._progress['total'] > 0
    
    # One-liner utilities
    get_message = lambda self: self._progress.get('message', '')
    get_current = lambda self: self._progress.get('current', 0)
    get_total = lambda self: self._progress.get('total', 0)


# Fungsi utilitas untuk progress tracking
def create_progress_tracker(callback: Optional[Callable] = None) -> ProgressTracker:
    """Create dan return instance ProgressTracker"""
    return ProgressTracker(callback)

def update_progress_safe(tracker: Optional[ProgressTracker], current: int, total: int, message: str) -> None:
    """Update progress dengan safe handling jika tracker None"""
    if tracker: tracker.update(current, total, message)


def download_with_progress(url: str, destination: str, progress_callback: Optional[Callable] = None) -> bool:
    """
    Download file with progress tracking.
    
    Args:
        url: URL to download from
        destination: Local file path to save to
        progress_callback: Optional callback for progress updates
        
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        import urllib.request
        import os
        from pathlib import Path
        
        # Create destination directory if needed
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        
        def report_progress(block_num, block_size, total_size):
            if progress_callback and total_size > 0:
                downloaded = block_num * block_size
                percentage = min(100, (downloaded / total_size) * 100)
                progress_callback(int(percentage), f"Downloading {os.path.basename(destination)}")
        
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        return True
        
    except Exception as e:
        print(f"Download failed: {e}")
        return False
