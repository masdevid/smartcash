"""
File: smartcash/dataset/services/downloader/ui_progress_mixin.py
Deskripsi: Mixin untuk menambahkan progress callback ke download service
"""

from typing import Callable, Optional

class UIProgressMixin:
    """Mixin untuk progress callback ke UI."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """
        Set callback untuk progress updates.
        
        Args:
            callback: Function(step, current, total, message) -> None
        """
        self._progress_callback = callback
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """Notify progress via callback."""
        if self._progress_callback:
            try:
                self._progress_callback(step, current, total, message)
            except Exception:
                pass  # Ignore callback errors