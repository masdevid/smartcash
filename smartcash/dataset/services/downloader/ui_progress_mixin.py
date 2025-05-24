"""
File: smartcash/dataset/services/downloader/ui_progress_mixin.py
Deskripsi: Progress mixin untuk UI downloader dengan callback management yang robust
"""

from typing import Optional, Callable, Dict, Any

class UIProgressMixin:
    """Mixin untuk progress tracking dalam UI downloader dengan callback management."""
    
    def __init__(self):
        """Initialize progress tracking state."""
        self._progress_callback: Optional[Callable] = None
        self._current_step = "idle"
        self._step_progress = 0
        self._overall_progress = 0
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """
        Set callback untuk progress updates.
        
        Args:
            callback: Function dengan signature (step, current, total, message)
        """
        self._progress_callback = callback
    
    def _notify_progress(self, step: str, current: int, total: int, message: str = "") -> None:
        """
        Notify progress update via callback.
        
        Args:
            step: Step name ('download', 'extract', etc.)
            current: Current progress value
            total: Total progress value  
            message: Optional progress message
        """
        if self._progress_callback:
            try:
                self._progress_callback(step, current, total, message)
                self._step_progress = current
            except Exception:
                # Silent fail untuk prevent callback errors dari mengganggu proses
                pass
    
    def _notify_step_start(self, step: str, message: str = "") -> None:
        """
        Notify step start.
        
        Args:
            step: Step name
            message: Step start message
        """
        self._current_step = step
        self._step_progress = 0
        self._notify_progress(step, 0, 100, message or f"Starting {step}")
    
    def _notify_step_complete(self, step: str, message: str = "") -> None:
        """
        Notify step completion.
        
        Args:
            step: Step name
            message: Step completion message
        """
        self._step_progress = 100
        self._notify_progress(step, 100, 100, message or f"{step} completed")
    
    def _notify_step_error(self, step: str, error_message: str) -> None:
        """
        Notify step error.
        
        Args:
            step: Step name
            error_message: Error message
        """
        self._notify_progress(step, 0, 100, f"Error: {error_message}")
    
    def get_current_progress(self) -> Dict[str, Any]:
        """
        Get current progress state.
        
        Returns:
            Dictionary dengan progress information
        """
        return {
            'current_step': self._current_step,
            'step_progress': self._step_progress,
            'overall_progress': self._overall_progress,
            'has_callback': self._progress_callback is not None
        }