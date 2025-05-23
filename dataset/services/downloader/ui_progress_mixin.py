"""
File: smartcash/dataset/services/downloader/ui_progress_mixin.py
Deskripsi: Mixin class untuk progress callback integration yang konsisten di semua downloader services
"""

from typing import Optional, Callable

class UIProgressMixin:
    """Mixin untuk standardisasi progress callback di semua downloader services."""
    
    def __init__(self):
        self._progress_callback: Optional[Callable] = None
        self._current_step = ""
        self._step_total = 100
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """
        Set progress callback function.
        
        Args:
            callback: Function dengan signature (step, current, total, message)
        """
        self._progress_callback = callback
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """
        Notify progress via callback dengan error handling.
        
        Args:
            step: Nama step (download, extract, validate, etc)
            current: Progress saat ini
            total: Total progress
            message: Pesan progress
        """
        if self._progress_callback:
            try:
                # Clamp values untuk safety
                current = max(0, min(total, current))
                total = max(1, total)
                
                # Store current step untuk reference
                self._current_step = step
                self._step_total = total
                
                # Call callback
                self._progress_callback(step, current, total, message)
            except Exception:
                # Ignore callback errors agar tidak mengganggu proses utama
                pass
    
    def _notify_step_start(self, step: str, message: str = "") -> None:
        """Notify step start."""
        self._notify_progress(step, 0, 100, message or f"Memulai {step}")
    
    def _notify_step_complete(self, step: str, message: str = "") -> None:
        """Notify step completion."""
        self._notify_progress(step, 100, 100, message or f"{step} selesai")
    
    def _notify_step_error(self, step: str, message: str = "") -> None:
        """Notify step error."""
        self._notify_progress(step, 0, 100, f"Error: {message}")
    
    def _calculate_overall_progress(self, step_name: str, step_progress: int, 
                                   total_steps: int, current_step: int) -> int:
        """
        Calculate overall progress dari step progress.
        
        Args:
            step_name: Nama step saat ini
            step_progress: Progress dalam step (0-100)
            total_steps: Total jumlah steps
            current_step: Step ke berapa saat ini (1-based)
            
        Returns:
            Overall progress (0-100)
        """
        if total_steps <= 0:
            return step_progress
            
        # Calculate base progress from completed steps
        completed_steps = max(0, current_step - 1)
        base_progress = (completed_steps / total_steps) * 100
        
        # Add current step progress
        current_step_contribution = (step_progress / 100) * (100 / total_steps)
        
        overall = int(base_progress + current_step_contribution)
        return max(0, min(100, overall))
    
    def get_progress_status(self) -> dict:
        """Get current progress status untuk debugging."""
        return {
            'has_callback': self._progress_callback is not None,
            'current_step': self._current_step,
            'step_total': self._step_total
        }