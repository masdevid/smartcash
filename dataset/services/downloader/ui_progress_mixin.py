"""
File: smartcash/dataset/services/downloader/ui_progress_mixin.py
Deskripsi: Progress mixin untuk UI downloader dengan callback management yang robust
"""

from typing import Optional, Callable, Dict, Any
from smartcash.components.observer import notify, EventTopics
from smartcash.components.observer.base_observer import BaseObserver
from smartcash.components.observer.manager_observer import get_observer_manager

class UIProgressMixin(BaseObserver):
    """Mixin untuk progress tracking dalam UI downloader dengan callback management dan observer integration."""
    
    def __init__(self):
        """Initialize progress tracking state."""
        # Inisialisasi BaseObserver
        super().__init__(name="UIProgressMixin", priority=0)
        
        # Progress tracking
        self._progress_callback: Optional[Callable] = None
        self._current_step = "idle"
        self._step_progress = 0
        self._overall_progress = 0
        
        # Observer manager
        self._observer_manager = get_observer_manager()
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """
        Set callback untuk progress updates.
        
        Args:
            callback: Function dengan signature (step, current, total, message)
        """
        self._progress_callback = callback
    
    def _notify_progress(self, step: str, current: int, total: int, message: str = "") -> None:
        """
        Notify progress update via callback dan observer.
        
        Args:
            step: Step name ('download', 'extract', etc.)
            current: Current progress value
            total: Total progress value  
            message: Optional progress message
        """
        # Update internal state
        self._step_progress = current
        
        # Notify via callback
        if self._progress_callback:
            try:
                self._progress_callback(step, current, total, message)
            except Exception:
                # Silent fail untuk prevent callback errors dari mengganggu proses
                pass
        
        # Notify via observer
        try:
            # Hitung persentase jika total > 0
            percentage = int((current / total) * 100) if total > 0 else 0
            
            # Kirim event DOWNLOAD_PROGRESS
            self._send_observer_event('DOWNLOAD_PROGRESS', {
                'step': step,
                'progress': current,
                'total': total,
                'percentage': percentage,
                'message': message,
                'step_name': step
            })
        except Exception:
            # Silent fail untuk observer errors
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
        
        # Kirim event DOWNLOAD_START
        try:
            self._send_observer_event('DOWNLOAD_START', {
                'step': step,
                'message': message or f"Starting {step}",
                'progress': 0,
                'total_steps': 3,  # download, extract, validate
                'current_step': 1
            })
        except Exception:
            pass
            
        self._notify_progress(step, 0, 100, message or f"Starting {step}")
    
    def _notify_step_complete(self, step: str, message: str = "") -> None:
        """
        Notify step completion.
        
        Args:
            step: Step name
            message: Step completion message
        """
        self._step_progress = 100
        
        # Jika ini adalah step terakhir, kirim event DOWNLOAD_COMPLETE
        if step == "extract" or step == "validate":
            try:
                self._send_observer_event('DOWNLOAD_COMPLETE', {
                    'step': step,
                    'message': message or f"{step} completed",
                    'status': 'success'
                })
            except Exception:
                pass
                
        self._notify_progress(step, 100, 100, message or f"{step} completed")
    
    def _notify_step_error(self, step: str, error_message: str) -> None:
        """
        Notify step error.
        
        Args:
            step: Step name
            error_message: Error message
        """
        # Kirim event DOWNLOAD_ERROR
        try:
            self._send_observer_event('DOWNLOAD_ERROR', {
                'step': step,
                'message': error_message,
                'status': 'error'
            })
        except Exception:
            pass
            
        self._notify_progress(step, 0, 100, f"Error: {error_message}")
    
    def _send_observer_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Send observer event dengan multiple fallback methods."""
        import time
        data['timestamp'] = time.time()
        data['event_type'] = event_type
        
        try:
            if self._observer_manager and hasattr(self._observer_manager, 'notify'):
                self._observer_manager.notify(event_type, self, **data)
                return
        except Exception:
            pass
        
        try:
            from smartcash.components.observer import EventDispatcher
            EventDispatcher.notify(event_type, self, **data)
            return
        except Exception:
            pass
        
        try:
            notify(event_type, self, **data)
        except Exception:
            pass
    
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
            'has_callback': self._progress_callback is not None,
            'has_observer_manager': self._observer_manager is not None
        }
        
    def update(self, event_type: str, sender: Any, **kwargs) -> None:
        """Implementasi metode update dari BaseObserver."""
        # Metode ini diperlukan untuk implementasi BaseObserver
        # Kita tidak perlu implementasi khusus karena UIProgressMixin adalah pengirim event, bukan penerima
        pass
    
    def should_process_event(self, event_type: str) -> bool:
        """Implementasi metode should_process_event dari BaseObserver."""
        # Selalu return True karena kita tidak memfilter event
        return True