"""
File: smartcash/ui/dataset/download/services/progress_bridge.py
Deskripsi: Bridge untuk mengirim progress dari service ke UI observer
"""

from typing import Dict, Any, Optional
from smartcash.components.observer import notify, EventTopics

class ProgressBridge:
    """Bridge untuk komunikasi progress dari service ke UI."""
    
    def __init__(self, observer_manager=None, namespace: str = "download"):
        self.observer_manager = observer_manager
        self.namespace = namespace
    
    def notify_progress(self, progress: int, total: int = 100, message: str = "", 
                       step: Optional[str] = None, current_step: int = 1, total_steps: int = 5) -> None:
        """Notify progress update."""
        percentage = int((progress / total) * 100) if total > 0 else 0
        
        event_data = {
            'progress': percentage,
            'message': message,
            'namespace': self.namespace
        }
        
        # Overall progress
        self._send_event('DOWNLOAD_PROGRESS', event_data)
        
        # Step progress jika ada
        if step:
            step_data = {
                'step_progress': percentage,
                'step_message': f"{step}: {message}",
                'current_step': current_step,
                'total_steps': total_steps,
                'namespace': self.namespace
            }
            self._send_event('DOWNLOAD_STEP_PROGRESS', step_data)
    
    def notify_start(self, message: str = "Memulai proses", total_steps: int = 5) -> None:
        """Notify process start."""
        self._send_event('DOWNLOAD_START', {
            'message': message,
            'total_steps': total_steps,
            'namespace': self.namespace
        })
    
    def notify_complete(self, message: str = "Proses selesai", duration: float = 0) -> None:
        """Notify process complete."""
        self._send_event('DOWNLOAD_COMPLETE', {
            'message': message,
            'duration': duration,
            'namespace': self.namespace
        })
    
    def notify_error(self, message: str = "Terjadi error") -> None:
        """Notify error."""
        self._send_event('DOWNLOAD_ERROR', {
            'message': message,
            'namespace': self.namespace
        })
    
    def _send_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Send event via observer atau notify."""
        try:
            if self.observer_manager:
                self.observer_manager.notify(event_type, self, **data)
            else:
                # Fallback ke global notify
                notify(event_type, self, **data)
        except Exception:
            # Ignore notification errors
            pass