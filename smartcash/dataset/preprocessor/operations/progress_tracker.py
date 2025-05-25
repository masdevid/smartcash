"""
File: smartcash/dataset/preprocessor/operations/progress_tracker.py
Deskripsi: Unified progress tracking system untuk semua operasi preprocessing
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from smartcash.common.logger import get_logger


class ProgressTracker:
    """Unified progress tracker untuk koordinasi multi-level progress reporting."""
    
    def __init__(self, logger=None):
        """Initialize progress tracker dengan callback management."""
        self.logger = logger or get_logger()
        self._callbacks: List[Callable] = []
        self._progress_state = {'overall': 0, 'step': 0, 'current': 0}
        self._operation_metadata = {}
        self._active_operations = set()
        
    def register_callback(self, callback: Callable) -> None:
        """Register progress callback untuk UI notifications."""
        if callback and callable(callback) and callback not in self._callbacks:
            self._callbacks.append(callback)
            self.logger.debug(f"📡 Progress callback registered: {len(self._callbacks)} total")
    
    def setup_progress_tracking(self, operation: str, estimated_total: int = 100, **metadata) -> None:
        """Setup progress tracking untuk operation baru."""
        self._active_operations.add(operation)
        self._operation_metadata[operation] = {
            'start_time': datetime.now(),
            'estimated_total': estimated_total,
            'metadata': metadata
        }
        self._reset_progress_state()
        self.logger.debug(f"🎯 Progress tracking setup untuk: {operation}")
    
    def update_multi_level_progress(self, **kwargs) -> None:
        """
        Update multi-level progress dengan flexible parameters.
        
        Args:
            **kwargs: Progress parameters (overall_progress, step, current_progress, etc.)
        """
        # Extract dan normalize progress values
        overall = self._normalize_progress(kwargs.get('overall_progress', kwargs.get('progress', 0)))
        step = kwargs.get('step', 0)
        current = self._normalize_progress(kwargs.get('current_progress', 0))
        
        # Update internal state jika ada perubahan
        state_changed = (
            overall != self._progress_state['overall'] or
            step != self._progress_state['step'] or
            current != self._progress_state['current']
        )
        
        if state_changed:
            self._progress_state.update({'overall': overall, 'step': step, 'current': current})
            
            # Prepare notification parameters
            notification_params = {
                'progress': overall,
                'total': 100,
                'message': kwargs.get('message', 'Processing...'),
                'status': kwargs.get('status', 'info'),
                'step': step,
                'split_step': kwargs.get('split_step', kwargs.get('split', '')),
                'current_progress': current,
                'current_total': kwargs.get('current_total', 100),
                **{k: v for k, v in kwargs.items() if k not in [
                    'overall_progress', 'progress', 'current_progress', 'message', 'status'
                ]}
            }
            
            self.notify_progress_callbacks(**notification_params)
    
    def notify_progress_callbacks(self, **kwargs) -> None:
        """Notify semua registered callbacks dengan error handling."""
        if not self._callbacks:
            return
        
        failed_callbacks = []
        for callback in self._callbacks:
            try:
                callback(**kwargs)
            except Exception as e:
                failed_callbacks.append(callback)
                self.logger.debug(f"🔧 Progress callback error: {str(e)}")
        
        # Remove failed callbacks
        for failed_callback in failed_callbacks:
            self._callbacks.remove(failed_callback)
    
    def complete_progress_tracking(self, operation: str, message: str = "Operation completed") -> None:
        """Complete progress tracking untuk operation."""
        if operation in self._active_operations:
            self._active_operations.remove(operation)
            
            # Calculate operation duration
            if operation in self._operation_metadata:
                start_time = self._operation_metadata[operation]['start_time']
                duration = (datetime.now() - start_time).total_seconds()
                completion_message = f"{message} (selesai dalam {duration:.1f} detik)"
            else:
                completion_message = message
            
            # Notify completion
            self.update_multi_level_progress(
                overall_progress=100, message=completion_message, 
                status='success', step=3
            )
            
            self.logger.success(f"✅ Progress tracking completed untuk: {operation}")
    
    def error_progress_tracking(self, operation: str, error_message: str) -> None:
        """Set error state untuk progress tracking."""
        if operation in self._active_operations:
            self._active_operations.remove(operation)
            
            self.update_multi_level_progress(
                overall_progress=0, message=f"Error: {error_message}", 
                status='error', step=0
            )
            
            self.logger.error(f"❌ Progress tracking error untuk {operation}: {error_message}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Dapatkan summary current progress state."""
        return {
            'active_operations': list(self._active_operations),
            'registered_callbacks': len(self._callbacks),
            'current_progress': self._progress_state.copy(),
            'operation_metadata': {
                op: {
                    'duration_seconds': (datetime.now() - meta['start_time']).total_seconds(),
                    **meta['metadata']
                } for op, meta in self._operation_metadata.items()
            }
        }
    
    def reset_progress_state(self) -> None:
        """Reset progress state untuk operation baru."""
        self._reset_progress_state()
        self._active_operations.clear()
        self._operation_metadata.clear()
        self.logger.debug("🔄 Progress state reset")
    
    def _normalize_progress(self, progress: int) -> int:
        """Normalize progress value ke range 0-100."""
        return max(0, min(100, int(progress) if progress is not None else 0))
    
    def _reset_progress_state(self) -> None:
        """Internal method untuk reset progress state."""
        self._progress_state = {'overall': 0, 'step': 0, 'current': 0}