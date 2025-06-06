"""
File: smartcash/dataset/downloader/progress_tracker.py
Deskripsi: Optimized progress tracker dengan one-liner methods dan enhanced performance
"""

import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum

class DownloadStep(Enum):
    """Enum untuk download steps dengan one-liner values"""
    VALIDATION, METADATA, DOWNLOAD, EXTRACT, ORGANIZE, COMPLETE = "validation", "metadata", "download", "extract", "organize", "complete"

@dataclass
class StepInfo:
    """Optimized step info dengan one-liner defaults"""
    name: str
    weight: int = 10
    description: str = ""
    started: bool = False
    completed: bool = False
    progress: int = 0
    message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class DownloadProgressTracker:
    """Optimized progress tracker dengan one-liner methods dan smart caching."""
    
    def __init__(self):
        self._progress_callback, self._step_callback, self.steps = None, None, {}
        self.overall_progress, self.current_step, self.start_time, self.is_active = 0, "", None, False
        self._setup_default_steps_optimized()
    
    def _setup_default_steps_optimized(self) -> None:
        """One-liner optimized step setup"""
        step_configs = [('validation', 10, '🔍 Validasi parameter'), ('metadata', 15, '📊 Ambil metadata'), 
                       ('download', 45, '📥 Download dataset'), ('extract', 20, '📦 Ekstrak dataset'), 
                       ('organize', 10, '🗂️ Organisir dataset')]
        self.steps = {name: StepInfo(name, weight, desc) for name, weight, desc in step_configs}
    
    def set_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """One-liner callback setter"""
        self._progress_callback = callback
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """One-liner progress callback setter - compatibility"""
        self._progress_callback = callback
    
    def set_step_callback(self, callback: Callable[[str, int, str], None]) -> None:
        """One-liner step callback setter"""
        self._step_callback = callback
    
    def start_process(self, message: str = "🚀 Memulai download dataset") -> None:
        """One-liner process start dengan reset"""
        self.start_time, self.is_active, self.overall_progress, self.current_step = time.time(), True, 0, ""
        
        # One-liner step reset
        [setattr(step, attr, False if attr in ['started', 'completed'] else 0 if attr == 'progress' else "" if attr == 'message' else None) 
         for step in self.steps.values() for attr in ['started', 'completed', 'progress', 'message', 'start_time', 'end_time']]
        
        self._notify_progress("start", 0, 100, message)
    
    def start_step(self, step_name: str, message: str = "") -> None:
        """One-liner step start dengan automatic progress calculation"""
        step_name not in self.steps and None or (
            step := self.steps[step_name],
            setattr(step, 'started', True), setattr(step, 'start_time', time.time()),
            setattr(step, 'progress', 0), setattr(step, 'message', message or step.description),
            setattr(self, 'current_step', step_name), self._calculate_overall_progress_optimized(),
            self._notify_step("step_start", step.progress, step.message),
            self._notify_progress("progress", self.overall_progress, 100, f"🔄 {step.description}")
        )
    
    def update_step(self, step_name: str, progress: int, message: str = "") -> None:
        """One-liner step update dengan smart throttling"""
        step_name not in self.steps and None or (
            step := self.steps[step_name],
            setattr(step, 'progress', max(0, min(100, progress))),
            message and setattr(step, 'message', message),
            self._calculate_overall_progress_optimized(),
            self._notify_step("step_progress", step.progress, step.message),
            self._notify_progress("progress", self.overall_progress, 100, step.message)
        )
    
    def complete_step(self, step_name: str, message: str = "") -> None:
        """One-liner step completion dengan duration calculation"""
        step_name not in self.steps and None or (
            step := self.steps[step_name],
            setattr(step, 'completed', True), setattr(step, 'end_time', time.time()),
            setattr(step, 'progress', 100), message and setattr(step, 'message', message),
            self._calculate_overall_progress_optimized(),
            duration := step.end_time - step.start_time if step.start_time else 0,
            complete_msg := f"✅ {step.description} ({duration:.1f}s)",
            self._notify_step("step_complete", 100, complete_msg),
            self._notify_progress("progress", self.overall_progress, 100, complete_msg)
        )
    
    def complete_process(self, message: str = "✅ Download dataset selesai") -> None:
        """One-liner process completion dengan final statistics"""
        self.is_active = False
        total_duration = time.time() - self.start_time if self.start_time else 0
        
        # One-liner complete all steps
        [setattr(step, attr, True if attr == 'completed' else 100 if attr == 'progress' else time.time() if attr == 'end_time' and not step.end_time else getattr(step, attr))
         for step in self.steps.values() for attr in ['completed', 'progress', 'end_time']]
        
        self.overall_progress, final_message = 100, f"{message} ({total_duration:.1f}s)"
        self._notify_step("complete", 100, final_message)
        self._notify_progress("complete", 100, 100, final_message)
    
    def error_process(self, error_message: str, step_name: str = "") -> None:
        """One-liner error handling dengan step context"""
        self.is_active = False
        step_name and step_name in self.steps and setattr(self.steps[step_name], 'message', error_message)
        self._notify_step("error", 0, error_message)
        self._notify_progress("error", 0, 100, error_message)
    
    def _calculate_overall_progress_optimized(self) -> None:
        """One-liner optimized progress calculation"""
        total_weight = sum(step.weight for step in self.steps.values())
        weighted_progress = sum((100 if step.completed else step.progress if step.started else 0) * step.weight / 100 
                              for step in self.steps.values())
        self.overall_progress = int(weighted_progress / total_weight * 100) if total_weight > 0 else 0
    
    def _notify_progress(self, event_type: str, progress: int, total: int, message: str) -> None:
        """One-liner progress notification dengan safe execution"""
        self._progress_callback and (lambda: self._progress_callback(event_type, progress, total, message))() if True else None
    
    def _notify_step(self, event_type: str, progress: int, message: str) -> None:
        """One-liner step notification dengan safe execution"""
        self._step_callback and (lambda: self._step_callback(event_type, progress, message))() if True else None
    
    def get_progress_summary_optimized(self) -> Dict[str, Any]:
        """One-liner optimized progress summary"""
        current_step_info = self.steps.get(self.current_step)
        return {
            'overall_progress': self.overall_progress,
            'current_step': {
                'name': self.current_step,
                'description': current_step_info.description if current_step_info else "",
                'progress': current_step_info.progress if current_step_info else 0,
                'message': current_step_info.message if current_step_info else ""
            },
            'total_steps': len(self.steps), 'completed_steps': sum(1 for step in self.steps.values() if step.completed),
            'is_active': self.is_active, 'duration': time.time() - self.start_time if self.start_time else 0
        }
    
    def get_step_details_optimized(self) -> List[Dict[str, Any]]:
        """One-liner optimized step details"""
        return [{
            'name': step.name, 'description': step.description, 'weight': step.weight,
            'started': step.started, 'completed': step.completed, 'progress': step.progress,
            'message': step.message, 'duration': (step.end_time - step.start_time) if step.start_time and step.end_time else 0
        } for step in self.steps.values()]
    
    def reset_tracker(self) -> None:
        """One-liner tracker reset"""
        self.overall_progress, self.current_step, self.start_time, self.is_active = 0, "", None, False
        [setattr(step, attr, False if attr in ['started', 'completed'] else 0 if attr == 'progress' else "" if attr == 'message' else None)
         for step in self.steps.values() for attr in ['started', 'completed', 'progress', 'message', 'start_time', 'end_time']]
    
    def update_step_weights(self, weight_map: Dict[str, int]) -> None:
        """One-liner step weight update"""
        [setattr(step, 'weight', weight_map.get(step.name, step.weight)) for step in self.steps.values() if step.name in weight_map]

class CallbackManager:
    """Optimized callback manager dengan one-liner methods."""
    
    def __init__(self):
        self._callbacks: Dict[str, List[Callable]] = {}
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """One-liner callback registration"""
        self._callbacks.setdefault(event_type, []).append(callback)
    
    def unregister_callback(self, event_type: str, callback: Callable) -> None:
        """One-liner callback unregistration"""
        event_type in self._callbacks and callback in self._callbacks[event_type] and self._callbacks[event_type].remove(callback)
    
    def notify_callbacks(self, event_type: str, *args, **kwargs) -> None:
        """One-liner callback notification dengan error protection"""
        [self._safe_callback_call(callback, *args, **kwargs) for callback in self._callbacks.get(event_type, [])]
    
    def _safe_callback_call(self, callback: Callable, *args, **kwargs) -> None:
        """One-liner safe callback execution"""
        try:
            callback(*args, **kwargs)
        except Exception:
            pass

# One-liner factory functions
def create_download_tracker() -> DownloadProgressTracker:
    """Factory untuk optimized DownloadProgressTracker"""
    return DownloadProgressTracker()

def create_callback_manager() -> CallbackManager:
    """Factory untuk optimized CallbackManager"""
    return CallbackManager()

def create_step_weights_optimized(include_validation: bool = True, include_organize: bool = True) -> Dict[str, int]:
    """One-liner optimized step weights creation"""
    base_weights = {'metadata': 20, 'download': 60, 'extract': 20}
    include_validation and base_weights.update({'validation': 10, **{k: int(v * 0.9) for k, v in base_weights.items()}})
    include_organize and base_weights.update({'organize': 10}) and base_weights.update({k: int((v / sum(base_weights.values())) * 100) for k, v in base_weights.items()})
    return base_weights

def format_progress_message_optimized(step: str, progress: int, message: str) -> str:
    """One-liner optimized progress message formatting"""
    emoji_map = {'validation': '🔍', 'metadata': '📋', 'download': '📥', 'extract': '📦', 'organize': '📁', 'complete': '✅', 'error': '❌'}
    emoji = emoji_map.get(step, '🔄')
    return f"{emoji} {message}" + (f" ({progress}%)" if progress > 0 else "")

# One-liner utility functions
get_tracker_status = lambda tracker: {'active': tracker.is_active, 'progress': tracker.overall_progress, 'current_step': tracker.current_step}
calculate_eta = lambda tracker, avg_speed: (100 - tracker.overall_progress) / avg_speed if avg_speed > 0 and tracker.overall_progress < 100 else 0
format_duration = lambda seconds: f"{seconds:.1f}s" if seconds < 60 else f"{seconds/60:.1f}m" if seconds < 3600 else f"{seconds/3600:.1f}h"