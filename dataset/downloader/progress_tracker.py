"""
File: smartcash/dataset/downloader/progress_tracker.py
Deskripsi: Progress tracker dengan callback management untuk download process
"""

import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum

class DownloadStep(Enum):
    """Enum untuk step download process."""
    VALIDATION = "validation"
    METADATA = "metadata"
    DOWNLOAD = "download"
    EXTRACT = "extract"
    ORGANIZE = "organize"
    COMPLETE = "complete"

@dataclass
class StepInfo:
    """Info untuk setiap step download."""
    name: str
    weight: int  # Percentage weight dalam overall progress
    description: str
    started: bool = False
    completed: bool = False
    progress: int = 0
    message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class DownloadProgressTracker:
    """Progress tracker untuk download process dengan callback management."""
    
    def __init__(self):
        self._progress_callback: Optional[Callable] = None
        self._step_callback: Optional[Callable] = None
        self.steps: Dict[str, StepInfo] = {}
        self.overall_progress = 0
        self.current_step = ""
        self.start_time = None
        self.is_active = False
        
        self._setup_default_steps()
    
    def _setup_default_steps(self) -> None:
        """Setup default steps untuk download process."""
        self.steps = {
            'validation': StepInfo('validation', 10, 'Validasi parameter'),
            'metadata': StepInfo('metadata', 15, 'Ambil metadata dataset'),
            'download': StepInfo('download', 45, 'Download dataset'),
            'extract': StepInfo('extract', 20, 'Ekstrak dataset'),
            'organize': StepInfo('organize', 10, 'Organisir dataset')
        }
    
    def set_progress_callback(self, callback: Callable[[str, int, str], None]) -> None:
        """Set callback untuk overall progress updates."""
        self._progress_callback = callback
    
    def set_step_callback(self, callback: Callable[[str, int, str], None]) -> None:
        """Set callback untuk step progress updates."""
        self._step_callback = callback
    
    def start_process(self, message: str = "Memulai download dataset") -> None:
        """Start download process tracking."""
        self.start_time = time.time()
        self.is_active = True
        self.overall_progress = 0
        self.current_step = ""
        
        # Reset all steps
        for step in self.steps.values():
            step.started = step.completed = False
            step.progress = 0
            step.message = ""
            step.start_time = step.end_time = None
        
        self._notify_progress("start", 0, message)
    
    def start_step(self, step_name: str, message: str = "") -> None:
        """Start specific step."""
        if step_name not in self.steps:
            return
        
        step = self.steps[step_name]
        step.started = True
        step.start_time = time.time()
        step.progress = 0
        step.message = message or step.description
        
        self.current_step = step_name
        self._calculate_overall_progress()
        
        self._notify_step("step_start", step.progress, step.message)
        self._notify_progress("progress", self.overall_progress, f"Memulai {step.description}")
    
    def update_step(self, step_name: str, progress: int, message: str = "") -> None:
        """Update progress untuk specific step."""
        if step_name not in self.steps:
            return
        
        step = self.steps[step_name]
        step.progress = max(0, min(100, progress))
        if message:
            step.message = message
        
        self._calculate_overall_progress()
        self._notify_step("step_progress", step.progress, step.message)
        self._notify_progress("progress", self.overall_progress, step.message)
    
    def complete_step(self, step_name: str, message: str = "") -> None:
        """Complete specific step."""
        if step_name not in self.steps:
            return
        
        step = self.steps[step_name]
        step.completed = True
        step.end_time = time.time()
        step.progress = 100
        if message:
            step.message = message
        
        self._calculate_overall_progress()
        
        duration = step.end_time - step.start_time if step.start_time else 0
        complete_msg = f"{step.description} selesai ({duration:.1f}s)"
        
        self._notify_step("step_complete", 100, complete_msg)
        self._notify_progress("progress", self.overall_progress, complete_msg)
    
    def complete_process(self, message: str = "Download dataset selesai") -> None:
        """Complete seluruh download process."""
        self.is_active = False
        total_duration = time.time() - self.start_time if self.start_time else 0
        
        # Complete semua steps
        for step in self.steps.values():
            if not step.completed:
                step.completed = True
                step.progress = 100
                if not step.end_time:
                    step.end_time = time.time()
        
        self.overall_progress = 100
        final_message = f"{message} ({total_duration:.1f}s)"
        
        self._notify_step("complete", 100, final_message)
        self._notify_progress("complete", 100, final_message)
    
    def error_process(self, error_message: str, step_name: str = "") -> None:
        """Handle error dalam download process."""
        self.is_active = False
        
        if step_name and step_name in self.steps:
            step = self.steps[step_name]
            step.message = error_message
        
        self._notify_step("error", 0, error_message)
        self._notify_progress("error", 0, error_message)
    
    def _calculate_overall_progress(self) -> None:
        """Calculate overall progress berdasarkan step weights."""
        total_weighted_progress = 0
        total_weight = sum(step.weight for step in self.steps.values())
        
        for step in self.steps.values():
            if step.completed:
                weighted_progress = step.weight
            elif step.started:
                weighted_progress = (step.progress / 100) * step.weight
            else:
                weighted_progress = 0
            
            total_weighted_progress += weighted_progress
        
        self.overall_progress = int((total_weighted_progress / total_weight) * 100) if total_weight > 0 else 0
    
    def _notify_progress(self, event_type: str, progress: int, message: str) -> None:
        """Notify overall progress."""
        if self._progress_callback:
            try:
                self._progress_callback(event_type, progress, message)
            except Exception:
                pass  # Silent fail to prevent callback errors
    
    def _notify_step(self, event_type: str, progress: int, message: str) -> None:
        """Notify step progress."""
        if self._step_callback:
            try:
                self._step_callback(event_type, progress, message)
            except Exception:
                pass  # Silent fail to prevent callback errors
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary current progress."""
        current_step_info = self.steps.get(self.current_step, None)
        
        return {
            'overall_progress': self.overall_progress,
            'current_step': {
                'name': self.current_step,
                'description': current_step_info.description if current_step_info else "",
                'progress': current_step_info.progress if current_step_info else 0,
                'message': current_step_info.message if current_step_info else ""
            },
            'total_steps': len(self.steps),
            'completed_steps': sum(1 for step in self.steps.values() if step.completed),
            'is_active': self.is_active,
            'duration': time.time() - self.start_time if self.start_time else 0
        }
    
    def get_step_details(self) -> List[Dict[str, Any]]:
        """Get detail semua steps."""
        return [
            {
                'name': step.name,
                'description': step.description,
                'weight': step.weight,
                'started': step.started,
                'completed': step.completed,
                'progress': step.progress,
                'message': step.message,
                'duration': (step.end_time - step.start_time) if step.start_time and step.end_time else 0
            }
            for step in self.steps.values()
        ]

class CallbackManager:
    """Manager untuk callback registration dan execution."""
    
    def __init__(self):
        self._callbacks: Dict[str, List[Callable]] = {}
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register callback untuk event type dengan one-liner."""
        self._callbacks.setdefault(event_type, []).append(callback)
    
    def unregister_callback(self, event_type: str, callback: Callable) -> None:
        """Unregister callback dengan one-liner."""
        self._callbacks.get(event_type, []).remove(callback) if callback in self._callbacks.get(event_type, []) else None
    
    def notify_callbacks(self, event_type: str, *args, **kwargs) -> None:
        """Notify semua callbacks untuk event type."""
        for callback in self._callbacks.get(event_type, []):
            try:
                callback(*args, **kwargs)
            except Exception:
                pass  # Silent fail untuk prevent callback errors

# Factory functions
def create_download_tracker() -> DownloadProgressTracker:
    """Factory untuk create DownloadProgressTracker."""
    return DownloadProgressTracker()

def create_callback_manager() -> CallbackManager:
    """Factory untuk create CallbackManager."""
    return CallbackManager()

# Utility functions
def create_step_weights(include_validation: bool = True, include_organize: bool = True) -> Dict[str, int]:
    """Create step weights berdasarkan konfigurasi."""
    weights = {
        'metadata': 20,
        'download': 60,
        'extract': 20
    }
    
    if include_validation:
        weights = {'validation': 10, **{k: int(v * 0.9) for k, v in weights.items()}}
    
    if include_organize:
        weights['organize'] = 10
        total = sum(weights.values())
        weights = {k: int((v / total) * 100) for k, v in weights.items()}
    
    return weights

def format_progress_message(step: str, progress: int, message: str) -> str:
    """Format progress message dengan emoji dan info yang konsisten."""
    emoji_map = {
        'validation': '🔍',
        'metadata': '📋',
        'download': '📥',
        'extract': '📦',
        'organize': '📁',
        'complete': '✅',
        'error': '❌'
    }
    
    emoji = emoji_map.get(step, '🔄')
    return f"{emoji} {message} ({progress}%)" if progress > 0 else f"{emoji} {message}"