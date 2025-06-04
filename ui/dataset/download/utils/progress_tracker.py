"""
File: smartcash/ui/dataset/download/utils/progress_tracker.py
Deskripsi: Simplified progress tracker dengan direct ProgressTracker integration
"""

import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class ProgressStage(Enum):
    """Enum untuk stage progress download."""
    INIT = "init"
    VALIDATION = "validation"
    METADATA = "metadata"
    DOWNLOAD = "download" 
    EXTRACT = "extract"
    ORGANIZE = "organize"
    VERIFY = "verify"
    COMPLETE = "complete"
    ERROR = "error"

@dataclass
class ProgressStep:
    """Data class untuk progress step."""
    name: str
    stage: ProgressStage
    weight: int  # Percentage weight dalam overall progress
    description: str
    started: bool = False
    completed: bool = False
    progress: int = 0  # 0-100
    message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class DownloadProgressTracker:
    """Progress tracker untuk download process dengan ProgressTracker integration."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
        self.progress_tracker = ui_components.get('tracker')
        
        self.steps: List[ProgressStep] = []
        self.current_step_index = 0
        self.overall_progress = 0
        self.start_time = None
        self.is_active = False
        self._progress_callbacks: List[Callable] = []
        
        self._setup_default_steps()
    
    def _setup_default_steps(self) -> None:
        """Setup default steps untuk download process."""
        self.steps = [
            ProgressStep("validation", ProgressStage.VALIDATION, 5, "Validasi parameter"),
            ProgressStep("metadata", ProgressStage.METADATA, 10, "Ambil metadata dataset"),
            ProgressStep("download", ProgressStage.DOWNLOAD, 50, "Download dataset"),
            ProgressStep("extract", ProgressStage.EXTRACT, 15, "Ekstrak dataset"),
            ProgressStep("organize", ProgressStage.ORGANIZE, 15, "Organisir dataset"),
            ProgressStep("verify", ProgressStage.VERIFY, 5, "Verifikasi hasil")
        ]
    
    def add_progress_callback(self, callback: Callable[[str, int, str], None]) -> None:
        """Add callback untuk progress updates."""
        if callback not in self._progress_callbacks:
            self._progress_callbacks.append(callback)
    
    def start_tracking(self, custom_steps: Optional[List[ProgressStep]] = None) -> None:
        """Start progress tracking."""
        if custom_steps:
            self.steps = custom_steps
        
        self.start_time = time.time()
        self.is_active = True
        self.current_step_index = 0
        self.overall_progress = 0
        
        for step in self.steps:
            step.started = False
            step.completed = False
            step.progress = 0
            step.message = ""
            step.start_time = None
            step.end_time = None
        
        self._setup_ui_progress()
        self._notify_callbacks("start", 0, "Memulai proses download")
        
        self.logger and self.logger.info(f"🚀 Progress tracking dimulai dengan {len(self.steps)} tahap")
    
    def start_step(self, step_name: str, message: str = "") -> None:
        """Start specific step."""
        step = self._find_step(step_name)
        if not step:
            self.logger and self.logger.warning(f"⚠️ Step tidak ditemukan: {step_name}")
            return
        
        step.started = True
        step.start_time = time.time()
        step.message = message or step.description
        step.progress = 0
        
        self.current_step_index = self.steps.index(step)
        self._update_ui_progress()
        self._notify_callbacks("step_start", step.progress, step.message)
        
        self.logger and self.logger.info(f"🔄 {step.description}: {step.message}")
    
    def update_step_progress(self, step_name: str, progress: int, message: str = "") -> None:
        """Update progress untuk specific step."""
        step = self._find_step(step_name)
        if not step:
            return
        
        step.progress = max(0, min(100, progress))
        if message:
            step.message = message
        
        self._calculate_overall_progress()
        self._update_ui_progress()
        self._notify_callbacks("step_progress", step.progress, step.message)
    
    def complete_step(self, step_name: str, message: str = "") -> None:
        """Complete specific step."""
        step = self._find_step(step_name)
        if not step:
            return
        
        step.completed = True
        step.end_time = time.time()
        step.progress = 100
        if message:
            step.message = message
        
        self._calculate_overall_progress()
        self._update_ui_progress()
        self._notify_callbacks("step_complete", 100, step.message)
        
        duration = step.end_time - step.start_time if step.start_time else 0
        self.logger and self.logger.success(f"✅ {step.description} selesai ({duration:.1f}s)")
    
    def complete_tracking(self, message: str = "Download selesai") -> None:
        """Complete seluruh tracking process."""
        self.is_active = False
        total_duration = time.time() - self.start_time if self.start_time else 0
        
        for step in self.steps:
            if not step.completed:
                step.completed = True
                step.progress = 100
                if not step.end_time:
                    step.end_time = time.time()
        
        self.overall_progress = 100
        self._complete_ui_progress(message)
        self._notify_callbacks("complete", 100, message)
        
        self.logger and self.logger.success(f"🎉 {message} ({total_duration:.1f}s)")
    
    def error_tracking(self, error_message: str) -> None:
        """Handle error dalam tracking."""
        self.is_active = False
        self._error_ui_progress(error_message)
        self._notify_callbacks("error", 0, error_message)
        
        self.logger and self.logger.error(f"❌ Progress error: {error_message}")
    
    def _find_step(self, step_name: str) -> Optional[ProgressStep]:
        """Find step by name."""
        return next((step for step in self.steps if step.name == step_name), None)
    
    def _calculate_overall_progress(self) -> None:
        """Calculate overall progress berdasarkan step weights."""
        total_weighted_progress = 0
        total_weight = sum(step.weight for step in self.steps)
        
        for step in self.steps:
            if step.completed:
                weighted_progress = step.weight
            elif step.started:
                weighted_progress = (step.progress / 100) * step.weight
            else:
                weighted_progress = 0
            
            total_weighted_progress += weighted_progress
        
        self.overall_progress = int((total_weighted_progress / total_weight) * 100) if total_weight > 0 else 0
    
    def _setup_ui_progress(self) -> None:
        """Setup UI progress untuk tracking."""
        if self.progress_tracker:
            self.progress_tracker.show('download')
        elif 'show_for_operation' in self.ui_components:
            self.ui_components['show_for_operation']('download')
    
    def _update_ui_progress(self) -> None:
        """Update UI progress bars."""
        if not self.is_active:
            return
        
        current_step = self._get_current_step()
        
        if self.progress_tracker:
            self.progress_tracker.update(
                'overall', 
                self.overall_progress, 
                f"Progress keseluruhan ({self.overall_progress}%)"
            )
            
            if current_step:
                self.progress_tracker.update(
                    'step', 
                    current_step.progress, 
                    f"{current_step.description}: {current_step.message}"
                )
        elif 'update_progress' in self.ui_components:
            self.ui_components['update_progress'](
                'overall', 
                self.overall_progress, 
                f"📊 Progress keseluruhan ({self.overall_progress}%)"
            )
            
            if current_step:
                self.ui_components['update_progress'](
                    'step', 
                    current_step.progress, 
                    f"🔄 {current_step.description}: {current_step.message}"
                )
    
    def _complete_ui_progress(self, message: str) -> None:
        """Complete UI progress dengan success state."""
        if self.progress_tracker:
            self.progress_tracker.complete(message)
        elif 'complete_operation' in self.ui_components:
            self.ui_components['complete_operation'](message)
    
    def _error_ui_progress(self, error_message: str) -> None:
        """Set UI error state."""
        if self.progress_tracker:
            self.progress_tracker.error(error_message)
        elif 'error_operation' in self.ui_components:
            self.ui_components['error_operation'](error_message)
    
    def _get_current_step(self) -> Optional[ProgressStep]:
        """Get current active step."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    def _notify_callbacks(self, event_type: str, progress: int, message: str) -> None:
        """Notify semua registered callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(event_type, progress, message)
            except Exception:
                pass
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary progress saat ini."""
        current_step = self._get_current_step()
        total_duration = time.time() - self.start_time if self.start_time else 0
        
        return {
            'overall_progress': self.overall_progress,
            'current_step': {
                'name': current_step.name if current_step else None,
                'description': current_step.description if current_step else None,
                'progress': current_step.progress if current_step else 0,
                'message': current_step.message if current_step else ""
            },
            'total_steps': len(self.steps),
            'completed_steps': sum(1 for step in self.steps if step.completed),
            'duration': total_duration,
            'is_active': self.is_active,
            'steps_summary': [
                {
                    'name': step.name,
                    'description': step.description,
                    'completed': step.completed,
                    'progress': step.progress,
                    'weight': step.weight
                }
                for step in self.steps
            ]
        }

def create_download_progress_tracker(ui_components: Dict[str, Any]) -> DownloadProgressTracker:
    """Factory function untuk create download progress tracker."""
    return DownloadProgressTracker(ui_components)

def create_custom_progress_steps(include_validation: bool = True, 
                                include_verification: bool = True) -> List[ProgressStep]:
    """Create custom progress steps berdasarkan kebutuhan."""
    steps = []
    
    if include_validation:
        steps.append(ProgressStep("validation", ProgressStage.VALIDATION, 5, "Validasi parameter"))
    
    steps.extend([
        ProgressStep("metadata", ProgressStage.METADATA, 10, "Ambil metadata dataset"),
        ProgressStep("download", ProgressStage.DOWNLOAD, 50, "Download dataset"),
        ProgressStep("extract", ProgressStage.EXTRACT, 15, "Ekstrak dataset"),
        ProgressStep("organize", ProgressStage.ORGANIZE, 15, "Organisir dataset")
    ])
    
    if include_verification:
        steps.append(ProgressStep("verify", ProgressStage.VERIFY, 5, "Verifikasi hasil"))
    
    # Adjust weights untuk ensure total = 100%
    total_weight = sum(step.weight for step in steps)
    if total_weight != 100:
        for step in steps:
            step.weight = int((step.weight / total_weight) * 100)
        
        current_total = sum(step.weight for step in steps)
        if current_total != 100:
            largest_step = max(steps, key=lambda s: s.weight)
            largest_step.weight += (100 - current_total)
    
    return steps