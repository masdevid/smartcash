"""
File: smartcash/ui/setup/env_config/utils/progress_tracker.py
Deskripsi: Utility untuk tracking progress setup dengan state management yang lebih baik
"""

from typing import Dict, Any, Optional, List, Callable
from enum import Enum, auto
from dataclasses import dataclass
from smartcash.ui.components.progress_tracker import ProgressTracker, ProgressConfig, ProgressBarConfig, ProgressLevel
from smartcash.ui.setup.env_config.utils.ui_updater import update_status_panel

class SetupStage(Enum):
    """Tahapan setup environment"""
    INIT = auto()
    DRIVE_MOUNT = auto()
    CONFIG_SYNC = auto()
    FOLDER_SETUP = auto()
    ENV_SETUP = auto()
    COMPLETE = auto()

@dataclass
class StageProgress:
    """Data kemajuan untuk setiap tahapan"""
    name: str
    weight: int
    current: int = 0
    total: int = 100

class SetupProgressTracker:
    """📊 Progress tracker untuk setup workflow dengan state management"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.current_stage: Optional[SetupStage] = None
        self.stages: Dict[SetupStage, StageProgress] = {
            SetupStage.INIT: StageProgress("Initializing", 5),
            SetupStage.DRIVE_MOUNT: StageProgress("Mounting Drive", 15),
            SetupStage.CONFIG_SYNC: StageProgress("Syncing Configs", 20),
            SetupStage.FOLDER_SETUP: StageProgress("Setting Up Folders", 30),
            SetupStage.ENV_SETUP: StageProgress("Configuring Environment", 25),
            SetupStage.COMPLETE: StageProgress("Setup Complete", 5)
        }
        self.overall_progress = 0
        self.callbacks: List[Callable[[str, int, str], None]] = []
        self._initialize_progress_tracker()
    
    def _initialize_progress_tracker(self) -> None:
        """Initialize progress tracker component"""
        if 'progress_tracker' not in self.ui_components:
            config = ProgressConfig(
                bars=[
                    ProgressBarConfig(
                        name="overall",
                        label="Overall Progress",
                        level=ProgressLevel.PRIMARY,
                        visible=True,
                        show_percentage=True
                    ),
                    ProgressBarConfig(
                        name="current",
                        label="Current Task",
                        level=ProgressLevel.SECONDARY,
                        visible=True,
                        show_percentage=True
                    )
                ]
            )
            self.ui_components['progress_tracker'] = ProgressTracker(config)
    
    def register_callback(self, callback: Callable[[str, int, str], None]) -> None:
        """Register callback untuk progress update"""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def start_stage(self, stage: SetupStage) -> None:
        """Mulai tahapan baru"""
        self.current_stage = stage
        stage_data = self.stages[stage]
        self._update_progress(stage_data.name, 0, f"🚀 Starting {stage_data.name.lower()}...")
    
    def update_stage_progress(self, progress: int, message: str = "") -> None:
        """Update progress tahapan saat ini"""
        if self.current_stage is None:
            return
            
        stage_data = self.stages[self.current_stage]
        stage_data.current = max(0, min(progress, 100))
        
        # Hitung progress keseluruhan
        total_weight = sum(stage.weight for stage in self.stages.values())
        completed_weight = 0
        
        for stage, data in self.stages.items():
            if stage == self.current_stage:
                completed_weight += (data.weight * progress) / 100
            elif stage.value < self.current_stage.value:
                completed_weight += data.weight
        
        self.overall_progress = min(100, (completed_weight / total_weight) * 100)
        
        # Update UI
        self._update_progress(
            stage_data.name,
            int(self.overall_progress),
            message or f"{stage_data.name} in progress..."
        )
    
    def complete_stage(self, message: str = "") -> None:
        """Tandai tahapan saat ini selesai"""
        if self.current_stage is None:
            return
            
        stage_data = self.stages[self.current_stage]
        stage_data.current = 100
        
        self._update_progress(
            stage_data.name,
            self.overall_progress,
            message or f"✅ {stage_data.name} completed"
        )
    
    def error(self, message: str) -> None:
        """Tandai error pada tahapan saat ini"""
        if self.current_stage is not None:
            stage_data = self.stages[self.current_stage]
            self._update_progress(
                stage_data.name,
                self.overall_progress,
                f"❌ Error: {message}",
                is_error=True
            )
    
    def update_step(self, step_name: str, description: str = "") -> None:
        """Update the current step with a description (compatibility method)"""
        if self.current_stage is not None:
            stage_data = self.stages[self.current_stage]
            stage_data.name = step_name
            self._update_progress(step_name, stage_data.current, description or step_name)
    
    def _update_progress(self, stage_name: str, progress: int, message: str, is_error: bool = False) -> None:
        """Update progress UI dan trigger callbacks"""
        # Update progress bars
        if 'progress_tracker' in self.ui_components:
            tracker = self.ui_components['progress_tracker']
            tracker.update_bar('overall', self.overall_progress)
            tracker.update_bar('current', progress, message)
        
        # Update status panel
        update_status_panel(self.ui_components, message, 'error' if is_error else 'info')
        
        # Trigger callbacks
        for callback in self.callbacks:
            try:
                callback(stage_name, progress, message)
            except Exception as e:
                print(f"Error in progress callback: {e}")
        
        # Update status panel
        status_type = "error" if is_error else "info"
        update_status_panel(
            self.ui_components.get('status_panel'),
            message,
            status_type
        )
        
        # Trigger callbacks
        for callback in self.callbacks:
            try:
                callback(stage_name, progress, message)
            except Exception as e:
                print(f"⚠️ Error in progress callback: {e}")

def track_setup_progress(ui_components: Dict[str, Any]) -> SetupProgressTracker:
    """🎯 Create and initialize progress tracker instance"""
    return SetupProgressTracker(ui_components)