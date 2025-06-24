"""
File: smartcash/ui/setup/env_config/utils/progress_tracker.py
Deskripsi: Utility untuk tracking progress setup dengan state management yang lebih baik
"""

from typing import Dict, Any, Optional, List, Callable
from enum import Enum, auto
from dataclasses import dataclass
from IPython.display import display
import ipywidgets as widgets

# Import the progress tracker factory functions
from smartcash.ui.components.progress_tracker.factory import (
    create_dual_progress_tracker
)

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
    
    def __init__(self, ui_components: Dict[str, Any], logger=None):
        """Initialize the progress tracker
        
        Args:
            ui_components: Dictionary to store UI components
            logger: Optional logger instance (will create a basic one if not provided)
        """
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
        
        # Initialize logger
        self.logger = logger or self._create_dummy_logger()
        
        # Initialize the progress tracker UI
        self._initialize_progress_tracker()
    
    def _create_dummy_logger(self):
        """Create a basic logger if none is provided"""
        class DummyLogger:
            def info(self, msg): print(f"[INFO] {msg}")
            def warning(self, msg): print(f"[WARN] {msg}")
            def error(self, msg): print(f"[ERROR] {msg}")
            def debug(self, msg): print(f"[DEBUG] {msg}")
        return DummyLogger()
    
    def _initialize_progress_tracker(self) -> None:
        """Initialize progress tracker component using the factory pattern"""
        if 'progress_tracker' not in self.ui_components:
            # Create a dual progress tracker (overall + current task)
            tracker = create_dual_progress_tracker(
                operation="Environment Setup",
                auto_hide=False,
                show_step_info=True
            )
            
            # Store the tracker and its container in the UI components
            self.ui_components['progress_tracker'] = tracker
            self.ui_components['progress_container'] = tracker.container
            
            # Display the progress container
            display(tracker.container)
            if 'progress_container' in self.ui_components:
                self.ui_components['progress_container'].children = (progress_container,)
            else:
                self.ui_components['progress_container'] = progress_container
                
            # Force display update
            display(progress_container)
    
    def register_callback(self, callback: Callable[[str, int, str], None]) -> None:
        """Register callback untuk progress update"""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def update_stage(self, stage: SetupStage, message: str = "") -> None:
        """Update to a new stage"""
        self.current_stage = stage
        stage_name = self.stages[stage].name
        self.logger.info(f"Starting stage: {stage_name}")
        self.update_progress(stage, 0, message)
        
        # Update the progress tracker with the new stage
        if 'progress_tracker' in self.ui_components:
            tracker = self.ui_components['progress_tracker']
            tracker.update_current(progress=0, message=stage_name)
    
    def complete_stage(self, message: str = "") -> None:
        """Mark current stage as complete"""
        if self.current_stage is not None:
            self.update_progress(self.current_stage, 100, message)
    
    def update_within_stage(self, progress: int, message: str = "") -> None:
        """Update progress within the current stage
        
        Args:
            progress: Progress percentage (0-100)
            message: Optional status message
        """
        if self.current_stage is not None:
            self.stages[self.current_stage].current = progress
            self.update_progress(self.current_stage, progress, message)
    
    def update_step(self, step_name: str, progress: int = None, description: str = "") -> None:
        """Update the current step with progress and description
        
        Args:
            step_name: Name of the current step or progress percentage if progress is None
            progress: Optional progress percentage (0-100)
            description: Optional description of the step
        """
        if progress is None and step_name.isdigit():
            # Handle case where step_name is actually the progress (for backward compatibility)
            progress = int(step_name)
            step_name = description or f"Progress: {progress}%"
        
        if self.current_stage is not None:
            # If progress is provided, update the current stage's progress
            if progress is not None:
                self.stages[self.current_stage].current = progress
                
            # Get current stage data
            stage_data = self.stages[self.current_stage]
            
            # Log the step update
            self.logger.info(f"Step: {step_name} ({progress}%)" if progress is not None 
                           else f"Step: {step_name}")
            
            # Update progress with the new step name and description
            self.update_progress(
                self.current_stage, 
                stage_data.current, 
                description or step_name
            )
    
    def complete(self, message: str = "Setup completed successfully") -> None:
        """Mark setup as complete"""
        self.update_stage(SetupStage.COMPLETE, message)
        self.complete_stage()
        
        # Update the progress tracker with completion
        if 'progress_tracker' in self.ui_components:
            tracker = self.ui_components['progress_tracker']
            tracker.complete(message=message)
    
    def error(self, message: str) -> None:
        """Mark setup as failed with an error"""
        if 'progress_tracker' in self.ui_components:
            tracker = self.ui_components['progress_tracker']
            tracker.error(message=message)
    
    def hide(self) -> None:
        """Hide the progress tracker"""
        if 'progress_container' in self.ui_components:
            self.ui_components['progress_container'].layout.visibility = 'hidden'
    
    def show(self) -> None:
        """Show the progress tracker"""
        if 'progress_container' in self.ui_components:
            self.ui_components['progress_container'].layout.visibility = 'visible'
    
    def update_progress(self, stage: SetupStage, progress: int, message: str = "") -> None:
        """Update progress for a specific stage"""
        if stage not in self.stages:
            return
            
        # Update the stage progress
        self.stages[stage].current = progress
        
        # Calculate overall progress
        total_weight = sum(s.weight for s in self.stages.values())
        weighted_progress = sum(
            (s.current / s.total) * s.weight 
            for s in self.stages.values()
        )
        self.overall_progress = int((weighted_progress / total_weight) * 100)
        
        # Update UI using the factory-created tracker
        if 'progress_tracker' in self.ui_components:
            tracker = self.ui_components['progress_tracker']
            
            # Update overall progress bar
            tracker.update_overall(
                progress=self.overall_progress,
                message=f"Overall: {self.overall_progress}%"
            )
            
            # Update current task progress
            tracker.update_current(
                progress=progress,
                message=f"{self.stages[stage].name}: {message}" if message else self.stages[stage].name
            )
        
        # Call registered callbacks
        for callback in self.callbacks:
            try:
                callback("overall", self.overall_progress, self.stages[stage].name)
                callback("current", progress, message)
            except Exception as e:
                print(f"Error in progress callback: {e}")
            
            # Small delay to allow UI to update
            import time
            time.sleep(0.05)
            print(f"⚠️ Error updating progress: {e}")
            print(f"Stage: {stage_name}, Progress: {progress}, Message: {message}")

def track_setup_progress(ui_components: Dict[str, Any], logger=None) -> SetupProgressTracker:
    """🎯 Create and initialize progress tracker instance
    
    Args:
        ui_components: Dictionary to store UI components
        logger: Optional logger instance to use for progress tracking
        
    Returns:
        Configured SetupProgressTracker instance
    """
    # Create and initialize the progress tracker
    progress_tracker = SetupProgressTracker(ui_components, logger=logger)
    
    # Ensure the progress tracker is visible
    progress_tracker.show()
    
    # Start with the initial stage
    progress_tracker.update_stage(SetupStage.INIT, "Initializing setup...")
    
    return progress_tracker