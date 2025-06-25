"""
File: smartcash/ui/setup/env_config/utils/dual_progress_tracker.py
Deskripsi: Dual progress tracker implementation using the new ProgressTracker component
"""

from typing import Dict, Any, Optional, List, Callable
from enum import Enum, auto
from smartcash.ui.components.progress_tracker.factory import create_dual_progress_tracker
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker as NewProgressTracker

class SetupStage(Enum):
    """Stages for setup progress tracking"""
    DRIVE_MOUNT = auto()
    FOLDER_SETUP = auto()
    SYMLINK_SETUP = auto()
    CONFIG_SYNC = auto()
    ENV_SETUP = auto()
    COMPLETE = auto()

class DualProgressTracker:
    """Dual progress tracker for setup workflow with two-level progress tracking.
    
    This provides a backward-compatible interface for the old progress tracker
    while using the new ProgressTracker component under the hood.
    """
    
    def __init__(self, ui_components: Dict[str, Any] = None, logger=None):
        """Initialize the dual progress tracker.
        
        Args:
            ui_components: Dictionary containing UI components
            logger: Optional logger instance
        """
        self.logger = logger
        self.ui_components = ui_components or {}
        self.current_stage = None
        self.overall_progress = 0
        self.stage_progress = 0
        self.callbacks = []
        self._initialized = False
        self._new_tracker: Optional[NewProgressTracker] = None
        
    def _init_tracker(self):
        """Initialize the underlying progress tracker if not already done."""
        if self._initialized:
            return
            
        self._new_tracker = create_dual_progress_tracker(
            operation="Environment Setup",
            auto_hide=False
        )
        
        # Add to UI components if provided
        if self.ui_components:
            self.ui_components['progress_tracker'] = self._new_tracker
            self.ui_components['progress_container'] = self._new_tracker.container
            
        self._initialized = True
    
    def update_stage(self, stage: SetupStage, message: str = None):
        """Update the current stage of the setup process.
        
        Args:
            stage: The current setup stage
            message: Optional message to display
        """
        self._init_tracker()
        self.current_stage = stage
        stage_name = stage.name.replace('_', ' ').title()
        
        if message is None:
            message = f"{stage_name}..."
            
        self._new_tracker.update_overall(
            self.overall_progress,
            f"{stage_name}: {message}"
        )
        self._process_callbacks(stage, 0, message)
    
    def update_within_stage(self, progress: int, message: str = ""):
        """Update progress within the current stage.
        
        Args:
            progress: Progress percentage (0-100)
            message: Optional progress message
        """
        if not self._initialized or self.current_stage is None:
            return
            
        self.stage_progress = max(0, min(100, progress))
        stage_name = self.current_stage.name.replace('_', ' ').title()
        
        self._new_tracker.update_current(
            progress,
            f"{stage_name}: {message}" if message else stage_name
        )
        
        # Update overall progress based on stage weights
        self._update_overall_progress()
        self._process_callbacks(self.current_stage, progress, message)
    
    def complete_stage(self, message: str = ""):
        """Mark the current stage as complete.
        
        Args:
            message: Optional completion message
        """
        if not self._initialized or self.current_stage is None:
            return
            
        self.update_within_stage(100, message or f"{self.current_stage.name.replace('_', ' ').title()} completed")
    
    def complete(self, message: str = "Setup completed successfully!"):
        """Mark the entire setup as complete.
        
        Args:
            message: Completion message
        """
        self._init_tracker()
        self.current_stage = SetupStage.COMPLETE
        self.overall_progress = 100
        self.stage_progress = 100
        
        self._new_tracker.complete(message)
        self._process_callbacks(self.current_stage, 100, message)
    
    def error(self, message: str):
        """Report an error in the setup process.
        
        Args:
            message: Error message
        """
        self._init_tracker()
        self._new_tracker.error(message)
        
        if self.logger:
            self.logger.error(message)
    
    def add_callback(self, callback: Callable):
        """Add a callback function to be called on progress updates.
        
        Args:
            callback: Function with signature (stage, progress, message)
        """
        if callable(callback) and callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """Remove a previously registered callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _update_overall_progress(self):
        """Update the overall progress based on the current stage and progress."""
        if self.current_stage is None:
            return
            
        # Simple linear progress calculation - can be enhanced with stage weights
        stage_count = len(SetupStage) - 1  # Exclude COMPLETE stage
        stage_index = list(SetupStage).index(self.current_stage)
        
        # Calculate progress based on stage completion and within-stage progress
        stage_weight = 100 / stage_count
        completed_stages = stage_index - 1  # -1 because we don't count the current stage
        stage_progress = (self.stage_progress / 100) * stage_weight
        
        self.overall_progress = min(100, int(completed_stages * stage_weight + stage_progress))
        
        # Update the overall progress bar
        self._new_tracker.update_overall(self.overall_progress)
    
    def _process_callbacks(self, stage: SetupStage, progress: int, message: str):
        """Process all registered callbacks.
        
        Args:
            stage: Current setup stage
            progress: Current progress percentage
            message: Progress message
        """
        for callback in list(self.callbacks):  # Create a copy to avoid modification during iteration
            try:
                if callable(callback):
                    callback(stage, progress, message)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error in progress callback: {e}", exc_info=True)

# Backward compatibility
track_setup_progress = DualProgressTracker
