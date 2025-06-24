"""
File: smartcash/ui/setup/env_config/utils/progress_tracker.py
Deskripsi: Utility untuk tracking progress setup dengan state management yang lebih baik
"""

from typing import Dict, Any, Optional, List, Callable
from enum import Enum, auto
from dataclasses import dataclass
from IPython.display import display
import ipywidgets as widgets

class SetupStage(Enum):
    """Tahapan setup environment"""
    INIT = auto()
    DRIVE_MOUNT = auto()
    CONFIG_SYNC = auto()
    FOLDER_SETUP = auto()
    ENV_SETUP = auto()
    COMPLETE = auto()

# Ensure the enum is available for import
__all__ = ['SetupStage', 'SetupProgressTracker', 'track_setup_progress']

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
        """Initialize progress tracker component with a single progress bar"""
        if 'progress_tracker' not in self.ui_components:
            # Create progress bar
            progress_bar = widgets.FloatProgress(
                value=0,
                min=0,
                max=100,
                description='Progress:',
                bar_style='info',
                style={'bar_color': '#4CAF50'},
                orientation='horizontal',
                layout=widgets.Layout(width='100%')
            )
            
            # Create status text
            status_text = widgets.HTML(value='<div style="padding: 5px 0;">Initializing setup...</div>')
            
            # Create container
            container = widgets.VBox([
                widgets.HTML('<h3>Environment Setup</h3>'),
                progress_bar,
                status_text
            ], layout=widgets.Layout(width='100%'))
            
            # Store components
            self.ui_components['progress_tracker'] = {
                'bar': progress_bar,
                'text': status_text,
                'container': container
            }
            self.ui_components['progress_container'] = container
            
            # Display the container
            display(container)
                
    def register_callback(self, callback: Callable[[str, int, str], None]) -> None:
        """Register callback untuk progress update"""
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def update_stage(self, stage: SetupStage, message: str = "") -> None:
        """Update to a new stage
        
        Args:
            stage: The stage to update to
            message: Optional message to display
        """
        try:
            self.logger.debug(f"[DEBUG] update_stage called with stage: {stage}, message: {message}")
            
            # Validate stage type
            if not isinstance(stage, SetupStage):
                error_msg = f"Invalid stage type: {type(stage).__name__}. Expected SetupStage enum."
                self.logger.error(error_msg)
                return
                
            # Store previous stage for completion
            previous_stage = self.current_stage
            
            # Set new stage
            self.current_stage = stage
            self.logger.debug(f"[DEBUG] Current stage updated to: {self.current_stage}")
            
            # Get stage name safely
            stage_info = self.stages.get(stage)
            if not stage_info:
                self.logger.error(f"Stage {stage} not found in stages configuration")
                return
                
            stage_name = stage_info.name
            self.logger.info(f"🔄 Starting stage: {stage_name}")
            
            # Complete previous stage if different
            if previous_stage and previous_stage != stage:
                self.logger.debug(f"[DEBUG] Completing previous stage: {previous_stage}")
                try:
                    self.complete_stage(f"Moving to {stage_name}")
                except Exception as e:
                    self.logger.error(f"Error completing previous stage: {e}")
            
            # Update progress for the new stage
            self.update_within_stage(0, f"Starting: {stage_name}")
            
            # Update UI
            self._update_ui(f"Starting: {stage_name}")
            
            # Log the stage transition
            self.logger.info(f"✅ Transitioned to stage: {stage_name}")
            
        except Exception as e:
            error_msg = f"Error in update_stage: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def complete_stage(self, message: str = "") -> None:
        """Mark current stage as complete
        
        Args:
            message: Optional completion message
        """
        try:
            if not hasattr(self, 'current_stage') or self.current_stage is None:
                self.logger.warning("No active stage to complete")
                return
                
            # Ensure the stage is in our stages dictionary
            if not hasattr(self, 'stages') or self.current_stage not in self.stages:
                self.logger.warning(f"Unknown stage: {self.current_stage}")
                return
            
            # Get the current stage progress
            stage_progress = self.stages.get(self.current_stage)
            if stage_progress is None:
                self.logger.warning(f"No progress data found for stage: {self.current_stage}")
                return
                
            try:
                # Update the current stage to 100%
                stage_progress.current = 100
                
                # Update overall progress
                self._update_overall_progress()
                
                # Log completion
                if message:
                    self.logger.info(f"✅ Completed stage {stage_progress.name}: {message}")
                else:
                    self.logger.info(f"✅ Completed stage {stage_progress.name}")
                
                # Update UI with completion message
                self._update_ui(f"Completed: {stage_progress.name}")
                
            except Exception as e:
                error_msg = f"Error updating stage completion: {str(e)}"
                self.logger.error(error_msg)
                if hasattr(self, 'logger'):
                    import traceback
                    self.logger.debug(traceback.format_exc())
            
            # Clear current stage after updating
            self.current_stage = None
            
        except Exception as e:
            error_msg = f"Error in complete_stage: {str(e)}"
            self.logger.error(error_msg)
            if hasattr(self, 'logger'):
                import traceback
                self.logger.debug(traceback.format_exc())
            
            # Ensure we don't leave the stage in an inconsistent state
            if hasattr(self, 'current_stage'):
                self.current_stage = None
    
    def update_within_stage(self, progress: int, message: str = "") -> None:
        """Update progress within the current stage
        
        Args:
            progress: Progress percentage (0-100)
            message: Optional status message
        """
        if self.current_stage is None:
            self.logger.warning("No active stage to update progress")
            return
            
        # Update the stage progress
        stage_progress = self.stages[self.current_stage]
        stage_progress.current = progress
        
        # Calculate overall progress
        self._update_overall_progress()
        
        # Update UI
        self._update_ui(message)
        
        # Log the update
        self.logger.debug(f"Progress update - Stage: {stage_progress.name}, Progress: {progress}%, Message: {message}")
        
        # Trigger callbacks
        for callback in self.callbacks:
            try:
                callback(stage_progress.name, progress, message)
            except Exception as e:
                self.logger.error(f"Error in progress callback: {e}")
    
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
            tracker['bar'].bar_style = 'success'
            tracker['bar'].description = "100%"
            tracker['text'].value = f'''
                <div style="padding: 5px 0; color: #4CAF50;">
                    <strong>✅ {message}</strong>
                </div>
            '''
    
    def error(self, message: str) -> None:
        """Mark setup as failed with an error"""
        if 'progress_tracker' in self.ui_components:
            tracker = self.ui_components['progress_tracker']
            tracker['bar'].bar_style = 'danger'
            tracker['text'].value = f'''
                <div style="padding: 5px 0; color: #f44336;">
                    <strong>❌ {message}</strong>
                </div>
            '''
    
    def hide(self) -> None:
        """Hide the progress tracker"""
        if 'progress_tracker' in self.ui_components and 'container' in self.ui_components['progress_tracker']:
            self.ui_components['progress_tracker']['container'].layout.visibility = 'hidden'
    
    def show(self) -> None:
        """Show the progress tracker"""
        if 'progress_tracker' in self.ui_components and 'container' in self.ui_components['progress_tracker']:
            self.ui_components['progress_tracker']['container'].layout.visibility = 'visible'
    
    def _update_ui(self, message: str = "") -> None:
        """Update the UI with the current progress and message"""
        if 'progress_tracker' in self.ui_components:
            tracker = self.ui_components['progress_tracker']
            if 'bar' in tracker:
                tracker['bar'].value = self.overall_progress
            if 'text' in tracker and message:
                tracker['text'].value = f'<div style="padding: 5px 0;">{message}</div>'
    
    def update_progress(self, stage: SetupStage, progress: int, message: str = "") -> None:
        """Update progress for a specific stage
        
        Args:
            stage: The stage to update
            progress: Progress percentage (0-100)
            message: Optional status message
        """
        try:
            # Validate stage type
            if not isinstance(stage, SetupStage):
                self.logger.error(f"Invalid stage type: {type(stage).__name__}. Expected SetupStage enum.")
                return
                
            # Validate progress value
            progress = max(0, min(100, int(progress)))
            
            # Ensure stages dictionary exists
            if not hasattr(self, 'stages') or not isinstance(self.stages, dict):
                self.logger.error("Stages dictionary not properly initialized")
                return
                
            # Get stage info
            stage_info = self.stages.get(stage)
            if not stage_info:
                self.logger.warning(f"Stage {stage} not found in stages configuration")
                return
                
            # Update stage progress
            stage_info.current = progress
            
            # Update overall progress
            self._update_overall_progress()
            
            # Update UI
            self._update_ui(message)
            
            # Log the update
            self.logger.debug(f"Progress update - Stage: {stage_info.name}, Progress: {progress}%")
            
        except Exception as e:
            error_msg = f"Error in update_progress: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _update_overall_progress(self) -> None:
        """Calculate and update the overall progress based on all stages"""
        try:
            if not hasattr(self, 'stages') or not self.stages:
                self.overall_progress = 0
                return
                
            total_weight = sum(stage.weight for stage in self.stages.keys() if stage != SetupStage.COMPLETE)
            if total_weight <= 0:
                self.overall_progress = 0
                return
                
            # Calculate weighted progress
            weighted_progress = 0
            for stage, stage_info in self.stages.items():
                if stage == SetupStage.COMPLETE:
                    continue
                    
                # Get progress for this stage (0-100)
                stage_progress = min(100, max(0, stage_info.current))
                
                # Add weighted contribution to overall progress
                weighted_progress += (stage_progress * stage_info.weight) / total_weight
                
            # Update overall progress (0-100)
            self.overall_progress = min(100, max(0, int(weighted_progress)))
            
            # Update UI if needed
            if hasattr(self, 'ui_components') and 'progress_tracker' in self.ui_components:
                tracker = self.ui_components['progress_tracker']
                if 'bar' in tracker and hasattr(tracker['bar'], 'value'):
                    tracker['bar'].value = self.overall_progress
                    
            # Log the update
            current_stage_name = ""
            if hasattr(self, 'current_stage') and self.current_stage is not None:
                stage_info = self.stages.get(self.current_stage)
                if stage_info:
                    current_stage_name = stage_info.name
            
            self.logger.debug(f"Overall progress: {self.overall_progress}%, Current stage: {current_stage_name}")
                    
        except Exception as e:
            error_msg = f"Error in _update_overall_progress: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
                import traceback
                self.logger.debug(traceback.format_exc())
            else:
                print(error_msg)
                # Get current stage info if available for debugging
                if hasattr(self, 'current_stage') and self.current_stage is not None:
                    stage_info = self.stages.get(self.current_stage)
                    if stage_info:
                        current_stage_name = stage_info.name
                        current_progress = stage_info.current
            
            # Calculate overall progress if stages are available
            if hasattr(self, 'stages') and self.stages:
                total_weight = sum(stage.weight for stage in self.stages.values())
                if total_weight > 0:
                    weighted_sum = sum(
                        stage.weight * (stage.current / 100) 
                        for stage in self.stages.values()
                    )
                    self.overall_progress = min(100, int(weighted_sum / total_weight * 100))
            
            # Update UI if progress tracker is available
            if 'progress_tracker' in getattr(self, 'ui_components', {}):
                try:
                    tracker = self.ui_components['progress_tracker']
                    if 'bar' in tracker and hasattr(tracker['bar'], 'value'):
                        tracker['bar'].value = self.overall_progress
                        tracker['bar'].description = f"{self.overall_progress}%"
                    
                    if 'text' in tracker and hasattr(tracker['text'], 'value'):
                        tracker['text'].value = f'''
                            <div style="padding: 5px 0;">
                                <div><strong>Current Task:</strong> {current_stage_name} ({current_progress}%)</div>
                            </div>
                        '''
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.error(f"Error updating UI: {str(e)}")
            
            # Process callbacks if available
            if hasattr(self, 'callbacks') and isinstance(self.callbacks, list):
                # Create a copy of callbacks to avoid modification during iteration
                callbacks = self.callbacks.copy()
                for callback in callbacks:
                    if not callable(callback):
                        continue
                        
                    try:
                        # Only pass valid stage info if available
                        if (hasattr(self, 'current_stage') and 
                            self.current_stage is not None and 
                            hasattr(self, 'stages') and 
                            self.current_stage in self.stages):
                            
                            # Call with overall progress
                            callback("overall", self.overall_progress, self.stages[self.current_stage].name)
                            
                            # Get current progress for the current stage
                            current_stage_progress = 0
                            if (hasattr(self, 'current_stage') and 
                                self.current_stage is not None and 
                                hasattr(self, 'stages') and 
                                self.current_stage in self.stages):
                                current_stage_progress = self.stages[self.current_stage].current
                                
                            # Call with current stage progress
                            callback("current", current_stage_progress, "")
                            
                    except Exception as e:
                        if hasattr(self, 'logger'):
                            self.logger.error(f"Error in progress callback: {e}")
                        continue
                        
                    # Small delay to allow UI to update
                    import time
                    time.sleep(0.01)  # Reduced delay for better responsiveness
                
        except Exception as e:
            error_msg = f"Error in _update_overall_progress: {str(e)}"
            if hasattr(self, 'logger'):
                self.logger.error(error_msg)
                import traceback
                self.logger.debug(traceback.format_exc())
            else:
                print(error_msg)
    
    def complete(self, message: str = "Setup completed successfully") -> None:
        """Mark setup as complete"""
        self.update_stage(SetupStage.COMPLETE, message)
        self.complete_stage()
        
        # Update the progress tracker with completion
        if 'progress_tracker' in self.ui_components:
            tracker = self.ui_components['progress_tracker']
            tracker['bar'].bar_style = 'success'
            tracker['bar'].description = "100%"
            tracker['text'].value = f'''
                <div style="padding: 5px 0; color: #4CAF50;">
                    <strong>✅ {message}</strong>
                </div>
            '''
    
    def error(self, message: str) -> None:
        """Mark setup as failed with an error"""
        if 'progress_tracker' in self.ui_components:
            tracker = self.ui_components['progress_tracker']
            tracker['bar'].bar_style = 'danger'
            tracker['text'].value = f'''
                <div style="padding: 5px 0; color: #f44336;">
                    <strong>❌ {message}</strong>
                </div>
            '''

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