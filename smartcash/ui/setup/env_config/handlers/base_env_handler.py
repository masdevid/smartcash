"""
Base handler for environment configuration with progress tracking.

This module provides a base class for all environment configuration handlers
with common functionality including progress tracking and stage management.
"""

from enum import Enum, auto
from typing import Dict, Any, Optional, TypeVar, Tuple

from smartcash.ui.handlers.base_handler import BaseHandler
from smartcash.ui.setup.env_config.constants import SetupStage, STAGE_WEIGHTS

T = TypeVar('T')

class BaseEnvHandler(BaseHandler):
    """Base handler for environment configuration with progress tracking.
    
    This class provides common functionality for all environment configuration
    handlers, including stage-based progress tracking and status management.
    
    Key Features:
    - Stage-based progress tracking using SetupStage enum
    - Automatic progress calculation between stages
    - Status message management
    - Error handling with stage context
    """
    
    # Stage weights for progress calculation (must sum to 100)
    # These weights are imported from constants.py
    STAGE_WEIGHTS = STAGE_WEIGHTS
    
    # Default configuration for the handler
    DEFAULT_CONFIG: Dict[str, Any] = {
        'auto_initialize': True
    }
    
    def __init__(self, module_name: str, parent_module: str = 'env_config', 
                 config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize the base environment handler.
        
        Args:
            module_name: Name of the module (usually __name__)
            parent_module: Name of the parent module (default: 'env_config')
            config: Optional configuration overrides
            **kwargs: Additional keyword arguments for BaseHandler
        """
        super().__init__(
            module_name=module_name,
            parent_module=parent_module,
            **kwargs
        )
        
        # Initialize configuration
        self._config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._progress_tracker = None  # Will be set by the component
        self._current_stage = SetupStage.INIT
        self._stage_progress = 0.0  # Progress within current stage (0-100)
        self._stage_weights = dict(self.STAGE_WEIGHTS)  # Copy of stage weights
        
        # Validate stage weights
        total_weight = sum(self._stage_weights.values())
        if abs(total_weight - 100) > 0.001:  # Allow for floating point errors
            raise ValueError(f"Stage weights must sum to 100, got {total_weight}")
    
    @property
    def progress_tracker(self):
        """Get the progress tracker instance.
        
        Components should set this after creating the progress tracker.
        """
        return self._progress_tracker
    
    @progress_tracker.setter
    def progress_tracker(self, tracker):
        """Set the progress tracker instance.
        
        Components should call this after creating the progress tracker.
        """
        self._progress_tracker = tracker
    
    def set_stage(self, stage: SetupStage, message: str = "") -> None:
        """Set the current stage of the setup process.
        
        Args:
            stage: The new stage to transition to
            message: Optional message to log with the stage transition
        """
        if not isinstance(stage, SetupStage):
            raise ValueError(f"Invalid stage type: {type(stage)}. Must be a SetupStage enum.")
            
        self._current_stage = stage
        self._stage_progress = 0.0
        
        # Log the stage transition
        stage_name = stage.name.replace('_', ' ').title()
        log_message = f"Entering stage: {stage_name}"
        if message:
            log_message += f" - {message}"
            
        self.logger.info(log_message)
        self.update_stage_progress(0, message=log_message)
    
    def get_stage_progress_range(self) -> Tuple[float, float]:
        """Get the progress range for the current stage.
        
        Returns:
            Tuple of (start_percent, end_percent) for the current stage
        """
        # Get all stages up to and including the current one
        stages = list(SetupStage)
        current_idx = stages.index(self._current_stage)
        
        # Calculate start and end percentages
        start_pct = sum(self._stage_weights[s] for s in stages[:current_idx])
        end_pct = start_pct + self._stage_weights[self._current_stage]
        
        return start_pct, end_pct
    
    def update_stage_progress(self, progress: float, message: str = "") -> None:
        """Update progress within the current stage.
        
        Args:
            progress: Progress within current stage (0-100)
            message: Optional progress message
        """
        if not self._progress_tracker:
            self.logger.warning("Progress tracker not set. Component should set progress_tracker.")
            return
            
        try:
            # Update stage progress (clamped to 0-100)
            self._stage_progress = max(0.0, min(100.0, float(progress)))
            
            # Calculate overall progress based on stage weights
            start_pct, end_pct = self.get_stage_progress_range()
            stage_weight = end_pct - start_pct
            overall_progress = start_pct + (self._stage_progress / 100.0 * stage_weight)
            
            # Update progress trackers
            self._progress_tracker.update_current(self._stage_progress, message)
            self._progress_tracker.update_primary(overall_progress, message)
            
            self.logger.debug(
                f"Stage {self._current_stage.name}: {self._stage_progress:.1f}% "
                f"(Overall: {overall_progress:.1f}%) - {message}"
            )
            
        except (TypeError, ValueError) as e:
            self.logger.error(f"Invalid progress value: {progress} - {str(e)}")
            raise
        except AttributeError as e:
            self.logger.error(f"Progress tracker missing required method: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error updating progress: {str(e)}", exc_info=True)
            raise
    
    def update_progress(self, progress: float, message: str = "") -> None:
        """Update progress within the current stage (alias for update_stage_progress).
        
        Args:
            progress: Progress within current stage (0-100)
            message: Optional progress message
        """
        self.update_stage_progress(progress, message)
    
    def complete_stage(self, message: str = "") -> None:
        """Mark the current stage as complete and move to the next stage.
        
        Args:
            message: Optional completion message
        """
        # Complete current stage
        self.update_stage_progress(100.0, message or f"Completed {self._current_stage.name}")
        
        # Move to next stage
        stages = list(SetupStage)
        current_idx = stages.index(self._current_stage)
        if current_idx < len(stages) - 1:
            next_stage = stages[current_idx + 1]
            self.set_stage(next_stage, "Moving to next stage")
    
    def complete_progress(self, message: str = "Operation completed") -> None:
        """Mark progress as complete.
        
        Args:
            message: Completion message
        """
        # Ensure we're at 100% in the final stage
        if self._current_stage != SetupStage.COMPLETE:
            self.set_stage(SetupStage.COMPLETE, "Finalizing setup")
            
        self.update_stage_progress(100.0, message)
        if self._progress_tracker:
            self._progress_tracker.complete(message)
    
    def error_progress(self, message: str) -> None:
        """Mark progress as error.
        
        Args:
            message: Error message
        """
        error_msg = f"Error in {self._current_stage.name}: {message}"
        self.logger.error(error_msg)
        if self._progress_tracker:
            self._progress_tracker.error(error_msg)
    
    def reset_progress(self) -> None:
        """Reset progress tracker."""
        if self._progress_tracker:
            self._progress_tracker.reset()
    
    async def initialize(self) -> bool:
        """Initialize the handler.
        
        Returns:
            bool: True if initialization was successful
        """
        if self._config.get('auto_initialize', True):
            try:
                self.update_progress('overall', 0, "Initializing...")
                result = await self._initialize_impl()
                if result:
                    self.complete_progress("Initialization completed successfully")
                return result
            except Exception as e:
                self.error_progress(f"Initialization failed: {str(e)}")
                raise
        return True
    
    async def _initialize_impl(self) -> bool:
        """Implementation of initialization logic.
        
        Subclasses should override this method.
        
        Returns:
            bool: True if initialization was successful
        """
        return True
    
    async def cleanup(self) -> None:
        """Clean up resources used by the handler."""
        if self._progress_tracker:
            self._progress_tracker.close()
