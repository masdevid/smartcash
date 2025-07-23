"""
File: smartcash/ui/model/evaluation/operations/evaluation_base_operation.py
Description: Base operation class for evaluation operations following the operation pattern.
"""

import time
from abc import abstractmethod
from enum import Enum, auto
from typing import Dict, Any, Callable, Optional, TYPE_CHECKING

from smartcash.ui.core.mixins.operation_mixin import OperationMixin
from smartcash.ui.core.mixins.logging_mixin import LoggingMixin

if TYPE_CHECKING:
    from smartcash.ui.model.evaluation.evaluation_uimodule import EvaluationUIModule


class EvaluationOperationPhase(Enum):
    """Operation phases for progress tracking."""
    INITIALIZING = auto()
    VALIDATING = auto()
    LOADING_MODELS = auto()
    EVALUATING = auto()
    COMPUTING_METRICS = auto()
    FINALIZING = auto()
    COMPLETED = auto()
    FAILED = auto()


class EvaluationBaseOperation(OperationMixin, LoggingMixin):
    """
    Abstract base class for evaluation operation handlers.
    
    Provides common functionality for all evaluation operations including:
    - Dual progress tracking (overall scenarios % and current scenario test %)
    - Error handling and logging
    - Operation lifecycle management
    - Configuration validation
    - Callback execution
    - Backend service integration
    """

    def __init__(
        self, 
        ui_module: 'EvaluationUIModule',
        config: Dict[str, Any], 
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> None:
        """
        Initialize the evaluation operation.
        
        Args:
            ui_module: Reference to the parent UI module
            config: Configuration dictionary for the operation
            callbacks: Optional callbacks for operation events
        """
        # Initialize mixins
        super().__init__()
        
        # Store references
        self._ui_module = ui_module
        self._config = config or {}
        self._callbacks = callbacks or {}
        
        # Operation state
        self._phase = EvaluationOperationPhase.INITIALIZING
        self._start_time = time.time()
        self._operation_id = str(id(self))
        
        # Progress tracking for dual mode
        self._reset_progress_tracking()
        
        # Initialize UI components
        self._initialize_ui_components()

    def _reset_progress_tracking(self) -> None:
        """Reset progress tracking state for a new operation."""
        self._completed_scenarios = set()
        self._current_scenario = None
        self._current_scenario_progress = 0
        self._overall_progress = 0
        self._total_scenarios = 0
        self._current_scenario_index = 0

    def _initialize_ui_components(self) -> None:
        """Initialize UI components for progress tracking."""
        try:
            # Get progress tracker from UI module
            self._progress_tracker = self._ui_module.get_component('progress_tracker')
            
            # Initialize dual progress bars
            if self._progress_tracker:
                # Overall progress: % of total scenarios completed
                self._progress_tracker.reset_progress(
                    overall_label="Overall Evaluation Progress",
                    current_label="Current Scenario Progress"
                )
        except Exception as e:
            self.logger.error(f"Failed to initialize UI components: {e}")

    def update_overall_progress(self, completed_scenarios: int, total_scenarios: int, message: str = "") -> None:
        """
        Update overall progress (% of total scenarios completed).
        
        Args:
            completed_scenarios: Number of completed scenarios
            total_scenarios: Total number of scenarios
            message: Progress message
        """
        try:
            if total_scenarios > 0:
                progress_percent = int((completed_scenarios / total_scenarios) * 100)
                self._overall_progress = progress_percent
                
                if self._progress_tracker:
                    self._progress_tracker.update_overall_progress(
                        progress_percent, 
                        message or f"Completed {completed_scenarios}/{total_scenarios} scenarios"
                    )
                
                # Log to UI
                self._ui_module.log_info(f"📊 Overall: {completed_scenarios}/{total_scenarios} scenarios ({progress_percent}%)")
                
        except Exception as e:
            self.logger.error(f"Failed to update overall progress: {e}")

    def update_current_progress(self, current_percent: int, message: str = "") -> None:
        """
        Update current scenario progress (% of current running scenario test).
        
        Args:
            current_percent: Progress percentage for current scenario (0-100)
            message: Progress message
        """
        try:
            self._current_scenario_progress = current_percent
            
            if self._progress_tracker:
                self._progress_tracker.update_current_progress(
                    current_percent,
                    message or f"Current scenario: {current_percent}%"
                )
            
            # Log to UI for detailed progress
            if current_percent % 25 == 0 or current_percent == 100:  # Log at 25%, 50%, 75%, 100%
                self._ui_module.log_info(f"🔄 Current scenario: {current_percent}% - {message}")
                
        except Exception as e:
            self.logger.error(f"Failed to update current progress: {e}")

    def start_scenario(self, scenario_name: str, scenario_index: int, total_scenarios: int) -> None:
        """
        Start a new scenario evaluation.
        
        Args:
            scenario_name: Name of the scenario being evaluated
            scenario_index: Index of current scenario (0-based)
            total_scenarios: Total number of scenarios
        """
        try:
            self._current_scenario = scenario_name
            self._current_scenario_index = scenario_index
            self._total_scenarios = total_scenarios
            
            # Reset current scenario progress
            self.update_current_progress(0, f"Starting {scenario_name} evaluation...")
            
            # Update overall progress
            self.update_overall_progress(scenario_index, total_scenarios, f"Starting scenario: {scenario_name}")
            
            self._ui_module.log_info(f"🎯 Starting scenario {scenario_index + 1}/{total_scenarios}: {scenario_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to start scenario: {e}")

    def complete_scenario(self, scenario_name: str, success: bool = True) -> None:
        """
        Complete a scenario evaluation.
        
        Args:
            scenario_name: Name of the completed scenario
            success: Whether the scenario completed successfully
        """
        try:
            if success:
                self._completed_scenarios.add(scenario_name)
                self.update_current_progress(100, f"✅ {scenario_name} completed successfully")
                
                # Update overall progress
                completed_count = len(self._completed_scenarios)
                self.update_overall_progress(completed_count, self._total_scenarios, f"Completed: {scenario_name}")
                
                self._ui_module.log_success(f"✅ Scenario completed: {scenario_name}")
            else:
                self.update_current_progress(100, f"❌ {scenario_name} failed")
                self._ui_module.log_error(f"❌ Scenario failed: {scenario_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to complete scenario: {e}")

    def complete_operation(self, success: bool = True, message: str = "") -> None:
        """
        Complete the entire evaluation operation.
        
        Args:
            success: Whether the operation completed successfully
            message: Completion message
        """
        try:
            if success:
                self.update_overall_progress(self._total_scenarios, self._total_scenarios, "🎉 All scenarios completed!")
                self.update_current_progress(100, "✅ Evaluation finished")
                self._ui_module.log_success(f"🎉 Evaluation completed successfully! {message}")
            else:
                self._ui_module.log_error(f"❌ Evaluation failed: {message}")
                
            self._phase = EvaluationOperationPhase.COMPLETED if success else EvaluationOperationPhase.FAILED
            
        except Exception as e:
            self.logger.error(f"Failed to complete operation: {e}")

    def clear_operation_logs(self) -> None:
        """Clear operation logs from the operation container before starting a new operation."""
        try:
            # Try to get operation container from UI module
            operation_container = self._ui_module.get_component('operation_container')
            if operation_container and hasattr(operation_container, 'clear_logs'):
                operation_container.clear_logs()
                self.logger.debug("✅ Evaluation operation logs cleared")
            elif operation_container and hasattr(operation_container, 'clear'):
                operation_container.clear()
                self.logger.debug("✅ Evaluation operation container cleared")
            else:
                self.logger.debug("⚠️ No clear method available on evaluation operation container")
        except Exception as e:
            self.logger.error(f"Failed to clear evaluation operation logs: {e}")

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute the evaluation operation.
        
        Returns:
            Dictionary containing operation results
        """
        pass

    def get_operation_info(self) -> Dict[str, Any]:
        """
        Get information about the current operation.
        
        Returns:
            Dictionary containing operation information
        """
        return {
            'operation_id': self._operation_id,
            'phase': self._phase.name,
            'current_scenario': self._current_scenario,
            'completed_scenarios': len(self._completed_scenarios),
            'total_scenarios': self._total_scenarios,
            'overall_progress': self._overall_progress,
            'current_scenario_progress': self._current_scenario_progress,
            'start_time': self._start_time,
            'elapsed_time': time.time() - self._start_time
        }