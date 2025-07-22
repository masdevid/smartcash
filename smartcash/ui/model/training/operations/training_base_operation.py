"""
File: smartcash/ui/model/training/operations/training_base_operation.py
Description: Base class for training operation handlers.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from smartcash.ui.core.mixins.logging_mixin import LoggingMixin
from smartcash.ui.core.mixins.operation_mixin import OperationMixin

if TYPE_CHECKING:
    from smartcash.ui.model.training.training_uimodule import TrainingUIModule


class BaseTrainingOperation(OperationMixin, LoggingMixin, ABC):
    """
    Abstract base class for training operation handlers.

    Inherits from OperationMixin to provide direct access to UI orchestration
    methods like `update_progress`, `log_operation`, and `update_operation_status`.
    """

    def __init__(self, ui_module: 'TrainingUIModule', config: Dict[str, Any], callbacks: Optional[Dict[str, Callable]] = None):
        # Initialize the OperationMixin
        super().__init__()

        # Store reference to the main UI module to power the OperationMixin methods.
        self._ui_module = ui_module

        # Make the mixin methods work by pointing them to the UI module's components.
        # The mixin expects a logger and ui_components to be present on self.
        self.logger = ui_module.logger
        self._ui_components = getattr(ui_module, '_ui_components', {})
        self._is_initialized = True  # Handlers are instantiated ready to run.

        self.config = config
        self.callbacks = callbacks or {}

    def _get_callback(self, name: str) -> Optional[Callable]:
        """Safely retrieves a callback by name."""
        return self.callbacks.get(name)

    def _execute_callback(self, name: str, *args, **kwargs) -> None:
        """Executes a callback if it exists."""
        callback = self._get_callback(name)
        if callback:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Callback {name} failed: {e}")

    def _progress_adapter(self, level: str, percentage: float, message: str = "") -> None:
        """
        Adapter method to bridge backend progress callbacks to the OperationMixin methods.
        
        This allows backend services to report progress using their format while the UI
        consumes it through the standardized OperationMixin interface.

        Args:
            level: Progress level ('overall' or 'current')
            percentage: Progress percentage (0-100)
            message: Optional progress message
        """
        if hasattr(self, 'update_progress'):
            # Use the correct method signature based on OperationMixin
            # The update_progress method should accept progress, message parameters
            self.update_progress(
                progress=int(percentage),
                message=f"{level.title()}: {message}" if message else level.title()
            )

    # Mock dual progress methods until OperationMixin is available
    def start_dual_progress(self, operation_name: str, total_steps: int) -> None:
        """Mock dual progress start method."""
        self.log_operation(f"🔄 Starting {operation_name} ({total_steps} steps)", 'info')
        self._total_steps = total_steps
        self._current_step = 0

    def update_dual_progress(self, current_step: int, current_percent: float, message: str = "") -> None:
        """Mock dual progress update method."""
        self._current_step = current_step
        overall_percent = (current_step / self._total_steps * 100) if hasattr(self, '_total_steps') and self._total_steps > 0 else 0
        
        if message:
            self.log_operation(f"📊 Step {current_step}/{getattr(self, '_total_steps', '?')}: {message} ({current_percent:.1f}%)", 'info')
        
        # Update UI progress if available
        if hasattr(self, 'update_progress'):
            self.update_progress(
                progress=int(overall_percent),
                message=f"Step {current_step}/{getattr(self, '_total_steps', '?')}"
            )

    def complete_dual_progress(self, message: str = "") -> None:
        """Mock dual progress complete method."""
        final_message = message or "Operation completed"
        self.log_operation(f"✅ {final_message}", 'success')
        
        # Update UI progress to 100%
        if hasattr(self, 'update_progress'):
            self.update_progress(
                progress=100,
                message=final_message
            )

    def error_dual_progress(self, error_message: str) -> None:
        """Mock dual progress error method."""
        self.log_operation(f"❌ {error_message}", 'error')
        
        # Update UI progress with error
        if hasattr(self, 'update_progress'):
            self.update_progress(
                progress=0,
                message=f"Error: {error_message}"
            )

    def update_charts(self, metrics: Dict[str, Any]) -> None:
        """Update live charts with new metrics."""
        try:
            # Get chart update methods from UI module
            if hasattr(self._ui_module, '_chart_updaters'):
                chart_updaters = self._ui_module._chart_updaters
                
                # Update loss chart
                if 'loss_chart' in chart_updaters and callable(chart_updaters['loss_chart']):
                    loss_data = {
                        'train_loss': metrics.get('train_loss', 0.0),
                        'val_loss': metrics.get('val_loss', 0.0),
                        'epoch': metrics.get('epoch', 0)
                    }
                    chart_updaters['loss_chart'](loss_data)
                
                # Update mAP chart
                if 'map_chart' in chart_updaters and callable(chart_updaters['map_chart']):
                    map_data = {
                        'mAP@0.5': metrics.get('mAP@0.5', 0.0),
                        'mAP@0.75': metrics.get('mAP@0.75', 0.0),
                        'epoch': metrics.get('epoch', 0)
                    }
                    chart_updaters['map_chart'](map_data)
                    
                self.logger.debug(f"✅ Charts updated with metrics: {list(metrics.keys())}")
                
        except Exception as e:
            self.logger.warning(f"Failed to update charts: {e}")

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        The main execution method for the operation.
        Subclasses must implement this method to perform their specific logic.
        
        Returns:
            Dict with 'success' bool and 'message' str indicating operation result
        """
        raise NotImplementedError("Subclasses must implement the execute method.")