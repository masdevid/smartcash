"""
File: smartcash/ui/dataset/preprocessing/operations/base_operation.py
Description: Base class for preprocessing operation handlers.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from smartcash.ui.core.mixins.operation_mixin import OperationMixin
from ....components.progress_tracker.types import ProgressLevel

if TYPE_CHECKING:
    from smartcash.ui.dataset.preprocessing.preprocessing_uimodule import PreprocessingUIModule


class BasePreprocessingOperation(OperationMixin, ABC):
    """
    Abstract base class for preprocessing operation handlers.

    Inherits from OperationMixin to provide direct access to UI orchestration
    methods like `update_progress`, `log_operation`, and `update_operation_status`.
    """

    def __init__(self, ui_module: 'PreprocessingUIModule', config: Dict[str, Any], callbacks: Optional[Dict[str, Callable]] = None):
        # Initialize the OperationMixin
        super().__init__()

        # Store reference to the main UI module to power the OperationMixin methods.
        self._ui_module = ui_module

        # Make the mixin methods work by pointing them to the UI module's components.
        # The mixin expects a logger and ui_components to be present on self.
        self.logger = ui_module.logger
        self._ui_components = getattr(ui_module, '_ui_components', {})
        self._operation_manager = getattr(ui_module, '_operation_manager', None)
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
                self.logger.error(f"Error executing callback '{name}': {e}")

    def _progress_adapter(self, level: str, current: int, total: int, message: str, secondary_current: Optional[int] = None, secondary_total: Optional[int] = None, secondary_message: str = ''):
        """Adapts a backend progress callback to the UI's ProgressTracker with dual progress support."""
        # Calculate main progress percentage
        percentage = (current / total) * 100 if total > 0 else 0
        
        # Handle dual progress if secondary parameters are provided
        if secondary_current is not None and secondary_total is not None and secondary_total > 0:
            secondary_percentage = (secondary_current / secondary_total) * 100
            
            # For dual progress: main progress for overall, secondary for current step
            if level == 'overall':
                # Overall progress as main, current step as secondary
                self.update_progress(
                    progress=int(percentage),
                    message=message,
                    secondary_progress=int(secondary_percentage), 
                    secondary_message=secondary_message
                )
            else:
                # Current progress as main, overall as secondary (alternative layout)
                self.update_progress(
                    progress=int(secondary_percentage),
                    message=secondary_message,
                    secondary_progress=int(percentage),
                    secondary_message=message
                )
        else:
            # Single progress mode (backward compatibility)
            progress_level = ProgressLevel.OVERALL if level == 'overall' else ProgressLevel.CURRENT
            self.update_progress(progress_level, percentage, message)

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        The main execution method for the operation.
        Subclasses must implement this method to perform their specific logic.
        
        Returns:
            Dict with 'success' bool and 'message' str indicating operation result
        """
        raise NotImplementedError("Subclasses must implement the execute method.")
