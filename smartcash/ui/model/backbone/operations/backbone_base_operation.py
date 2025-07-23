"""
File: smartcash/ui/model/backbone/operations/backbone_base_operation.py
Description: Base class for backbone operation handlers.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from smartcash.ui.core.mixins.operation_mixin import OperationMixin
from smartcash.ui.core.mixins.logging_mixin import LoggingMixin

if TYPE_CHECKING:
    from smartcash.ui.model.backbone.backbone_uimodule import BackboneUIModule


class BaseBackboneOperation(OperationMixin, LoggingMixin, ABC):
    """
    Abstract base class for backbone operation handlers.

    Inherits from OperationMixin to provide direct access to UI orchestration
    methods like `update_progress`, `log_operation`, and `update_operation_status`.
    """

    def __init__(self, ui_module: 'BackboneUIModule', config: Dict[str, Any], callbacks: Optional[Dict[str, Callable]] = None):
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
        """Enhanced dual progress update method with proper progress calculation."""
        self._current_step = current_step
        
        # Calculate overall progress: previous steps + current step progress
        if hasattr(self, '_total_steps') and self._total_steps > 0:
            # Progress from completed steps
            completed_steps_progress = ((current_step - 1) / self._total_steps) * 100
            # Progress from current step
            current_step_progress = (current_percent / 100) * (1 / self._total_steps) * 100
            # Combined overall progress
            overall_percent = completed_steps_progress + current_step_progress
            overall_percent = min(100, max(0, overall_percent))  # Clamp between 0-100
        else:
            overall_percent = current_percent
        
        if message:
            self.log_operation(f"📊 Step {current_step}/{getattr(self, '_total_steps', '?')}: {message} ({current_percent:.1f}%)", 'info')
        
        # Update UI progress if available with enhanced progress
        if hasattr(self, 'update_progress'):
            self.update_progress(
                progress=int(overall_percent),
                message=f"Step {current_step}/{getattr(self, '_total_steps', '?')}: {message}" if message else f"Step {current_step}/{getattr(self, '_total_steps', '?')}"
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

    def clear_operation_logs(self) -> None:
        """Clear operation logs from the operation container before starting a new operation."""
        try:
            operation_container = self._ui_components.get('operation_container')
            if not operation_container:
                self.logger.debug("No operation container available for clearing")
                return
                
            # Try multiple clear methods in order of preference
            clear_methods = [
                ('clear_logs', 'Operation logs cleared'),
                ('clear_operations', 'Operation container cleared via clear_operations'),
                ('clear', 'Operation container cleared via clear'),
                ('reset', 'Operation container reset'),
                ('clear_output', 'Operation output cleared')
            ]
            
            for method_name, success_msg in clear_methods:
                if hasattr(operation_container, method_name):
                    try:
                        method = getattr(operation_container, method_name)
                        method()
                        self.logger.debug(f"✅ {success_msg}")
                        return
                    except Exception as method_error:
                        self.logger.debug(f"Failed to use {method_name}: {method_error}")
                        continue
            
            # If operation_container is a dict, try to clear its contents
            if isinstance(operation_container, dict):
                # Check for nested containers that might have clear methods
                for key, container in operation_container.items():
                    if hasattr(container, 'clear'):
                        try:
                            container.clear()
                            self.logger.debug(f"✅ Cleared {key} container")
                            return
                        except Exception:
                            continue
                            
            # If all else fails, just log that we couldn't clear (but don't show as warning)
            self.logger.debug("No suitable clear method found - operation will proceed without clearing")
            
        except Exception as e:
            self.logger.error(f"Failed to clear operation logs: {e}")

    def error_dual_progress(self, error_message: str) -> None:
        """Mock dual progress error method."""
        self.log_operation(f"❌ {error_message}", 'error')
        
        # Update UI progress with error
        if hasattr(self, 'update_progress'):
            self.update_progress(
                progress=0,
                message=f"Error: {error_message}"
            )

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        The main execution method for the operation.
        Subclasses must implement this method to perform their specific logic.
        
        Returns:
            Dict with 'success' bool and 'message' str indicating operation result
        """
        raise NotImplementedError("Subclasses must implement the execute method.")