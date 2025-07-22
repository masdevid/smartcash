"""
Base operation class for augmentation operations following the dependency module pattern.

This module provides a base class for all augmentation operations with common functionality
for progress tracking, error handling, and UI integration.
"""

import time
from pathlib import Path
from abc import abstractmethod
from enum import Enum, auto
from typing import Dict, Any, Callable, Optional, TYPE_CHECKING

from smartcash.ui.core.mixins.operation_mixin import OperationMixin
from smartcash.ui.core.mixins.logging_mixin import LoggingMixin

if TYPE_CHECKING:
    from smartcash.ui.dataset.augmentation.augmentation_uimodule import AugmentationUIModule


class OperationPhase(Enum):
    """Operation phases for progress tracking."""
    INITIALIZING = auto()
    VALIDATING = auto()
    PROCESSING = auto()
    FINALIZING = auto()
    COMPLETED = auto()
    FAILED = auto()


class AugmentationBaseOperation(OperationMixin, LoggingMixin):
    """
    Abstract base class for augmentation operation handlers.

    Provides common functionality for all augmentation operations including:
    - Progress tracking and UI updates
    - Error handling and logging
    - Operation lifecycle management
    - Configuration validation
    - Callback execution
    """

    def __init__(
        self, 
        ui_module: 'AugmentationUIModule',
        config: Dict[str, Any], 
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> None:
        """
        Initialize the augmentation operation.
        
        Args:
            ui_module: Reference to the parent UI module
            config: Configuration dictionary for the operation
            callbacks: Optional callbacks for operation events
        """
        # Initialize parent classes
        OperationMixin.__init__(self)
        LoggingMixin.__init__(self)
        
        # Initialize instance attributes
        self._ui_module = ui_module
        self._config = config or {}
        self._callbacks = callbacks or {}
        self._ui_components = getattr(ui_module, '_ui_components', {})
        self._phase = OperationPhase.INITIALIZING
        self._current_phase = OperationPhase.INITIALIZING
        
        # Initialize common attributes
        self._last_update = None
        self._operation_id = str(id(self))
        self._start_time = time.time()
        self._progress = 0.0
        self._is_running = False
        self._results: Dict[str, Any] = {}
        self._errors: list[str] = []
        
        # Initialize backend APIs
        self._backend_apis = self._load_backend_apis()
        
        # Initialize logger if not already set
        if not hasattr(self, 'logger'):
            import logging
            self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Register operation with UI module if possible
        if hasattr(ui_module, 'register_operation'):
            ui_module.register_operation(self)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the operation configuration."""
        return self._config
    
    @property
    def operation_id(self) -> str:
        """Get the unique operation ID."""
        return self._operation_id
    
    @property
    def phase(self) -> OperationPhase:
        """Get the current operation phase."""
        return self._phase
    
    def update_operation_status(self, message: str, level: str = 'info') -> None:
        """
        Update operation status with the given message and level.
        
        Args:
            message: Status message to display
            level: Status level ('info', 'success', 'warning', 'error')
        """
        if level == 'success':
            self._set_phase(OperationPhase.COMPLETED, message)
        elif level == 'error':
            self._set_phase(OperationPhase.FAILED, message)
        else:
            self.log_info(message)
    
    def _set_phase(self, phase: OperationPhase, message: str = '') -> None:
        """
        Set the current operation phase and log the transition.
        
        Args:
            phase: New operation phase
            message: Optional message describing the phase change
        """
        self._phase = phase
        phase_name = phase.name.lower().replace('_', ' ').title()
        log_message = f"{phase_name}: {message}" if message else phase_name
        
        if phase == OperationPhase.FAILED:
            self.log_error(log_message)
        elif phase == OperationPhase.COMPLETED:
            self.log_success(log_message)
        else:
            self.log_info(log_message)
    
    def log_operation_start(self, operation_name: str) -> None:
        """Log the start of an operation."""
        self._start_time = time.time()
        self.log_info(f"Memulai {operation_name}...")
        self._set_phase(OperationPhase.PROCESSING)
    
    def log_operation_complete(self, operation_name: str) -> None:
        """Log the completion of an operation."""
        duration = time.time() - self._start_time
        self.log_success(f"{operation_name} selesai dalam {duration:.2f} detik")
        self._set_phase(OperationPhase.COMPLETED)
    
    def _handle_error(self, message: str, exception: Optional[Exception] = None) -> Dict[str, Any]:
        """
        Handle an error during operation execution.
        
        Args:
            message: Error message
            exception: Optional exception that caused the error
            
        Returns:
            Error response dictionary
        """
        error_details = str(exception) if exception else message
        self.log_error(f"{message}: {error_details}")
        self._set_phase(OperationPhase.FAILED, message)
        
        return {
            'status': 'error',
            'message': message,
            'error': error_details,
            'operation_id': self._operation_id
        }
    
    def get_backend_api(self, api_name: str) -> Optional[Callable]:
        """
        Get a backend API function by name.
        
        Args:
            api_name: Name of the API function to retrieve
            
        Returns:
            Callable API function or None if not found
        """
        backend = self._config.get('backend')
        if not backend or not hasattr(backend, 'get_api'):
            return None
        return backend.get_api(api_name)
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute the operation.
        
        Returns:
            Dictionary containing operation results
        """
        pass
    
    
    def _progress_adapter(self, level: str, current: int, total: int, message: str) -> None:
        """Adapts a backend progress callback to the UI's ProgressTracker.
        
        Args:
            level: Progress level ('overall' or 'current')
            current: Current progress value
            total: Total progress value
            message: Progress message
        """
        # Convert to percentage (0-100)
        if total > 0:
            percentage = (current / total) * 100
        else:
            percentage = 0
            
        # Update progress in the UI
        if level == 'overall':
            self.update_progress(percentage, message)
        else:
            # For current progress, we could use a secondary progress bar if available
            self.update_progress(percentage, message)
    
    # Core operation methods
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute the main operation logic.
        
        Returns:
            Dictionary containing operation results with at least:
            - success: Boolean indicating operation success
            - message: Summary of operation results
            - details: Additional operation-specific details
        """
        raise NotImplementedError("Subclasses must implement the execute method")
    
    def _init_ui_components(self) -> None:
        """Initialize UI components and progress tracking."""
        self._ui_components = getattr(self._ui_module, '_ui_components', {})
        self._operation_manager = getattr(self._ui_module, '_operation_manager', None)
    
    def _load_backend_apis(self) -> Dict[str, Any]:
        """
        Load backend API modules for augmentation with lazy loading.
        
        Subclasses should override this to provide their specific backend APIs.
        
        Returns:
            Dictionary mapping API names to their implementations with the following keys:
            - 'run_pipeline': Main function for running the pipeline
            - 'get_samples': Function to get sample data
            - 'cleanup': Function to clean up temporary files
            
        Note:
            If the backend module is not available, falls back to empty dict.
            This is logged as a warning.
        """
        try:
            from smartcash.dataset.augmentor.service import create_augmentation_service
            from smartcash.dataset.augmentor import AugmentationService
            
            # Create service instance first
            service_instance = create_augmentation_service(self._config)
            
            if service_instance is None:
                # Fallback to direct service creation
                service_instance = AugmentationService(self._config)
            
            return {
                # Main pipeline execution function
                'run_pipeline': service_instance.run_augmentation_pipeline,  # Runs full augmentation + normalization pipeline
                'status': service_instance.get_augmentation_status,
                'preview': service_instance.create_live_preview,     # Create service instance for preview
                'cleanup': service_instance.cleanup_data, # Cleanup temporary files
            }
        except ImportError as e:
            self.logger.warning(
                "Backend augmentation module not available, using fallback: %s",
                str(e)
            )
            return {}
    
    def get_backend_api(self, api_name: str) -> Optional[Any]:
        """
        Get a backend API by name.
        
        Args:
            api_name: Name of the API to retrieve
            
        Returns:
            The API implementation or None if not available
        """
        return self._backend_apis.get(api_name)
    
    def _update_progress(self, progress: float, message: str) -> None:
        """
        Update operation progress and notify UI using OperationMixin's progress tracking.
        
        Args:
            progress: Progress value between 0 and 100
            message: Status message to display
        """
        self._progress = max(0, min(100, progress))  # Clamp between 0-100
        
        # Use OperationMixin's update_progress method
        self.update_progress(int(self._progress), message)
        
        # Execute progress callback if provided
        self._execute_callback('on_progress', self._progress, message)
    
    
    def _execute_callback(self, callback_name: str, *args, **kwargs) -> Any:
        """
        Execute a named callback if it exists.
        
        Args:
            callback_name: Name of the callback to execute
            *args: Positional arguments to pass to the callback
            **kwargs: Keyword arguments to pass to the callback
            
        Returns:
            The result of the callback or None if it doesn't exist
        """
        callback = self._callbacks.get(callback_name)
        if callable(callback):
            try:
                return callback(*args, **kwargs)
            except Exception as e:
                self.logger.error("Error in callback %s: %s", callback_name, str(e))
        return None
    
    def _log_operation_start(self, operation_name: str) -> None:
        """
        Log the start of an operation and initialize progress tracking.
        
        Args:
            operation_name: Name of the operation being started
        """
        self._start_time = time.time()
        self._is_running = True
        
        # Initialize progress tracking with combined log and status update
        start_message = f"🚀 Memulai {operation_name}..."
        self.start_progress(start_message, progress=0)
        self.log_with_status(
            message=start_message,
            status_message=start_message,
            status_level='info',
            log_level='info'
        )
    
    def _log_operation_complete(self, operation_name: str) -> None:
        """
        Log the successful completion of an operation and complete progress tracking.
        
        Args:
            operation_name: Name of the completed operation
        """
        duration = time.time() - self._start_time
        complete_message = f"✅ {operation_name} selesai dalam {duration:.2f} detik"
        
        # Complete progress tracking with combined log and status update
        self.complete_progress(complete_message)
        self.log_with_status(
            message=complete_message,
            status_message=complete_message,
            status_level='success',
            log_level='info'
        )
        
        self._is_running = False
        self._execute_callback('on_success')
    
    def _log_operation_error(self, operation_name: str, error: Exception) -> None:
        """
        Log an operation error and update progress tracking.
        
        Args:
            operation_name: Name of the failed operation
            error: Exception that caused the failure
        """
        error_msg = str(error)
        error_display = f"❌ Gagal {operation_name}: {error_msg}"
        
        # Update progress and log with combined status
        self.error_progress(error_display)
        self.log_with_status(
            message=error_display,
            status_message=error_display,
            status_level='error',
            log_level='error'
        )
        
        self._errors.append(error_msg)
        self._is_running = False
        self._execute_callback('on_error', error)
    
    def _get_output_dir(self, subdir: str = '') -> Path:
        """
        Get the output directory path, optionally with a subdirectory.
        
        Args:
            subdir: Optional subdirectory to append to the base output directory
            
        Returns:
            Path object for the output directory
        """
        base_dir = Path(self._config.get('output_dir', './output'))
        return base_dir / subdir if subdir else base_dir
    
    def _ensure_output_dir(self, subdir: str = '') -> Path:
        """
        Ensure the output directory exists and return its path.
        
        Args:
            subdir: Optional subdirectory to create
            
        Returns:
            Path object for the created directory
        """
        output_dir = self._get_output_dir(subdir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current operation status.
        
        Returns:
            Dictionary containing:
            - progress: Current progress percentage
            - phase: Current operation phase
            - is_running: Whether the operation is currently running
            - errors: List of error messages if any
        """
        return {
            'progress': self._progress,
            'phase': self._current_phase.name,
            'is_running': self._is_running,
            'errors': self._errors.copy()
        }

