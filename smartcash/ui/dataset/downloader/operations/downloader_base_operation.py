"""
File: smartcash/ui/dataset/downloader/operations/downloader_base_operation.py
Description: Base operation class for downloader operations following the operation pattern.
"""

import time
from pathlib import Path
from abc import abstractmethod
from enum import Enum, auto
from typing import Dict, Any, Callable, Optional, TYPE_CHECKING

from smartcash.ui.core.mixins.operation_mixin import OperationMixin
from smartcash.ui.core.mixins.logging_mixin import LoggingMixin
from smartcash.ui.core.mixins.colab_secrets_mixin import ColabSecretsMixin

if TYPE_CHECKING:
    from smartcash.ui.dataset.downloader.downloader_uimodule import DownloaderUIModule


class DownloaderOperationPhase(Enum):
    """Operation phases for progress tracking."""
    INITIALIZING = auto()
    VALIDATING = auto()
    DOWNLOADING = auto()
    EXTRACTING = auto()
    PROCESSING = auto()
    FINALIZING = auto()
    COMPLETED = auto()
    FAILED = auto()


class DownloaderBaseOperation(OperationMixin, LoggingMixin, ColabSecretsMixin):
    """
    Abstract base class for downloader operation handlers.
    
    Provides common functionality for all downloader operations including:
    - Progress tracking and UI updates
    - Error handling and logging
    - Operation lifecycle management
    - Configuration validation
    - Callback execution
    - Backend service integration
    - API key management via ColabSecretsMixin
    """
    
    # Specific secret names to try for Roboflow API key
    ROBOFLOW_SECRET_NAMES = [
        'ROBOFLOW_API_KEY',
        'roboflow_api_key',
    ]


    def __init__(
        self, 
        ui_module: 'DownloaderUIModule',
        config: Dict[str, Any], 
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> None:
        """
        Initialize the downloader operation.
        
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
        self._phase = DownloaderOperationPhase.INITIALIZING
        self._start_time = time.time()
        self._operation_id = str(id(self))
        
        # Initialize UI components
        self._ui_components = getattr(ui_module, '_ui_components', {})
        
        # Initialize logger if not already set
        if not hasattr(self, 'logger'):
            import logging
            self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
            
        # Initialize ColabSecretsMixin
        ColabSecretsMixin.__init__(self)
        
        # Load backend services
        self._backend_apis = self._load_backend_apis()
        
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
    def phase(self) -> DownloaderOperationPhase:
        """Get current operation phase."""
        return self._phase
    
    @phase.setter
    def phase(self, value: DownloaderOperationPhase) -> None:
        """Set operation phase."""
        self._phase = value
        phase_name = value.name.replace('_', ' ').title()
        self.logger.info(f"Phase changed to: {phase_name}")
    
    def get_roboflow_api_key(self) -> Optional[str]:
        """
        Get the Roboflow API key from available sources.
        
        Returns:
            The Roboflow API key if found, None otherwise.
        """
        return self.get_secret(self.ROBOFLOW_SECRET_NAMES)
    
    def _load_backend_apis(self) -> Dict[str, Any]:
        """
        Load backend API modules for download operations with lazy loading.
        
        Returns:
            Dictionary mapping API names to their implementations with the following keys:
            - 'download_service': Factory function to create a download service
            - 'cleanup_service': Factory function to create a cleanup service
            - 'validator': Factory function to create a dataset validator
            - 'config_validator': Function to validate downloader config
            - 'config_converter': Function to convert UI config to backend format
            
        Note:
            If the backend modules are not available, falls back to empty dict.
            This is logged as a warning.
        """
        try:
            from smartcash.dataset.downloader import (
                get_downloader_instance,
                get_cleanup_service,
                get_dataset_scanner,
                validate_config_quick,
                create_ui_compatible_config
            )
            
            return {
                'download_service': get_downloader_instance,
                'cleanup_service': get_cleanup_service,
                'scanner': get_dataset_scanner,
                'config_validator': validate_config_quick,
                'config_converter': create_ui_compatible_config
            }
        except ImportError as e:
            self.logger.warning(
                "Backend downloader module not available, using fallback: %s",
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
    
    def is_milestone_step(self, step: str, current: int) -> bool:
        """
        Check if the current step represents a milestone for logging.
        
        Args:
            step: The current download step (unused but kept for interface compatibility)
            current: Current progress value for the step
            
        Returns:
            True if this is a milestone step that should be logged
        """
        # Log every 10% progress or at the start/end of a step
        return (current % 10 == 0) or (current == 0) or (current == 100)
    
    def validate_downloader_config(self) -> tuple[Optional[dict], Optional[str]]:
        """
        Validate downloader configuration and convert to backend format.
        
        Returns:
            tuple: (validated_config, error_message) - If validation fails, 
                   validated_config will be None and error_message will contain the reason.
                   If validation passes, error_message will be None.
        """
        try:
            # Get config validator and converter from backend APIs
            config_converter = self.get_backend_api('config_converter')
            config_validator = self.get_backend_api('config_validator')
            
            if not all([config_converter, config_validator]):
                return None, "Configuration validation services are not available."
            
            # Convert UI config to backend format
            backend_config = config_converter(self._config)
            
            # Validate the config
            if not config_validator(backend_config):
                return None, "Invalid download configuration. Missing required parameters."
                
            return backend_config, None
            
        except Exception as e:
            self.logger.error(f"Error validating downloader config: {e}", exc_info=True)
            return None, f"Error validating configuration: {str(e)}"
    
    def log_operation_start(self, operation_name: str) -> None:
        """Log the start of an operation."""
        self._start_time = time.time()
        self.logger.info(f"Memulai {operation_name}...")
        self.phase = DownloaderOperationPhase.PROCESSING
    
    def log_operation_complete(self, operation_name: str) -> None:
        """Log the completion of an operation."""
        duration = time.time() - self._start_time
        self.logger.info(f"{operation_name} selesai dalam {duration:.2f} detik")
        self.phase = DownloaderOperationPhase.COMPLETED
    
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
        self.phase = DownloaderOperationPhase.FAILED
        
        return {
            'status': 'error',
            'message': message,
            'error': error_details,
            'operation_id': self._operation_id
        }
    
    def _execute_callback(self, callback_name: str, *args, **kwargs) -> None:
        """Execute a callback if it exists."""
        callback = self._callbacks.get(callback_name)
        if callback and callable(callback):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error executing callback '{callback_name}': {e}")
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute the operation.
        
        Returns:
            Dictionary containing operation results with at least:
            - success: Boolean indicating operation success
            - message: Summary of operation results
            - details: Additional operation-specific details
        """
        pass
