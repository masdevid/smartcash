"""
File: smartcash/ui/dataset/visualization/operations/visualization_base_operation.py
Description: Kelas dasar operasi untuk operasi visualisasi mengikuti pola operasi.
"""

import time
from abc import abstractmethod
from enum import Enum, auto
from typing import Dict, Any, Callable, Optional, List, Union, TYPE_CHECKING
from pathlib import Path

from smartcash.ui.core.mixins.operation_mixin import OperationMixin
from smartcash.ui.core.mixins.logging_mixin import LoggingMixin

if TYPE_CHECKING:
    from smartcash.ui.dataset.visualization.visualization_uimodule import VisualizationUIModule


class VisualizationOperationPhase(Enum):
    """Fase operasi sederhana untuk operasi visualisasi."""
    STARTED = auto()
    COMPLETED = auto()
    FAILED = auto()


class VisualizationBaseOperation(OperationMixin, LoggingMixin):
    """
    Kelas dasar abstrak untuk penangan operasi visualisasi.
    
    Menyediakan fungsionalitas umum untuk semua operasi visualisasi termasuk:
    - Manajemen siklus operasi
    - Penanganan kesalahan dan pencatatan log
    - Validasi konfigurasi
    - Eksekusi callback
    - Pembaruan UI
    """

    def __init__(
        self, 
        ui_module: 'VisualizationUIModule',
        config: Dict[str, Any], 
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> None:
        """
        Inisialisasi operasi visualisasi.
        
        Args:
            ui_module: Referensi ke modul UI induk
            config: Kamus konfigurasi untuk operasi
            callbacks: Callback opsional untuk event operasi
        """
        # Initialize mixins
        super().__init__()
        
        # Store references
        self._ui_module = ui_module
        self._config = config or {}
        self._callbacks = callbacks or {}
        
        # Operation state
        self._phase = VisualizationOperationPhase.STARTED
        self._start_time = time.time()
        self._operation_id = str(id(self))
        
        # Initialize UI components reference
        self._ui_components = getattr(ui_module, 'components', {})
        
        # Initialize logger if not already set
        if not hasattr(self, 'logger'):
            import logging
            self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
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
    def phase(self) -> VisualizationOperationPhase:
        """Get current operation phase."""
        return self._phase
    
    @phase.setter
    def phase(self, value: VisualizationOperationPhase) -> None:
        """Set operation phase and log the change."""
        self._phase = value
        phase_name = value.name.replace('_', ' ').title()
        self.logger.info(f"Phase changed to: {phase_name}")
        
        # Update UI if available
        if hasattr(self._ui_module, 'update_status'):
            self._ui_module.update_status(f"Status: {phase_name}")
    
    def execute(self) -> bool:
        """
        Execute the visualization operation.
        
        Returns:
            bool: True if the operation completed successfully, False otherwise
        """
        try:
            self.phase = VisualizationOperationPhase.STARTED
            
            # Validate and process
            if not self._validate_config():
                self.phase = VisualizationOperationPhase.FAILED
                return False
                
            # Execute the operation
            data = self._load_data()
            if data is None:
                self.phase = VisualizationOperationPhase.FAILED
                return False
                
            processed_data = self._process_data(data)
            if processed_data is None or not self._render(processed_data):
                self.phase = VisualizationOperationPhase.FAILED
                return False
            
            self.phase = VisualizationOperationPhase.COMPLETED
            return True
            
        except Exception as e:
            self.logger.error(f"Visualization operation failed: {str(e)}", exc_info=True)
            self.phase = VisualizationOperationPhase.FAILED
            return False
    
    @abstractmethod
    def _validate_config(self) -> bool:
        """
        Validate the operation configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def _load_data(self) -> Any:
        """
        Load the data to be visualized.
        
        Returns:
            The loaded data or None if loading failed
        """
        pass
    
    @abstractmethod
    def _process_data(self, data: Any) -> Any:
        """
        Process the data for visualization.
        
        Args:
            data: The data to process
            
        Returns:
            The processed data or None if processing failed
        """
        pass
    
    @abstractmethod
    def _render(self, data: Any) -> bool:
        """
        Render the visualization.
        
        Args:
            data: The data to visualize
            
        Returns:
            bool: True if rendering was successful, False otherwise
        """
        pass
    
    def _execute_callback(self, callback_name: str, *args, **kwargs) -> Any:
        """
        Execute a callback if it exists.
        
        Args:
            callback_name: Name of the callback to execute
            *args: Positional arguments to pass to the callback
            **kwargs: Keyword arguments to pass to the callback
            
        Returns:
            The result of the callback or None if it doesn't exist
        """
        callback = self._callbacks.get(callback_name)
        if callback and callable(callback):
            try:
                return callback(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in callback '{callback_name}': {e}", exc_info=True)
        return None
        
    def _load_backend_apis(self) -> Dict[str, Any]:
        """
        Load backend API modules for visualization operations with lazy loading.
        
        Returns:
            Dictionary mapping API names to their implementations with the following keys:
            - 'samples_service': For getting sample data
            - 'preprocessor_scanner': For scanning preprocessor data
            - 'augmentor_scanner': For scanning augmentor data
            
        Note:
            If the backend modules are not available, falls back to empty dict.
            This is logged as a warning.
        """
        backend_apis = {}
        
        # Try to import preprocessor samples API if available
        try:
            from smartcash.dataset.preprocessor.api import samples_api
            
            # Only add the service if the module has the required functions
            if all(hasattr(samples_api, func) for func in ['get_samples', 'generate_sample_previews', 
                                                         'get_class_samples', 'get_samples_summary']):
                backend_apis['samples_service'] = {
                    'get_samples': samples_api.get_samples,
                    'generate_sample_previews': samples_api.generate_sample_previews,
                    'get_class_samples': samples_api.get_class_samples,
                    'get_samples_summary': samples_api.get_samples_summary
                }
            else:
                self.logger.warning("Preprocessor samples API is missing required functions")
                
            # Try to import file scanner separately
            try:
                from smartcash.dataset.preprocessor.utils import file_scanner
                if hasattr(file_scanner, 'scan_directory'):
                    backend_apis['preprocessor_scanner'] = file_scanner.scan_directory
            except ImportError as e:
                self.logger.debug(f"Preprocessor file scanner not available: {str(e)}")
                
        except ImportError as e:
            self.logger.debug(f"Preprocessor backend module not available: {str(e)}")
        
        # Try to import augmentor scanner if available
        try:
            from smartcash.dataset.augmentor.utils import file_scanner as aug_file_scanner
            
            # Create a wrapper function that matches the expected interface
            def scan_augmentation_wrapper(directory):
                scanner = aug_file_scanner.FileScanner()
                if hasattr(scanner, 'scan_augmented_files'):
                    return scanner.scan_augmented_files(directory)
                return {'success': False, 'message': 'scan_augmented_files method not found', 'files': []}
                
            backend_apis['augmentor_scanner'] = scan_augmentation_wrapper
            
        except ImportError as e:
            self.logger.debug(f"Augmentor backend module not available: {str(e)}")
        except Exception as e:
            self.logger.debug(f"Error initializing augmentor scanner: {str(e)}")
        
        return backend_apis
    
    def get_backend_api(self, api_name: str) -> Optional[Any]:
        """
        Get a backend API by name.
        
        Args:
            api_name: Name of the API to retrieve
            
        Returns:
            The API implementation or None if not available
        """
        return self._backend_apis.get(api_name)
    
    def get_samples(self, data_dir: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Get samples using the preprocessor samples service.
        
        Args:
            data_dir: Directory containing the dataset
            **kwargs: Additional arguments to pass to get_samples
            
        Returns:
            Dictionary containing samples and metadata
        """
        samples_service = self.get_backend_api('samples_service')
        if not samples_service:
            return {'success': False, 'message': 'Samples service not available', 'samples': []}
        
        try:
            return samples_service['get_samples'](data_dir, **kwargs)
        except Exception as e:
            self.logger.error(f"Error getting samples: {e}", exc_info=True)
            return {'success': False, 'message': str(e), 'samples': []}
    
    def scan_preprocessor_directory(self, directory: Union[str, Path]) -> Dict[str, Any]:
        """
        Scan a directory for preprocessor data.
        
        Args:
            directory: Directory to scan
            
        Returns:
            Dictionary containing scan results
        """
        scanner = self.get_backend_api('preprocessor_scanner')
        if not scanner:
            return {'success': False, 'message': 'Preprocessor scanner not available', 'files': []}
        
        try:
            return scanner(directory)
        except Exception as e:
            self.logger.error(f"Error scanning directory: {e}", exc_info=True)
            return {'success': False, 'message': str(e), 'files': []}
    
    def scan_augmentor_directory(self, directory: Union[str, Path]) -> Dict[str, Any]:
        """
        Scan a directory for augmentor data.
        
        Args:
            directory: Directory to scan
            
        Returns:
            Dictionary containing scan results
        """
        scanner = self.get_backend_api('augmentor_scanner')
        if not scanner:
            return {'success': False, 'message': 'Augmentor scanner not available', 'files': []}
        
        try:
            return scanner(directory)
        except Exception as e:
            self.logger.error(f"Error scanning augmentation directory: {e}", exc_info=True)
            return {'success': False, 'message': str(e), 'files': []}
    
    def get_dataset_summary(self, data_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Get a summary of the dataset.
        
        Args:
            data_dir: Directory containing the dataset
            
        Returns:
            Dictionary containing dataset summary
        """
        samples_service = self.get_backend_api('samples_service')
        if not samples_service or 'get_samples_summary' not in samples_service:
            return {'success': False, 'message': 'Samples summary service not available'}
        
        try:
            return samples_service['get_samples_summary'](data_dir)
        except Exception as e:
            self.logger.error(f"Error getting dataset summary: {e}", exc_info=True)
            return {'success': False, 'message': str(e)}
