"""
File: smartcash/ui/dataset/downloader/operations/download_factory.py
Description: Factory functions for creating downloader operations.
"""

from typing import Dict, Any, Callable, Optional, TYPE_CHECKING, Union

from .download_operation import DownloadOperation
from .download_check_operation import DownloadCheckOperation  
from .download_cleanup_operation import DownloadCleanupOperation

if TYPE_CHECKING:
    from smartcash.ui.dataset.downloader.downloader_uimodule import DownloaderUIModule


class DownloaderOperationFactory:
    """
    Factory class for creating downloader operations.
    
    Provides centralized creation and management of downloader operation instances
    with consistent configuration and callback handling.
    """
    
    # Operation type mapping
    OPERATION_TYPES = {
        'download': DownloadOperation,
        'check': DownloadCheckOperation,
        'cleanup': DownloadCleanupOperation
    }

    @classmethod
    def create_operation(
        cls,
        operation_type: str,
        ui_module: 'DownloaderUIModule',
        config: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> Union[DownloadOperation, DownloadCheckOperation, DownloadCleanupOperation]:
        """
        Create a downloader operation instance.
        
        Args:
            operation_type: Type of operation ('download', 'check', 'cleanup')
            ui_module: Reference to the parent UI module
            config: Configuration dictionary for the operation
            callbacks: Optional callbacks for operation events
            
        Returns:
            Operation instance of the specified type
            
        Raises:
            ValueError: If operation_type is not supported
        """
        if operation_type not in cls.OPERATION_TYPES:
            supported_types = ', '.join(cls.OPERATION_TYPES.keys())
            raise ValueError(f"Unsupported operation type: {operation_type}. Supported types: {supported_types}")
        
        operation_class = cls.OPERATION_TYPES[operation_type]
        return operation_class(ui_module, config, callbacks)

    @classmethod
    def execute_operation(
        cls,
        operation_type: str,
        ui_module: 'DownloaderUIModule',
        config: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create and execute a downloader operation.
        
        Args:
            operation_type: Type of operation ('download', 'check', 'cleanup')
            ui_module: Reference to the parent UI module
            config: Configuration dictionary for the operation
            callbacks: Optional callbacks for operation events
            **kwargs: Additional arguments passed to the operation execute method
            
        Returns:
            Dictionary with operation results
            
        Raises:
            ValueError: If operation_type is not supported
        """
        operation = cls.create_operation(operation_type, ui_module, config, callbacks)
        return operation.execute(**kwargs)


# Convenience factory functions for direct operation creation

def create_download_operation(
    ui_module: 'DownloaderUIModule',
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None
) -> DownloadOperation:
    """
    Create a download operation instance.
    
    Args:
        ui_module: Reference to the parent UI module
        config: Configuration dictionary for the download
        callbacks: Optional callbacks for operation events
        
    Returns:
        DownloadOperation instance
    """
    return DownloadOperation(ui_module, config, callbacks)


def create_check_operation(
    ui_module: 'DownloaderUIModule',
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None
) -> DownloadCheckOperation:
    """
    Create a check operation instance.
    
    Args:
        ui_module: Reference to the parent UI module
        config: Configuration dictionary for the check
        callbacks: Optional callbacks for operation events
        
    Returns:
        DownloadCheckOperation instance
    """
    return DownloadCheckOperation(ui_module, config, callbacks)


def create_cleanup_operation(
    ui_module: 'DownloaderUIModule',
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None
) -> DownloadCleanupOperation:
    """
    Create a cleanup operation instance.
    
    Args:
        ui_module: Reference to the parent UI module
        config: Configuration dictionary for the cleanup
        callbacks: Optional callbacks for operation events
        
    Returns:
        DownloadCleanupOperation instance
    """
    return DownloadCleanupOperation(ui_module, config, callbacks)


# Direct execution functions

def execute_download_operation(
    ui_module: 'DownloaderUIModule',
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None
) -> Dict[str, Any]:
    """
    Create and execute a download operation.
    
    Args:
        ui_module: Reference to the parent UI module
        config: Configuration dictionary for the download
        callbacks: Optional callbacks for operation events
        
    Returns:
        Dictionary with operation results
    """
    operation = create_download_operation(ui_module, config, callbacks)
    return operation.execute()


def execute_check_operation(
    ui_module: 'DownloaderUIModule',
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None
) -> Dict[str, Any]:
    """
    Create and execute a check operation.
    
    Args:
        ui_module: Reference to the parent UI module
        config: Configuration dictionary for the check
        callbacks: Optional callbacks for operation events
        
    Returns:
        Dictionary with operation results
    """
    operation = create_check_operation(ui_module, config, callbacks)
    return operation.execute()


def execute_cleanup_operation(
    ui_module: 'DownloaderUIModule',
    config: Dict[str, Any],
    callbacks: Optional[Dict[str, Callable]] = None,
    targets_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create and execute a cleanup operation.
    
    Args:
        ui_module: Reference to the parent UI module
        config: Configuration dictionary for the cleanup
        callbacks: Optional callbacks for operation events
        targets_result: Optional pre-scanned targets result
        
    Returns:
        Dictionary with operation results
    """
    operation = create_cleanup_operation(ui_module, config, callbacks)
    return operation.execute(targets_result=targets_result)


# Utility functions

def get_supported_operations() -> list[str]:
    """
    Get list of supported operation types.
    
    Returns:
        List of supported operation type strings
    """
    return list(DownloaderOperationFactory.OPERATION_TYPES.keys())


def is_valid_operation_type(operation_type: str) -> bool:
    """
    Check if an operation type is supported.
    
    Args:
        operation_type: Operation type to check
        
    Returns:
        True if operation type is supported, False otherwise
    """
    return operation_type in DownloaderOperationFactory.OPERATION_TYPES