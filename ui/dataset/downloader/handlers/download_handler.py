"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py
Deskripsi: Orchestrator for dataset downloader operations.

This module provides the main entry point for setting up all downloader handlers.
It acts as an orchestrator that delegates to specialized handler classes.
"""
from typing import Dict, Any, Optional

# Import operation handlers
from .operations.download import DownloadOperation
from .operations.cleanup import CleanupOperation
from .operations.config import ConfigOperation


class DownloadHandler:
    """Orchestrator for dataset downloader operations.
    
    This class acts as an orchestrator that delegates to specialized handler classes
    for download, cleanup, and configuration operations.
    """

    def __init__(self, ui_components: Dict[str, Any], config: Dict[str, Any]):
        """Initialize with UI components and configuration."""
        self.ui_components = ui_components
        self.config = config
        self.operations = {}
        
        # Initialize operation handlers
        self._init_operation_handlers()
    
    def _init_operation_handlers(self) -> None:
        """Initialize all operation handlers."""
        # Initialize operation handlers
        self.operations['download'] = DownloadOperation(self.ui_components)
        self.operations['cleanup'] = CleanupOperation(self.ui_components)
        self.operations['config'] = ConfigOperation(self.ui_components)
        
        # Setup operation handlers
        self._setup_operation_handlers()
    
    def _setup_operation_handlers(self) -> None:
        """Setup all operation handlers."""
        # Setup download handler
        self.operations['download'].setup_download_handler(self.config)
        
        # Setup cleanup handler
        self.operations['cleanup'].setup_cleanup_handler(self.config)
        
        # Setup config handlers
        self.operations['config'].setup_config_handlers()
    
    def get_operation_handler(self, operation_name: str) -> Optional[Any]:
        """Get operation handler by name."""
        return self.operations.get(operation_name)
    
    def execute_operation(self, operation_name: str, *args, **kwargs) -> Any:
        """Execute an operation by name."""
        handler = self.get_operation_handler(operation_name)
        if handler and hasattr(handler, f"_execute_{operation_name}"):
            return getattr(handler, f"_execute_{operation_name}")(*args, **kwargs)
        return None
    



def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup all downloader handlers.
    
    This is the main entry point that initializes all handlers.
    """
    # Initialize progress callback
    from smartcash.ui.dataset.downloader.utils.progress_utils import create_progress_callback
    ui_components['progress_callback'] = create_progress_callback(ui_components)
    
    # Initialize download handler as orchestrator
    if '_downloader_orchestrator' not in ui_components:
        handler = DownloadHandler(ui_components, config)
        ui_components['_downloader_orchestrator'] = handler
    
    return ui_components


def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Setup check handler with backend scanner.
    
    This is a backward compatibility wrapper that delegates to the new handler.
    """
    if '_downloader_orchestrator' not in ui_components:
        setup_download_handlers(ui_components, config)


def setup_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Setup cleanup handler with confirmation dialog.
    
    This is a backward compatibility wrapper that delegates to the new handler.
    """
    if '_downloader_orchestrator' not in ui_components:
        setup_download_handlers(ui_components, config)


def setup_config_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup config handlers for save and reset buttons.
    
    This is a backward compatibility wrapper that delegates to the new handler.
    """
    if '_downloader_orchestrator' not in ui_components:
        # We need a config dict for the orchestrator
        setup_download_handlers(ui_components, {})