"""
Orchestrator for dataset downloader handlers.
"""
from typing import Dict, Any, Optional
from .operations.download import DownloadOperation
from .operations.cleanup import CleanupOperation
from .operations.config import ConfigOperation
from .confirmation import confirmation_handler


class DownloaderOrchestrator:
    """Orchestrator for dataset downloader operations."""
    
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


def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup all downloader handlers."""
    # Initialize progress callback
    from smartcash.ui.dataset.downloader.utils.progress_utils import create_progress_callback
    ui_components['progress_callback'] = create_progress_callback(ui_components)
    
    # Initialize orchestrator
    orchestrator = DownloaderOrchestrator(ui_components, config)
    
    # Store orchestrator reference in UI components
    ui_components['_downloader_orchestrator'] = orchestrator
    
    return ui_components
