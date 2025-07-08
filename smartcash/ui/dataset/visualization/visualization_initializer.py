"""
File: smartcash/ui/dataset/visualization/visualization_initializer.py
Description: Initializer for the visualization module
"""

from typing import Dict, Any, Optional

# Import core components
from smartcash.ui.core.initializers.base_initializer import BaseInitializer
from smartcash.ui.core.errors import SmartCashUIError

# Import visualization components and handlers
from .handlers import VisualizationUIHandler
from .components.visualization_ui import create_visualization_ui

class VisualizationInitializer(BaseInitializer):
    """Initializer for the visualization module."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the visualization module.
        
        Args:
            config: Optional configuration for initialization
        """
        # Initialize parent class
        super().__init__(module_name='visualization', parent_module='dataset')
        
        # Initialize config
        self._config = {
            'title': 'Data Visualization',
            'description': 'Visualization module for dataset analysis',
            **(config or {})
        }
        
        # Set title and description from config
        self.title = self._config.get('title', 'Data Visualization')
        self.description = self._config.get('description', 'Visualization module for dataset analysis')
        
        # Initialize components
        self.ui_components = {}
        self.handler = None

    def pre_initialize_checks(self) -> None:
        """Perform pre-initialization checks with fail-fast."""
        super().pre_initialize_checks()
        # Add any module-specific pre-initialization checks here
        self.logger.debug("✅ Pre-initialization checks passed")

    def post_initialize_cleanup(self) -> None:
        """Perform post-initialization cleanup."""
        super().post_initialize_cleanup()
        # Add any module-specific cleanup here
        self.logger.debug("✅ Post-initialization cleanup complete")
        
    def _initialize_impl(self, **kwargs) -> Dict[str, Any]:
        """Implementation of the initialization process.
        
        Returns:
            Dict with initialization status and UI components
            
        Raises:
            SmartCashUIError: If initialization fails
        """
        try:
            # Initialize UI components
            self.ui_components = create_visualization_ui()
            
            # Initialize handler
            self.handler = VisualizationUIHandler(
                ui_components=self.ui_components,
                logger=self.logger
            )
            
            # Register handlers
            self._register_handlers()
            
            return {
                'status': 'success',
                'ui_components': self.ui_components,
                'handler': self.handler
            }
            
        except Exception as e:
            error_msg = f"Error initializing visualization module: {str(e)}"
            self.handle_error(error_msg, exc_info=True)
            raise SmartCashUIError(error_msg) from e
    
    def _register_handlers(self) -> None:
        """Register event handlers for UI components."""
        if not self.handler or not self.ui_components:
            return
            
        # Register refresh button handler if it exists
        if 'refresh_button' in self.ui_components:
            self.ui_components['refresh_button'].on_click(self.handler.refresh_data)


def init_visualization_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Initialize visualization UI module.
    
    Args:
        config: Configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of UI components
    """
    initializer = VisualizationInitializer(config=config)
    return initializer._initialize_impl(**kwargs)
