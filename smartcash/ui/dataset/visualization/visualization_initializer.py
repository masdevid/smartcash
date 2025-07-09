"""
File: smartcash/ui/dataset/visualization/visualization_initializer.py
Description: Initializer for the visualization module
"""

from typing import Dict, Any, Optional

# Import core components
from smartcash.ui.core.initializers.display_initializer import DisplayInitializer
from smartcash.ui.core.errors import SmartCashUIError

# Import visualization components and handlers
from .handlers import VisualizationUIHandler
from .components.visualization_ui import create_visualization_ui

class VisualizationInitializer(DisplayInitializer):
    """Initializer for the visualization module with display capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the visualization module.
        
        Args:
            config: Optional configuration for initialization
        """
        # Initialize parent class with module info
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

    def create_ui_components(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Create and return the UI components for the visualization module.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components
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
            
            return self.ui_components
            
        except Exception as e:
            error_msg = f"Error creating visualization UI components: {str(e)}"
            self.handle_error(error_msg, exc_info=True)
            raise SmartCashUIError(error_msg) from e
    
    def _register_handlers(self) -> None:
        """Register event handlers for UI components."""
        if not getattr(self, 'handler', None) or not getattr(self, 'ui_components', None):
            return
            
        # Register refresh button handler if it exists
        if 'refresh_button' in self.ui_components and hasattr(self.handler, 'refresh_data'):
            self.ui_components['refresh_button'].on_click(self.handler.refresh_data)


def init_visualization_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
    """
    Initialize and display the visualization UI.
    
    This function uses DisplayInitializer to handle the UI display with proper
    error handling and logging.
    
    Args:
        config: Configuration dictionary
        **kwargs: Additional arguments passed to the initializer
    """
    initializer = VisualizationInitializer(config=config)
    initializer.initialize_ui(**kwargs)
