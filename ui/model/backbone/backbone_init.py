"""
File: smartcash/ui/model/backbone/backbone_init.py
Deskripsi: Initializer untuk Backbone Model Configuration
"""

from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from smartcash.ui.model.backbone.handlers.model_handler import BackboneModelHandler

class BackboneInitializer:
    """
    Initializer untuk Backbone Model Configuration.
    
    Handles initialization of backbone model configuration UI components.
    """
    
    def __init__(self, module_name: str = "backbone", parent_module: str = None):
        """Initialize the backbone model configuration.
        
        Args:
            module_name: Name of the module
            parent_module: Optional parent module name
        """
        self.module_name = module_name
        self.parent_module = parent_module
        self.handler = None
        self.ui_components = {}
        self._callbacks = []
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize the backbone configuration UI.
        
        Args:
            config: Optional initial configuration
            
        Returns:
            Dictionary containing UI components
        """
        # Initialize handler
        self.handler = BackboneModelHandler(
            module_name=self.module_name,
            parent_module=self.parent_module
        )
        
        # Create UI components
        self._create_ui_components()
        
        # Load config if provided
        if config:
            self.handler.load_config(config)
            self.handler.update_ui(self.ui_components, config)
        
        return self.ui_components
    
    def _create_ui_components(self) -> None:
        """Create the UI components for backbone configuration."""
        # Create main container
        self.ui_components['main_container'] = widgets.VBox(layout={'border': '1px solid #cccccc'})
        
        # Add your UI components here
        # Example:
        # self.ui_components['model_select'] = widgets.Dropdown(
        #     options=['resnet50', 'efficientnet', 'mobilenet'],
        #     description='Model:',
        #     disabled=False
        # )
        
        # Add components to container
        # children = [self.ui_components['model_select']]  # Add other components
        # self.ui_components['main_container'].children = children
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback to be called when configuration changes.
        
        Args:
            callback: Function that takes a config dict as argument
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a configuration change callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration.
        
        Returns:
            Current configuration as a dictionary
        """
        if self.handler and hasattr(self.handler, 'extract_config'):
            return self.handler.extract_config(self.ui_components)
        return {}
    
    def update_ui(self, config: Dict[str, Any]) -> None:
        """Update the UI with the given configuration.
        
        Args:
            config: Configuration to apply to the UI
        """
        if self.handler and hasattr(self.handler, 'update_ui'):
            self.handler.update_ui(self.ui_components, config)
    
    def reset(self) -> None:
        """Reset the configuration to defaults."""
        if self.handler and hasattr(self.handler, 'reset_config'):
            self.handler.reset_config(self.ui_components)
    
    def _load_existing_config(self) -> Optional[Dict[str, Any]]:
        """Try to load existing configuration from file"""
        try:
            import os
            import yaml
            
            config_path = os.path.join(os.getcwd(), 'config', 'model_config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception:
            pass
        return None
    
    def create_child_components(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create child components spesifik untuk backbone configuration.
        
        Args:
            config: Configuration dictionary (optional, falls back to self.config)
            
        Returns:
            Dictionary berisi form components dan sections
        """
        from smartcash.ui.model.backbone.components.ui_components import (
            create_backbone_child_components, get_layout_sections
        )
        
        # Use provided config or fall back to instance config
        config = config or self.config
        
        # Create all child components
        child_components = create_backbone_child_components(config)
        
        # Store layout sections for parent to use
        child_components['_layout_sections'] = get_layout_sections(child_components)
        
        return child_components
    
    def create_handler(self) -> BackboneModelHandler:
        """
        Create handler instance untuk backbone operations.
        
        Returns:
            BackboneModelHandler instance
        """
        # Ensure logger_bridge is passed to handler
        if 'logger_bridge' not in self.ui_components:
            self.ui_components['logger_bridge'] = self.logger_bridge
        
        return BackboneModelHandler(self.ui_components)
    
    def get_info_content(self) -> str:
        """
        Get content untuk info accordion.
        
        Returns:
            HTML string dengan informasi backbone model
        """
        from smartcash.ui.info_boxes.model_info import get_model_info_content
        return get_model_info_content()
    
    def get_container_layout(self) -> widgets.Layout:
        """
        Override untuk custom container layout jika diperlukan.
        
        Returns:
            Layout untuk main container
        """
        return widgets.Layout(
            width='100%',
            max_width='1280px',
            margin='0 auto',
            padding='15px',
            border='1px solid #e0e0e0',
            border_radius='8px',
            box_shadow='0 2px 4px rgba(0,0,0,0.05)'
        )