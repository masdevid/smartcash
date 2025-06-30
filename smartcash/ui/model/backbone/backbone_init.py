"""
File: smartcash/ui/model/backbone/backbone_init.py
Deskripsi: Initializer untuk Backbone Model Configuration yang extends ConfigCellInitializer
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from smartcash.ui.initializers.config_cell_initializer import ConfigCellInitializer
from smartcash.ui.model.backbone.handlers.model_handler import BackboneModelHandler

class BackboneInitializer(ConfigCellInitializer):
    """
    Initializer untuk Backbone Model Configuration.
    
    Extends ConfigCellInitializer dan hanya membuat child components spesifik.
    Parent components (header, status panel, log accordion) sudah dibuat oleh parent class.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize dengan config untuk backbone model"""
        # Try to load existing config first
        loaded_config = self._load_existing_config()
        
        # Get default config from config handler
        from .handlers.config_handler import BackboneConfigHandler
        default_config = BackboneConfigHandler.get_default_config()
        
        # Priority: provided config > loaded config > default config
        if loaded_config and 'model' in loaded_config:
            # Deep update model config from loaded config
            for key, value in loaded_config['model'].items():
                if key in default_config['model']:
                    if isinstance(value, dict) and isinstance(default_config['model'][key], dict):
                        default_config['model'][key].update(value)
                    else:
                        default_config['model'][key] = value
        
        # Apply provided config (highest priority)
        if config and 'model' in config:
            for key, value in config['model'].items():
                if key in default_config['model']:
                    if isinstance(value, dict) and isinstance(default_config['model'][key], dict):
                        default_config['model'][key].update(value)
                    else:
                        default_config['model'][key] = value
        
        # Initialize parent dengan merged config
        super().__init__(
            config=default_config,
            component_id='backbone',
            parent_id='model',
            title='Model Configuration',
            description='Konfigurasi backbone model YOLOv5 dengan EfficientNet-B4',
            icon='🤖'
        )
    
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