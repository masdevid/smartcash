"""
Configuration handler for dataset split functionality.

This module provides the SplitConfigHandler class which handles configuration
management for the dataset split functionality.
"""

from typing import Dict, Any, Optional
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler
from .config_extractor import extract_split_config
from .config_updater import update_split_ui, reset_ui_to_defaults
from .defaults import get_default_split_config

class SplitConfigHandler(ConfigCellHandler):
    """Handler for dataset split configuration.
    
    This class handles the extraction and application of configuration
    for the dataset split functionality using dedicated extractor and updater modules.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the split config handler.
        
        Args:
            config: Optional initial configuration
        """
        super().__init__(config or {})
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract configuration from UI components using the config_extractor module.
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            Dictionary containing the extracted configuration
        """
        return extract_split_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components with the given configuration using the config_updater module.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration to apply
        """
        update_split_ui(ui_components, config)
    
    def reset_ui(self, ui_components: Dict[str, Any]) -> None:
        """Reset UI components to their default values.
        
        Args:
            ui_components: Dictionary of UI components to reset
        """
        reset_ui_to_defaults(ui_components)
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get the default configuration for dataset splitting.
        
        Returns:
            Dictionary containing default configuration values
        """
        return get_default_split_config()
