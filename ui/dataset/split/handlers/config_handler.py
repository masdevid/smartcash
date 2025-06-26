"""
Configuration handler for dataset split functionality.

This module provides the SplitConfigHandler class which handles configuration
management for the dataset split functionality.
"""

from typing import Dict, Any, Optional, Tuple
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
        
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate the configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ratios = config.get('data', {}).get('split_ratios', {})
            train = ratios.get('train', 0)
            valid = ratios.get('valid', 0)
            test = ratios.get('test', 0)
            
            if not self.validate_split_ratios(train, valid, test):
                return False, "Split ratios must sum to 1.0"
                
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get the default configuration for this handler.
        
        Returns:
            Dictionary containing default configuration
        """
        return get_default_split_config()
        
    def validate_split_ratios(self, train: float, valid: float, test: float) -> bool:
        """Validate that the split ratios sum to 1.0.
        
        Args:
            train: Training set ratio
            valid: Validation set ratio
            test: Test set ratio
            
        Returns:
            bool: True if ratios are valid, False otherwise
        """
        return abs((train + valid + test) - 1.0) < 0.001
        
    def normalize_split_ratios(self, train: float, valid: float, test: float) -> Tuple[float, float, float]:
        """Normalize the split ratios to sum to 1.0.
        
        Args:
            train: Training set ratio
            valid: Validation set ratio
            test: Test set ratio
            
        Returns:
            Tuple of (train, valid, test) ratios that sum to 1.0
        """
        total = train + valid + test
        if abs(total - 1.0) < 0.001:
            return train, valid, test
        return train/total, valid/total, test/total
