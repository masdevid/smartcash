"""
File: smartcash/ui/dataset/augmentation/configs/augmentation_config_handler.py
Description: Pure mixin-based augmentation config handler using core ConfigurationMixin.
Uses composition over inheritance for better flexibility and testability.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from smartcash.ui.core.mixins.configuration_mixin import ConfigurationMixin
from smartcash.ui.core.mixins.logging_mixin import LoggingMixin
from smartcash.ui.logger import get_module_logger
from .augmentation_defaults import get_default_augmentation_config


class AugmentationConfigHandler(LoggingMixin, ConfigurationMixin):
    """
    Pure mixin-based config handler for augmentation management.
    
    Uses composition over inheritance - no BaseHandler inheritance chain.
    This follows the mixin pattern used throughout the UI module system.
    """
    
    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        """
        Initialize augmentation config handler using core mixins.
        
        Args:
            default_config: Optional default configuration
        """
        # Initialize mixins
        LoggingMixin.__init__(self, logger_name=__name__)
        ConfigurationMixin.__init__(self)
        
        # Set default config if not provided
        self._default_config = default_config or get_default_augmentation_config()
        
        # Initialize with default config
        self._config = self._default_config.copy()
        
        # Track config history for undo/redo
        self._config_history: List[Dict[str, Any]] = [self._config.copy()]
        self._history_index = 0
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self._config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the configuration with new values.
        
        Args:
            new_config: Dictionary with new configuration values
        """
        if not isinstance(new_config, dict):
            self.logger.warning("Invalid config format, expected dictionary")
            return
            
        # Update config with new values
        self._config.update(new_config)
        
        # Add to history
        self._add_to_history()
        
        self.logger.debug("Configuration updated")
    
    def reset_config(self) -> None:
        """Reset configuration to default values."""
        self._config = self._default_config.copy()
        self._add_to_history()
        self.logger.info("Configuration reset to defaults")
    
    def _add_to_history(self) -> None:
        """Add current config to history for undo/redo functionality."""
        # Remove any redo history
        self._config_history = self._config_history[:self._history_index + 1]
        
        # Add current config to history
        self._config_history.append(self._config.copy())
        self._history_index = len(self._config_history) - 1
        
        # Limit history size
        if len(self._config_history) > 50:  # Keep last 50 states
            self._config_history.pop(0)
            self._history_index -= 1
    
    def undo(self) -> bool:
        """
        Revert to previous configuration state.
        
        Returns:
            bool: True if undo was successful, False if no more history
        """
        if self._history_index > 0:
            self._history_index -= 1
            self._config = self._config_history[self._history_index].copy()
            self.logger.debug("Undo to previous configuration state")
            return True
        return False
    
    def redo(self) -> bool:
        """
        Redo next configuration state.
        
        Returns:
            bool: True if redo was successful, False if no more history
        """
        if self._history_index < len(self._config_history) - 1:
            self._history_index += 1
            self._config = self._config_history[self._history_index].copy()
            self.logger.debug("Redo to next configuration state")
            return True
        return False
    
    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate the configuration.
        
        Args:
            config: Optional config to validate (uses current config if None)
            
        Returns:
            bool: True if config is valid, False otherwise
        """
        cfg = config or self._config
        
        # Add validation logic here
        try:
            # Example validation
            if 'augmentation_type' not in cfg:
                self.logger.error("Missing required field: augmentation_type")
                return False
                
            if 'intensity' in cfg and not (0 <= cfg['intensity'] <= 1):
                self.logger.error("Intensity must be between 0 and 1")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return False
    
    def get_augmentation_pipeline(self) -> List[Dict[str, Any]]:
        """
        Get the current augmentation pipeline configuration.
        
        Returns:
            List of augmentation operations with their parameters
        """
        return self._config.get('pipeline', [])
    
    def add_augmentation_step(self, step: Dict[str, Any]) -> bool:
        """
        Add an augmentation step to the pipeline.
        
        Args:
            step: Dictionary containing augmentation step configuration
            
        Returns:
            bool: True if step was added successfully, False otherwise
        """
        if not isinstance(step, dict) or 'type' not in step:
            self.logger.error("Invalid augmentation step format")
            return False
            
        if 'pipeline' not in self._config:
            self._config['pipeline'] = []
            
        self._config['pipeline'].append(step)
        self._add_to_history()
        self.logger.info(f"Added augmentation step: {step.get('type')}")
        return True
    
    def remove_augmentation_step(self, index: int) -> bool:
        """
        Remove an augmentation step from the pipeline.
        
        Args:
            index: Index of the step to remove
            
        Returns:
            bool: True if step was removed successfully, False otherwise
        """
        if 'pipeline' not in self._config or not isinstance(self._config['pipeline'], list):
            self.logger.error("No pipeline exists or invalid format")
            return False
            
        if not (0 <= index < len(self._config['pipeline'])):
            self.logger.error(f"Invalid step index: {index}")
            return False
            
        removed = self._config['pipeline'].pop(index)
        self._add_to_history()
        self.logger.info(f"Removed augmentation step: {removed.get('type')}")
        return True
    
    def move_augmentation_step(self, from_index: int, to_index: int) -> bool:
        """
        Move an augmentation step in the pipeline.
        
        Args:
            from_index: Current index of the step
            to_index: Target index for the step
            
        Returns:
            bool: True if step was moved successfully, False otherwise
        """
        if 'pipeline' not in self._config or not isinstance(self._config['pipeline'], list):
            self.logger.error("No pipeline exists or invalid format")
            return False
            
        pipeline = self._config['pipeline']
        if not (0 <= from_index < len(pipeline) and 0 <= to_index < len(pipeline)):
            self.logger.error("Invalid source or target index")
            return False
            
        if from_index == to_index:
            return True  # No-op
            
        # Move the item
        item = pipeline.pop(from_index)
        pipeline.insert(to_index, item)
        
        self._add_to_history()
        self.logger.debug(f"Moved augmentation step from index {from_index} to {to_index}")
        return True
