"""
File: smartcash/ui/dataset/downloader/services/downloader_service.py
Description: Main downloader service that coordinates between different services.
"""

from typing import Dict, Any, Optional, Callable, List, Tuple
import logging
from pathlib import Path

from .core.base_service import BaseService
from . import get_dataset_scanner, get_config_validator, get_secret_manager

class DownloaderService(BaseService):
    """Main service for the downloader module that coordinates between different services.
    
    This service acts as a facade to the underlying services, providing a simplified
    interface for the downloader functionality.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize downloader service with required dependencies.
        
        Args:
            logger: Optional logger instance
        """
        super().__init__(logger)
        self._dataset_scanner = get_dataset_scanner(logger)
        self._config_validator = get_config_validator(logger)
        self._secret_manager = get_secret_manager(logger)
    
    def get_existing_dataset_count(self) -> int:
        """Get the count of existing dataset files.
        
        Returns:
            Number of existing dataset files
        """
        return self._dataset_scanner.get_existing_dataset_count()
    
    def check_existing_dataset(self) -> Tuple[bool, int, Dict[str, Any]]:
        """Check for existing dataset and get summary information.
        
        Returns:
            Tuple containing:
            - has_content: Whether the dataset has any content
            - total_images: Total number of images in the dataset
            - summary: Dictionary with dataset summary information
        """
        return self._dataset_scanner.check_existing_dataset()
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate downloader configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Dictionary with validation results
        """
        return self._config_validator.validate_config(config)
    
    def get_api_key(self) -> Optional[str]:
        """Get the API key from available sources.
        
        Returns:
            API key string or None if not found
        """
        return self._secret_manager.get_api_key()
    
    def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            Dictionary with validation results
        """
        return self._secret_manager.validate_api_key(api_key)
    
    def get_roboflow_config(self) -> Dict[str, Any]:
        """Get default Roboflow configuration.
        
        Returns:
            Dictionary with default Roboflow configuration
        """
        return {
            "valid": False,
            "errors": ["Not implemented yet"],
            "warnings": []
        }
    
    @staticmethod
    def is_milestone_step(step: str, progress: int) -> bool:
        """Check if this is a milestone step that should be logged.
        
        Args:
            step: Current step name
            progress: Current progress percentage
            
        Returns:
            True if this is a milestone step that should be logged
        """
        # Only log major milestones to prevent browser crash
        if progress % 10 == 0 or progress in (0, 100):
            return True
            
        # Always log step changes
        if progress == 0 and step:
            return True
            
        return False
    
    @classmethod
    def map_step_to_current_progress(cls, step: str, overall_progress: int) -> int:
        """Map step progress to current operation progress bar.
        
        Args:
            step: Current step name
            overall_progress: Overall progress percentage (0-100)
            
        Returns:
            Mapped progress percentage (0-100)
        """
        # Define step weights (sum should be 100)
        step_weights = {
            'preparing': 5,
            'downloading': 80,
            'extracting': 10,
            'verifying': 5
        }
        
        # Default to even distribution if step not found
        step_weight = step_weights.get(step.lower(), 100 // len(step_weights))
        
        # Map progress within the step's weight range
        step_progress = min(100, max(0, overall_progress))
        return (step_progress * step_weight) // 100
    
    def create_progress_callback(self, ui_components: Dict[str, Any]) -> Callable[[str, int, int, str], None]:
        """Create progress callback for operations.
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            Progress callback function
        """
        def progress_callback(step: str, current: int, total: int, message: str):
            try:
                progress_tracker = ui_components.get('progress_tracker')
                if progress_tracker:
                    # Update both overall and current operation progress
                    if hasattr(progress_tracker, 'update_overall'):
                        progress_tracker.update_overall(current, message)
                    if hasattr(progress_tracker, 'update_current'):
                        # Map step to current operation progress
                        step_progress = self.map_step_to_current_progress(step, current)
                        progress_tracker.update_current(step_progress, f"Step: {step}")
                
                # Only log important milestones, not every progress update
                if self.is_milestone_step(step, current):
                    logger = ui_components.get('logger')
                    if logger:
                        logger.info(message)
            except Exception as e:
                self.logger.error(f"Error in progress callback: {e}")
        
        return progress_callback
    
    def get_api_key_from_secrets(self) -> Optional[str]:
        """Get API key from Colab secrets.
        
        Returns:
            API key if found, None otherwise
        """
        try:
            from smartcash.ui.dataset.downloader.services import get_secret_manager
            secret_manager = get_secret_manager()
            return secret_manager.get_api_key()
        except Exception as e:
            self.logger.error(f"Error getting API key from secrets: {e}")
            return None
    
    def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Validate API key format.
        
        Args:
            api_key: API key to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            from smartcash.ui.dataset.downloader.services import get_secret_manager
            secret_manager = get_secret_manager()
            return secret_manager.validate_api_key(api_key)
        except Exception as e:
            self.logger.error(f"Error validating API key: {e}")
            return {
                'valid': False,
                'message': f"API key validation error: {str(e)}"
            }
    
    def format_validation_summary(self, validation: Dict[str, Any], html_format: bool = False) -> str:
        """Format validation summary for display.
        
        Args:
            validation: Validation results dictionary
            html_format: Whether to format as HTML
            
        Returns:
            Formatted validation summary string
        """
        try:
            from smartcash.ui.dataset.downloader.services import get_config_validator
            validator = get_config_validator()
            return validator.format_validation_summary(validation, html_format)
        except Exception as e:
            self.logger.error(f"Error formatting validation summary: {e}")
            return f"Error formatting summary: {str(e)}"
    
    def get_ui_components_status(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of UI components.
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            Dictionary with component status
        """
        try:
            from smartcash.ui.dataset.downloader.services.ui_utils import get_ui_components_status
            return get_ui_components_status(ui_components)
        except Exception as e:
            self.logger.error(f"Error getting UI components status: {e}")
            return {
                'valid': False,
                'errors': [f"UI components status error: {str(e)}"]
            }