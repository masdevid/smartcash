"""
File: smartcash/ui/dataset/downloader/services/downloader_service.py
Deskripsi: Consolidated downloader service with backend bridge functionality
"""

from typing import Dict, Any, Optional, Callable, List
import os
import logging
from pathlib import Path

class DownloaderService:
    """Consolidated service for downloader module with backend bridge functionality."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize downloader service.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def get_existing_dataset_count(self) -> int:
        """Get existing dataset count.
        
        Returns:
            Number of existing dataset files
        """
        try:
            from smartcash.ui.dataset.downloader.services.backend_utils import get_existing_dataset_count
            return get_existing_dataset_count(self.logger)
        except Exception as e:
            self.logger.error(f"Error getting dataset count: {e}")
            return 0
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate downloader configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            from smartcash.ui.dataset.downloader.services.validation_utils import validate_config
            return validate_config(config)
        except Exception as e:
            self.logger.error(f"Error validating config: {e}")
            return {
                'valid': False,
                'status': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': []
            }
    
    def create_progress_callback(self, progress_tracker) -> Callable:
        """Create progress callback for operations.
        
        Args:
            progress_tracker: Progress tracker widget
            
        Returns:
            Progress callback function
        """
        try:
            from smartcash.ui.dataset.downloader.handlers.utils import create_progress_callback
            return create_progress_callback(progress_tracker)
        except Exception as e:
            self.logger.error(f"Error creating progress callback: {e}")
            return lambda *args, **kwargs: None
    
    def get_api_key_from_secrets(self) -> Optional[str]:
        """Get API key from Colab secrets.
        
        Returns:
            API key if found, None otherwise
        """
        try:
            from smartcash.ui.dataset.downloader.services.colab_secrets import get_api_key_from_secrets
            return get_api_key_from_secrets()
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
            from smartcash.ui.dataset.downloader.services.colab_secrets import validate_api_key
            return validate_api_key(api_key)
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
            from smartcash.ui.dataset.downloader.services.validation_utils import format_validation_summary
            return format_validation_summary(validation, html_format)
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