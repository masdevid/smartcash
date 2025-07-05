"""
File: smartcash/ui/dataset/augmentation/handlers/operation/base_operation.py
Deskripsi: Base operation handler untuk augmentation module dengan centralized error handling
"""

from typing import Dict, Any, Optional, Tuple, List
import logging

# Import base handler
from smartcash.ui.dataset.augmentation.handlers.base_augmentation_handler import BaseAugmentationHandler

# Import error handling
from smartcash.ui.core.errors.handlers import handle_ui_errors


class BaseOperationHandler(BaseAugmentationHandler):
    """Base operation handler untuk augmentation module dengan centralized error handling
    
    Provides common functionality for all augmentation operation handlers:
    - Centralized error handling
    - Logging in Bahasa Indonesia
    - UI component management
    - Summary panel updates
    - Button state management
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """Initialize base operation handler
        
        Args:
            ui_components: Dictionary berisi komponen UI
        """
        super().__init__(ui_components=ui_components)
        self.logger.debug("BaseOperationHandler initialized")
    
    @handle_ui_errors(log_error=True)
    def execute(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute operation dengan centralized error handling
        
        Args:
            config: Dictionary konfigurasi augmentation
            **kwargs: Additional arguments
            
        Returns:
            Dictionary berisi hasil operasi
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement execute method")
    
    @handle_ui_errors(log_error=True)
    def validate_operation_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate operation config dengan centralized error handling
        
        Args:
            config: Dictionary konfigurasi operation
            
        Returns:
            Tuple berisi (is_valid, message)
        """
        # This method should be implemented by subclasses
        # Default implementation uses validate_config from BaseAugmentationHandler
        return self.validate_config(config)
    
    @handle_ui_errors(log_error=True)
    def prepare_summary(self, result: Dict[str, Any]) -> Tuple[str, str, str]:
        """Prepare summary untuk ditampilkan di summary panel
        
        Args:
            result: Dictionary berisi hasil operasi
            
        Returns:
            Tuple berisi (title, content, status)
        """
        # Default implementation
        status = result.get('status', False)
        message = result.get('message', '')
        
        if status:
            title = "Operasi Berhasil"
            content = f"<div>{message}</div>"
            status_str = "success"
        else:
            title = "Operasi Gagal"
            content = f"<div style='color: #d9534f;'>{message}</div>"
            status_str = "error"
        
        return title, content, status_str
    
    @handle_ui_errors(log_error=True)
    def update_operation_summary(self, result: Dict[str, Any]) -> None:
        """Update summary panel dengan hasil operasi
        
        Args:
            result: Dictionary berisi hasil operasi
        """
        title, content, status = self.prepare_summary(result)
        self.update_status_panel(title, content, status)
