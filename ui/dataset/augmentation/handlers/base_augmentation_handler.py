"""
File: smartcash/ui/dataset/augmentation/handlers/base_augmentation_handler.py
Deskripsi: Base handler untuk augmentation module dengan centralized error handling
"""

from typing import Dict, Any, Optional, Tuple
import logging
import ipywidgets as widgets

# Import base handler
from smartcash.ui.core.handlers import BaseHandler

# Import error handling
from smartcash.ui.core.errors.handlers import handle_ui_errors


class BaseAugmentationHandler(BaseHandler):
    """Base handler untuk augmentation module dengan centralized error handling
    
    Provides common functionality for all augmentation handlers:
    - Centralized error handling
    - Logging in Bahasa Indonesia
    - UI component management
    - Summary panel updates
    - Button state management
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """Initialize base augmentation handler
        
        Args:
            ui_components: Dictionary berisi komponen UI
        """
        super().__init__(module_name="dataset.augmentation.handlers.base")
        
        # Store UI components
        self.ui_components = ui_components or {}
        
        # Setup log redirection if UI components provided
        if ui_components:
            self.setup_log_redirection(ui_components)
            
        self.logger.debug("BaseAugmentationHandler initialized")
    
    @handle_ui_errors(log_error=True)
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate augmentation config dengan centralized error handling
        
        Args:
            config: Dictionary konfigurasi augmentation
            
        Returns:
            Tuple berisi (is_valid, message)
        """
        # Check if config has required keys
        if not config:
            return False, "Konfigurasi tidak boleh kosong"
        
        # Check if config has augmentation section
        if 'augmentation' not in config:
            return False, "Konfigurasi tidak memiliki section 'augmentation'"
        
        # Check if augmentation section has required keys
        aug_config = config['augmentation']
        required_keys = ['enabled', 'methods']
        
        for key in required_keys:
            if key not in aug_config:
                return False, f"Konfigurasi augmentation tidak memiliki key '{key}'"
        
        # Check if methods is a list
        if not isinstance(aug_config['methods'], list):
            return False, "Konfigurasi augmentation methods harus berupa list"
        
        return True, "Konfigurasi valid"
