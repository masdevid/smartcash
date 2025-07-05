"""
File: smartcash/ui/dataset/augmentation/handlers/augmentation_handler.py
Deskripsi: Main augmentation handler untuk augmentation module dengan centralized error handling
"""

from typing import Dict, Any, Optional, Tuple, List
import logging

# Import base handler
from smartcash.ui.dataset.augmentation.handlers.base_augmentation_handler import BaseAugmentationHandler

# Import operation handlers
from smartcash.ui.dataset.augmentation.handlers.operation import (
    AugmentationHandlerManager,
    AugmentOperationHandler,
    CheckOperationHandler,
    CleanupOperationHandler,
    PreviewOperationHandler
)

# Import error handling
from smartcash.ui.core.errors.handlers import handle_ui_errors


class AugmentationHandler(BaseAugmentationHandler):
    """Main augmentation handler untuk augmentation module dengan centralized error handling
    
    Provides functionality for dataset augmentation operations:
    - Centralized error handling
    - Logging in Bahasa Indonesia
    - UI component management
    - Summary panel updates
    - Button state management
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """Initialize augmentation handler
        
        Args:
            ui_components: Dictionary berisi komponen UI
        """
        super().__init__(ui_components=ui_components)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("AugmentationHandler initialized")
        
        # Initialize operation handlers
        self._initialize_operation_handlers()
    
    @handle_ui_errors(log_error=True)
    def _initialize_operation_handlers(self) -> None:
        """Initialize operation handlers dengan centralized error handling"""
        # Create operation handlers
        self.operation_manager = AugmentationHandlerManager(self.ui_components)
        
        # Register operation handlers
        self.operation_manager.register_handler('augment', AugmentOperationHandler(self.ui_components))
        self.operation_manager.register_handler('check', CheckOperationHandler(self.ui_components))
        self.operation_manager.register_handler('cleanup', CleanupOperationHandler(self.ui_components))
        self.operation_manager.register_handler('preview', PreviewOperationHandler(self.ui_components))
    
    @handle_ui_errors(log_error=True)
    def setup_handlers(self) -> None:
        """Setup handlers untuk augmentation module dengan centralized error handling"""
        # Setup button handlers
        self._setup_button_handlers()
        
        # Setup config handler
        self._setup_config_handler()
        
        # Check for existing preview
        self._check_existing_preview()
    
    @handle_ui_errors(log_error=True)
    def _setup_button_handlers(self) -> None:
        """Setup button handlers dengan centralized error handling"""
        # Get buttons
        augment_button = self.ui_components.get('augment_button')
        check_button = self.ui_components.get('check_button')
        cleanup_button = self.ui_components.get('cleanup_button')
        preview_button = self.ui_components.get('preview_button')
        save_button = self.ui_components.get('save_button')
        reset_button = self.ui_components.get('reset_button')
        
        # Setup button handlers
        if augment_button:
            augment_button.on_click(lambda b: self.handle_augment())
        
        if check_button:
            check_button.on_click(lambda b: self.handle_check())
        
        if cleanup_button:
            cleanup_button.on_click(lambda b: self.handle_cleanup())
        
        if preview_button:
            preview_button.on_click(lambda b: self.handle_preview())
        
        if save_button:
            save_button.on_click(lambda b: self.handle_save())
        
        if reset_button:
            reset_button.on_click(lambda b: self.handle_reset())
    
    @handle_ui_errors(log_error=True)
    def _setup_config_handler(self) -> None:
        """Setup config handler dengan centralized error handling"""
        # Get config handler
        config_handler = self.ui_components.get('config_handler')
        
        # Load config
        if config_handler:
            config = config_handler.load_config()
            config_handler.update_ui(self.ui_components, config)
    
    @handle_ui_errors(log_error=True)
    def _check_existing_preview(self) -> None:
        """Check existing preview dengan centralized error handling"""
        # Get config handler
        config_handler = self.ui_components.get('config_handler')
        
        # Get config
        config = config_handler.get_config() if config_handler else {}
        
        # Check for existing preview
        preview_handler = self.operation_manager.get_handler('preview')
        if preview_handler:
            preview_handler.check_and_load_existing_preview(config)
    
    @handle_ui_errors(log_error=True)
    def handle_augment(self) -> Dict[str, Any]:
        """Handle augment operation dengan centralized error handling
        
        Returns:
            Dictionary berisi hasil operasi
        """
        # Get config handler
        config_handler = self.ui_components.get('config_handler')
        
        # Extract config
        config = config_handler.extract_config_from_ui(self.ui_components) if config_handler else {}
        
        # Execute augment operation
        augment_handler = self.operation_manager.get_handler('augment')
        result = augment_handler.execute(config) if augment_handler else {'status': False, 'message': 'Augment handler not found'}
        
        return result
    
    @handle_ui_errors(log_error=True)
    def handle_check(self) -> Dict[str, Any]:
        """Handle check operation dengan centralized error handling
        
        Returns:
            Dictionary berisi hasil operasi
        """
        # Get config handler
        config_handler = self.ui_components.get('config_handler')
        
        # Extract config
        config = config_handler.extract_config_from_ui(self.ui_components) if config_handler else {}
        
        # Execute check operation
        check_handler = self.operation_manager.get_handler('check')
        result = check_handler.execute(config) if check_handler else {'status': False, 'message': 'Check handler not found'}
        
        return result
    
    @handle_ui_errors(log_error=True)
    def handle_cleanup(self) -> Dict[str, Any]:
        """Handle cleanup operation dengan centralized error handling
        
        Returns:
            Dictionary berisi hasil operasi
        """
        # Get config handler
        config_handler = self.ui_components.get('config_handler')
        
        # Extract config
        config = config_handler.extract_config_from_ui(self.ui_components) if config_handler else {}
        
        # Execute cleanup operation
        cleanup_handler = self.operation_manager.get_handler('cleanup')
        result = cleanup_handler.execute(config) if cleanup_handler else {'status': False, 'message': 'Cleanup handler not found'}
        
        return result
    
    @handle_ui_errors(log_error=True)
    def handle_preview(self) -> Dict[str, Any]:
        """Handle preview operation dengan centralized error handling
        
        Returns:
            Dictionary berisi hasil operasi
        """
        # Get config handler
        config_handler = self.ui_components.get('config_handler')
        
        # Extract config
        config = config_handler.extract_config_from_ui(self.ui_components) if config_handler else {}
        
        # Execute preview operation
        preview_handler = self.operation_manager.get_handler('preview')
        result = preview_handler.execute(config) if preview_handler else {'status': False, 'message': 'Preview handler not found'}
        
        return result
    
    @handle_ui_errors(log_error=True)
    def handle_save(self) -> Dict[str, Any]:
        """Handle save operation dengan centralized error handling
        
        Returns:
            Dictionary berisi hasil operasi
        """
        # Get config handler
        config_handler = self.ui_components.get('config_handler')
        
        # Extract config
        config = config_handler.extract_config_from_ui(self.ui_components) if config_handler else {}
        
        # Save config
        if config_handler:
            result = config_handler.save_config(config)
            
            # Log result
            if result.get('status'):
                self.log_message("✅ Konfigurasi berhasil disimpan", "success")
            else:
                self.log_message(f"❌ Gagal menyimpan konfigurasi: {result.get('message', 'Unknown error')}", "error")
            
            return result
        else:
            return {'status': False, 'message': 'Config handler not found'}
    
    @handle_ui_errors(log_error=True)
    def handle_reset(self) -> Dict[str, Any]:
        """Handle reset operation dengan centralized error handling
        
        Returns:
            Dictionary berisi hasil operasi
        """
        # Get config handler
        config_handler = self.ui_components.get('config_handler')
        
        # Reset config
        if config_handler:
            result = config_handler.reset_config()
            
            # Update UI
            config_handler.update_ui(self.ui_components, config_handler.get_config())
            
            # Log result
            if result.get('status'):
                self.log_message("✅ Konfigurasi berhasil direset", "success")
            else:
                self.log_message(f"❌ Gagal mereset konfigurasi: {result.get('message', 'Unknown error')}", "error")
            
            return result
        else:
            return {'status': False, 'message': 'Config handler not found'}
