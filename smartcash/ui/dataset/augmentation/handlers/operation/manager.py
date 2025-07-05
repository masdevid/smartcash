"""
File: smartcash/ui/dataset/augmentation/handlers/operation/manager.py
Deskripsi: Manager untuk augmentation operation handlers dengan centralized error handling
"""

from typing import Dict, Any, Optional, List, Callable
import logging

# Import base handler
from smartcash.ui.dataset.augmentation.handlers.base_augmentation_handler import BaseAugmentationHandler

# Import error handling
from smartcash.ui.core.errors.handlers import handle_ui_errors


class AugmentationHandlerManager(BaseAugmentationHandler):
    """Manager untuk augmentation operation handlers dengan centralized error handling
    
    Provides integration for all augmentation operation handlers:
    - Centralized error handling
    - Logging in Bahasa Indonesia
    - UI component management
    - Summary panel updates
    - Button state management
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """Initialize augmentation handler manager
        
        Args:
            ui_components: Dictionary berisi komponen UI
        """
        super().__init__(module_name="augmentation.handlers.operation.manager")
        
        # Store UI components
        self.ui_components = ui_components or {}
        
        # Setup log redirection if UI components provided
        if ui_components:
            self.setup_log_redirection(ui_components)
            
        self.logger.debug("AugmentationHandlerManager initialized")
        
        # Initialize operation handlers
        self._initialize_operation_handlers()
    
    @handle_ui_errors(log_error=True)
    def _initialize_operation_handlers(self) -> None:
        """Initialize operation handlers"""
        from .augment import AugmentOperationHandler
        from .check import CheckOperationHandler
        from .cleanup import CleanupOperationHandler
        
        # Create operation handlers
        self.augment_handler = AugmentOperationHandler(self.ui_components)
        self.check_handler = CheckOperationHandler(self.ui_components)
        self.cleanup_handler = CleanupOperationHandler(self.ui_components)
    
    @handle_ui_errors(log_error=True)
    def setup_handlers(self, config: Dict[str, Any]) -> None:
        """Setup handlers dengan centralized error handling
        
        Args:
            config: Dictionary konfigurasi augmentation
        """
        # Setup config handler dengan UI integration
        config_handler = self.ui_components.get('config_handler')
        if config_handler and hasattr(config_handler, 'set_ui_components'):
            config_handler.set_ui_components(self.ui_components)
        
        # Initialize dialog visibility state
        self.ui_components['_dialog_visible'] = False
        
        # Setup handlers
        self._setup_operation_handlers(config)
        self._setup_config_handlers(config)
        self._setup_preview_handler(config)
        
        self.logger.info("Semua handler berhasil disetup")
    
    @handle_ui_errors(log_error=True)
    def _setup_operation_handlers(self, config: Dict[str, Any]) -> None:
        """Setup operation handlers dengan centralized error handling
        
        Args:
            config: Dictionary konfigurasi augmentation
        """
        # Bind handlers dengan error handling
        handlers = {
            'augment_button': self._create_augment_handler(),
            'check_button': self._create_check_handler(),
            'cleanup_button': self._create_cleanup_handler()
        }
        
        self._bind_handlers_safe(handlers)
    
    @handle_ui_errors(log_error=True)
    def _create_augment_handler(self) -> Callable:
        """Create augment handler function
        
        Returns:
            Callable: Augment handler function
        """
        def augment_handler(btn=None):
            # Get config from UI
            config_handler = self.ui_components.get('config_handler')
            config = config_handler.extract_config_from_ui(self.ui_components) if config_handler else {}
            
            # Execute augment operation
            self.augment_handler.execute(config)
        
        return augment_handler
    
    @handle_ui_errors(log_error=True)
    def _create_check_handler(self) -> Callable:
        """Create check handler function
        
        Returns:
            Callable: Check handler function
        """
        def check_handler(btn=None):
            # Get config from UI
            config_handler = self.ui_components.get('config_handler')
            config = config_handler.extract_config_from_ui(self.ui_components) if config_handler else {}
            
            # Execute check operation
            self.check_handler.execute(config)
        
        return check_handler
    
    @handle_ui_errors(log_error=True)
    def _create_cleanup_handler(self) -> Callable:
        """Create cleanup handler function
        
        Returns:
            Callable: Cleanup handler function
        """
        def cleanup_handler(btn=None):
            # Get config from UI
            config_handler = self.ui_components.get('config_handler')
            config = config_handler.extract_config_from_ui(self.ui_components) if config_handler else {}
            
            # Execute cleanup operation
            self.cleanup_handler.execute(config)
        
        return cleanup_handler
    
    @handle_ui_errors(log_error=True)
    def _setup_config_handlers(self, config: Dict[str, Any]) -> None:
        """Setup config handlers dengan centralized error handling
        
        Args:
            config: Dictionary konfigurasi augmentation
        """
        from smartcash.ui.dataset.augmentation.utils.config_handlers import handle_save_config, handle_reset_config
        
        # Create handler functions yang return proper event handlers
        save_handler = handle_save_config(self.ui_components)
        reset_handler = handle_reset_config(self.ui_components)
        
        config_handlers = {
            'save_button': save_handler,
            'reset_button': reset_handler
        }
        
        self._bind_handlers_safe(config_handlers)
    
    @handle_ui_errors(log_error=True)
    def _setup_preview_handler(self, config: Dict[str, Any]) -> None:
        """Setup preview handler dengan centralized error handling
        
        Args:
            config: Dictionary konfigurasi augmentation
        """
        from .preview import PreviewOperationHandler
        
        # Create preview handler
        self.preview_handler = PreviewOperationHandler(self.ui_components)
        
        def preview_handler(btn=None):
            # Execute preview operation
            self.preview_handler.execute(config)
        
        # Bind preview handler
        preview_handlers = {
            'preview_button': preview_handler
        }
        
        self._bind_handlers_safe(preview_handlers)
        
        # Check and load existing preview
        self.preview_handler.check_and_load_existing_preview(config)
    
    @handle_ui_errors(log_error=True)
    def _bind_handlers_safe(self, handlers: Dict[str, Callable]) -> None:
        """Safe handler binding dengan centralized error handling
        
        Args:
            handlers: Dictionary berisi handler functions
        """
        for button_key, handler in handlers.items():
            button = self.ui_components.get(button_key)
            if button and hasattr(button, 'on_click'):
                try:
                    button.on_click(handler)
                except Exception as e:
                    # Use logger from BaseHandler
                    self.logger.warning(f"Error binding {button_key}: {str(e)}")
                    
                    # Update status panel if available
                    if self.ui_components:
                        self.update_status_panel(
                            self.ui_components,
                            f"⚠️ Error binding {button_key}: {str(e)}",
                            "warning"
                        )
