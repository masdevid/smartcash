"""
File: smartcash/ui/dataset/preprocessing/handlers/operation/manager.py
Deskripsi: Manager untuk preprocessing operations dengan centralized error handling
"""

from typing import Dict, Any, Optional
from smartcash.ui.handlers.error_handler import handle_ui_errors
from smartcash.ui.decorators.ui_decorators import safe_ui_operation
from smartcash.ui.dataset.preprocessing.handlers.base_preprocessing_handler import BasePreprocessingHandler

class PreprocessingHandlerManager(BasePreprocessingHandler):
    """Manager untuk preprocessing operations dengan centralized error handling.
    
    Coordinates all preprocessing operations:
    - Preprocessing execution
    - Dataset checking
    - Cleanup operations
    - UI integration
    """
    
    def __init__(self, ui_components: Optional[Dict[str, Any]] = None):
        """Initialize preprocessing handler manager.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        super().__init__(ui_components=ui_components)
        self._operation_handlers = {}
        
    def setup_handlers(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Setup all operation handlers dan bind ke UI components.
        
        Args:
            ui_components: Dictionary containing UI components
            
        Returns:
            Dictionary of UI components with handlers attached
        """
        self.set_ui_components(ui_components)
        
        # Setup operation handlers
        self._setup_operation_handlers()
        
        # Setup config handlers
        self._setup_config_handlers()
        
        self.logger.info("✅ Preprocessing handlers berhasil disetup")
        return ui_components
    
    @handle_ui_errors(error_component_title="Setup Error", log_error=True)
    def _setup_operation_handlers(self) -> None:
        """Setup operation handlers dengan proper error handling."""
        from .preprocess import PreprocessOperationHandler
        from .check import CheckOperationHandler
        from .cleanup import CleanupOperationHandler
        
        # Create operation handlers
        self._operation_handlers['preprocess'] = PreprocessOperationHandler(self.ui_components)
        self._operation_handlers['check'] = CheckOperationHandler(self.ui_components)
        self._operation_handlers['cleanup'] = CleanupOperationHandler(self.ui_components)
        
        # Bind handlers to buttons
        if preprocess_button := self.ui_components.get('preprocess_button'):
            preprocess_button.on_click(self._handle_preprocess_operation)
            
        if check_button := self.ui_components.get('check_button'):
            check_button.on_click(self._handle_check_operation)
            
        if cleanup_button := self.ui_components.get('cleanup_button'):
            cleanup_button.on_click(self._handle_cleanup_operation)
    
    @handle_ui_errors(error_component_title="Config Handler Setup Error", log_error=True)
    def _setup_config_handlers(self) -> None:
        """Setup config handlers dengan proper error handling."""
        # Get config handler
        config_handler = self.ui_components.get('config_handler')
        if not config_handler:
            self.logger.warning("⚠️ Config handler tidak tersedia")
            return
            
        # Set UI components for config handler
        if hasattr(config_handler, 'set_ui_components'):
            config_handler.set_ui_components(self.ui_components)
            
        # Bind save/reset handlers
        if save_button := self.ui_components.get('save_button'):
            save_button.on_click(self._handle_save_config)
            
        if reset_button := self.ui_components.get('reset_button'):
            reset_button.on_click(self._handle_reset_config)
    
    @safe_ui_operation(error_title="Preprocessing Error")
    def _handle_preprocess_operation(self, button=None) -> None:
        """Handle preprocessing operation dengan proper error handling."""
        handler = self._operation_handlers.get('preprocess')
        if not handler:
            self.logger.error("❌ Preprocess operation handler tidak tersedia")
            return
            
        handler.execute()
    
    @safe_ui_operation(error_title="Check Error")
    def _handle_check_operation(self, button=None) -> None:
        """Handle check operation dengan proper error handling."""
        handler = self._operation_handlers.get('check')
        if not handler:
            self.logger.error("❌ Check operation handler tidak tersedia")
            return
            
        handler.execute()
    
    @safe_ui_operation(error_title="Cleanup Error")
    def _handle_cleanup_operation(self, button=None) -> None:
        """Handle cleanup operation dengan proper error handling."""
        handler = self._operation_handlers.get('cleanup')
        if not handler:
            self.logger.error("❌ Cleanup operation handler tidak tersedia")
            return
            
        handler.execute()
    
    @safe_ui_operation(error_title="Save Config Error")
    def _handle_save_config(self, button=None) -> None:
        """Handle save config operation dengan proper error handling."""
        config_handler = self.ui_components.get('config_handler')
        if not config_handler:
            self.logger.error("❌ Config handler tidak tersedia")
            return
            
        config_handler.save_config()
        
    @safe_ui_operation(error_title="Reset Config Error")
    def _handle_reset_config(self, button=None) -> None:
        """Handle reset config operation dengan proper error handling."""
        config_handler = self.ui_components.get('config_handler')
        if not config_handler:
            self.logger.error("❌ Config handler tidak tersedia")
            return
            
        config_handler.reset_config()


# Factory function
def create_preprocessing_handler_manager(ui_components: Dict[str, Any]) -> PreprocessingHandlerManager:
    """Create preprocessing handler manager instance.
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        PreprocessingHandlerManager instance
    """
    manager = PreprocessingHandlerManager(ui_components)
    return manager
