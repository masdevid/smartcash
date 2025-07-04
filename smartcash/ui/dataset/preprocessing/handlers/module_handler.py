"""
File: smartcash/ui/dataset/preprocessing/handlers/module_handler.py
Deskripsi: Module handler untuk preprocessing dengan ModuleUIHandler integration.
"""

from typing import Dict, Any, Optional, List, Callable
from smartcash.ui.core.handlers.ui_handler import ModuleUIHandler
from smartcash.ui.handlers.error_handler import handle_ui_errors
from smartcash.ui.decorators.ui_decorators import safe_ui_operation


class PreprocessingModuleHandler(ModuleUIHandler):
    """Module handler untuk preprocessing dengan ModuleUIHandler integration.
    
    Features:
    - 🎯 Complete UI integration dengan core architecture
    - 🔧 Operation handler management
    - 📊 Status dan progress tracking
    - 🔄 UI-Config synchronization
    """
    
    def __init__(self, 
                 module_name: str = "preprocessing", 
                 parent_module: str = "dataset",
                 default_config: Optional[Dict[str, Any]] = None,
                 auto_setup_handlers: bool = True,
                 enable_sharing: bool = True):
        """Initialize preprocessing module handler.
        
        Args:
            module_name: Nama module
            parent_module: Parent module
            default_config: Default configuration
            auto_setup_handlers: Auto setup UI event handlers
            enable_sharing: Enable config sharing
        """
        super().__init__(module_name, parent_module)
        
        # Operation handlers registry
        self._operation_handlers: Dict[str, Any] = {}
        
        # Config
        self._default_config = default_config or {}
        self._auto_setup_handlers = auto_setup_handlers
        self._enable_sharing = enable_sharing
        
        self.logger.debug(f"🎯 PreprocessingModuleHandler initialized")
    
    def setup(self, ui_components: Dict[str, Any]) -> None:
        """Setup module handler dengan UI components.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        self.ui_components = ui_components
        
        # Setup operation handlers
        if self._auto_setup_handlers:
            self._setup_operation_handlers()
            self._setup_button_handlers()
        
        self.logger.info("✅ Preprocessing module handler setup complete")
    
    @handle_ui_errors(error_component_title="Operation Handler Setup Error", log_error=True)
    def _setup_operation_handlers(self) -> None:
        """Setup operation handlers dengan proper error handling."""
        from smartcash.ui.dataset.preprocessing.operations.preprocess import PreprocessOperationHandler
        from smartcash.ui.dataset.preprocessing.operations.check import CheckOperationHandler
        from smartcash.ui.dataset.preprocessing.operations.cleanup import CleanupOperationHandler
        
        # Create operation handlers
        self._operation_handlers['preprocess'] = PreprocessOperationHandler(self.ui_components)
        self._operation_handlers['check'] = CheckOperationHandler(self.ui_components)
        self._operation_handlers['cleanup'] = CleanupOperationHandler(self.ui_components)
        
        self.logger.info(f"✅ Setup {len(self._operation_handlers)} operation handlers")
    
    @handle_ui_errors(error_component_title="Button Handler Setup Error", log_error=True)
    def _setup_button_handlers(self) -> None:
        """Setup button handlers dengan proper error handling."""
        # Bind handlers to buttons
        if preprocess_button := self.ui_components.get('preprocess_button'):
            preprocess_button.on_click(self._handle_preprocess_operation)
            
        if check_button := self.ui_components.get('check_button'):
            check_button.on_click(self._handle_check_operation)
            
        if cleanup_button := self.ui_components.get('cleanup_button'):
            cleanup_button.on_click(self._handle_cleanup_operation)
            
        # Config buttons
        if save_button := self.ui_components.get('save_button'):
            save_button.on_click(self._handle_save_config)
            
        if reset_button := self.ui_components.get('reset_button'):
            reset_button.on_click(self._handle_reset_config)
            
        self.logger.info("✅ Button handlers setup complete")
    
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
    
    def get_operation_handler(self, name: str) -> Optional[Any]:
        """Get operation handler by name.
        
        Args:
            name: Operation handler name
            
        Returns:
            Operation handler instance or None
        """
        return self._operation_handlers.get(name)
    
    def execute_operation(self, name: str) -> Dict[str, Any]:
        """Execute operation by name.
        
        Args:
            name: Operation name
            
        Returns:
            Operation result
        """
        handler = self.get_operation_handler(name)
        if not handler:
            self.logger.error(f"❌ Operation handler '{name}' tidak tersedia")
            return {'status': False, 'error': f"Operation handler '{name}' tidak tersedia"}
            
        return handler.execute()
    
    def extract_config_from_ui(self) -> Dict[str, Any]:
        """Extract config dari UI components.
        
        Returns:
            Extracted configuration dictionary
        """
        config_handler = self.ui_components.get('config_handler')
        if not config_handler:
            self.logger.warning("⚠️ Config handler tidak tersedia untuk extract config")
            return {}
            
        # Extract config dari UI
        if hasattr(config_handler, 'extract_config'):
            try:
                return config_handler.extract_config(self.ui_components)
            except Exception as e:
                self.logger.error(f"❌ Error extracting config: {str(e)}")
                return {}
        else:
            self.logger.warning("⚠️ Config handler tidak memiliki extract_config method")
            return {}
    
    def update_ui_from_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Update UI components dari config.
        
        Args:
            config: Configuration dictionary to apply
        """
        config_handler = self.ui_components.get('config_handler')
        if not config_handler:
            self.logger.warning("⚠️ Config handler tidak tersedia untuk update UI")
            return
            
        # Use provided config or get from handler
        config_to_apply = config or getattr(config_handler, 'config', {})
        
        # Update UI dari config
        if hasattr(config_handler, 'update_ui'):
            try:
                config_handler.update_ui(self.ui_components, config_to_apply)
            except Exception as e:
                self.logger.error(f"❌ Error updating UI: {str(e)}")
        else:
            self.logger.warning("⚠️ Config handler tidak memiliki update_ui method")
    
    def get_module_info(self) -> Dict[str, Any]:
        """Get module information summary.
        
        Returns:
            Dictionary with module information
        """
        return {
            'module': f"{self.parent_module}.{self.module_name}",
            'operations': list(self._operation_handlers.keys()),
            'ui_components': len(self.ui_components),
            'status_history': len(self._status_history),
            'progress_history': len(self._progress_history)
        }
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        # Cleanup operation handlers
        for handler in self._operation_handlers.values():
            if hasattr(handler, 'cleanup'):
                try:
                    handler.cleanup()
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to cleanup handler: {e}")
        
        # Clear operation handlers
        self._operation_handlers.clear()
        
        # Call parent cleanup
        super().cleanup()
        
        self.logger.info("🧹 Preprocessing module handler cleanup complete")


# Factory function
def create_preprocessing_module_handler(ui_components: Dict[str, Any]) -> PreprocessingModuleHandler:
    """Create preprocessing module handler instance.
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        PreprocessingModuleHandler instance
    """
    handler = PreprocessingModuleHandler()
    handler.setup(ui_components)
    return handler
