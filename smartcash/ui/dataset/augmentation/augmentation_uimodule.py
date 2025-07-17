"""
File: smartcash/ui/dataset/augmentation/augmentation_uimodule.py
Description: Augmentation Module implementation using BaseUIModule mixin pattern.
"""

from typing import Dict, Any, Optional, List

# BaseUIModule imports
from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.core.decorators import suppress_ui_init_logs

# Augmentation module imports
from .components.augmentation_ui import create_augmentation_ui_components
from .configs.augmentation_config_handler import AugmentationConfigHandler
from .configs.augmentation_defaults import get_default_augmentation_config


class AugmentationUIModule(BaseUIModule):
    """
    Augmentation Module implementation using BaseUIModule with environment support.
    
    Features:
    - 🖼️ Image augmentation operations
    - 🎛️ Configurable augmentation pipelines
    - 🔄 Batch processing support
    - 📊 Preview functionality
    - ✅ Full compliance with OPERATION_CHECKLISTS.md requirements
    - 🇮🇩 Bahasa Indonesia interface
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize the Augmentation UIModule.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional arguments passed to BaseUIModule
        """
        # Initialize base class with merged config
        merged_config = get_default_augmentation_config()
        if config:
            merged_config.update(config)
            
        super().__init__(config=merged_config, **kwargs)
        
        # Initialize module components
        self._init_components()
        
        # Set up UI event handlers
        self._setup_event_handlers()
    
    def _init_components(self) -> None:
        """Initialize all module components."""
        # Initialize config handler
        self.config_handler = AugmentationConfigHandler(
            default_config=self.config
        )
        
        # Create UI components
        self.ui_components = create_augmentation_ui_components()
        
        # Initialize operation manager
        self._init_operation_manager()
    
    def _init_operation_manager(self) -> None:
        """Initialize the operation manager for augmentation operations."""
        from .operations.factory import AugmentationOperationFactory
        
        self.operation_factory = AugmentationOperationFactory()
        self.operation_manager = self.operation_factory.create_operation_manager(
            config_handler=self.config_handler
        )
    
    def _setup_event_handlers(self) -> None:
        """Set up UI event handlers."""
        # Connect UI component events to handlers
        if hasattr(self.ui_components, 'apply_button'):
            self.ui_components.apply_button.on_click(self._on_apply_clicked)
            
        if hasattr(self.ui_components, 'preview_button'):
            self.ui_components.preview_button.on_click(self._on_preview_clicked)
    
    def _on_apply_clicked(self, button) -> None:
        """Handle apply button click event."""
        try:
            # Get current configuration from UI
            config = self._get_ui_config()
            
            # Update config handler
            self.config_handler.update_config(config)
            
            # Execute augmentation operation
            operation = self.operation_factory.create_operation(
                'augment',
                config=config
            )
            
            # Run operation asynchronously
            self.run_operation(operation)
            
        except Exception as e:
            self.logger.error(f"Error applying augmentation: {str(e)}")
            self.show_error(f"Gagal menerapkan augmentasi: {str(e)}")
    
    def _on_preview_clicked(self, button) -> None:
        """Handle preview button click event."""
        try:
            # Get current configuration from UI
            config = self._get_ui_config()
            
            # Create and execute preview operation
            operation = self.operation_factory.create_operation(
                'preview',
                config=config
            )
            
            # Run preview operation
            self.run_operation(operation)
            
        except Exception as e:
            self.logger.error(f"Preview error: {str(e)}")
            self.show_error(f"Gagal menampilkan pratinjau: {str(e)}")
    
    def _get_ui_config(self) -> Dict[str, Any]:
        """Get current configuration from UI components."""
        config = {}
        
        # Example: Get values from UI components
        if hasattr(self.ui_components, 'augmentation_type_dropdown'):
            config['augmentation_type'] = self.ui_components.augmentation_type_dropdown.value
            
        if hasattr(self.ui_components, 'intensity_slider'):
            config['intensity'] = self.ui_components.intensity_slider.value
            
        # Add more UI component values to config as needed
        
        return config
    
    def render(self) -> Any:
        """Render the augmentation UI."""
        if not hasattr(self, 'ui_components') or not self.ui_components:
            self.logger.warning("UI components not initialized")
            return None
            
        # Return the main container
        if hasattr(self.ui_components, 'main_container'):
            return self.ui_components.main_container
            
        self.logger.warning("Main container not found in UI components")
        return None


# ==================== FACTORY FUNCTIONS ====================

# Global instance for singleton pattern
_augmentation_module_instance = None

def create_augmentation_uimodule(
    config: Optional[Dict[str, Any]] = None,
    auto_initialize: bool = True,
    **kwargs
) -> AugmentationUIModule:
    """
    Create a new Augmentation UIModule instance.
    
    Args:
        config: Optional configuration dictionary
        auto_initialize: Whether to auto-initialize the module
        **kwargs: Additional arguments
        
    Returns:
        AugmentationUIModule instance
    """
    global _augmentation_module_instance
    
    # Reset existing instance if any
    if _augmentation_module_instance is not None:
        _augmentation_module_instance.cleanup()
    
    # Create new instance
    _augmentation_module_instance = AugmentationUIModule(config=config, **kwargs)
    
    # Auto-initialize if requested
    if auto_initialize:
        _augmentation_module_instance.initialize()
    
    return _augmentation_module_instance

def get_augmentation_uimodule() -> Optional[AugmentationUIModule]:
    """Get the current Augmentation UIModule instance."""
    return _augmentation_module_instance

def reset_augmentation_uimodule() -> None:
    """Reset the global Augmentation UIModule instance."""
    global _augmentation_module_instance
    if _augmentation_module_instance is not None:
        _augmentation_module_instance.cleanup()
    _augmentation_module_instance = None

def initialize_augmentation_ui(**kwargs) -> AugmentationUIModule:
    """
    Initialize the Augmentation UI with the given configuration.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        Initialized AugmentationUIModule instance
    """
    return create_augmentation_uimodule(config=kwargs, auto_initialize=True)
