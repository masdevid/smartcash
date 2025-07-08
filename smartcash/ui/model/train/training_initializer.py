"""
File: smartcash/ui/model/train/training_initializer.py
Training module initializer following DisplayInitializer pattern.
"""

from typing import Dict, Any, Optional

from smartcash.ui.core.initializers.display_initializer import create_ui_display_function
from smartcash.ui.core.initializers.module_initializer import ModuleInitializer

from .constants import DEFAULT_CONFIG
from .components.training_ui import create_training_ui
from .handlers.training_ui_handler import TrainingUIHandler


class TrainingInitializer(ModuleInitializer):
    """Initializer for training module."""
    
    def __init__(self):
        """Initialize the training module initializer."""
        super().__init__(
            module_name="train",
            parent_module="model"
        )
        self.default_config = DEFAULT_CONFIG
    
    def create_ui_components(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create UI components for training module.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Dictionary containing UI components
        """
        try:
            # Use provided config or default
            effective_config = config or self.default_config
            
            # Create UI components
            ui_components = create_training_ui(effective_config)
            
            # Create and setup UI handler
            ui_handler = TrainingUIHandler(ui_components)
            ui_components['ui_handler'] = ui_handler
            
            # Update config summary with current configuration
            if 'config_summary' in ui_components:
                summary_widget = ui_components['config_summary']
                # The summary is already created with the config, but we could update it here if needed
            
            self.logger.info("✅ Training UI components created successfully")
            
            return ui_components
            
        except Exception as e:
            error_msg = f"Failed to create training UI components: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _initialize_impl(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Implementation of initialization logic (required abstract method).
        
        Args:
            *args: Variable arguments
            **kwargs: Keyword arguments, may include 'config'
            
        Returns:
            Dictionary containing initialized UI components
        """
        config = kwargs.get('config')
        return self.initialize_module(config)
    
    def initialize_module(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Initialize the training module.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Dictionary containing initialized UI components
        """
        try:
            self.logger.info("🚀 Initializing training module...")
            
            # Create UI components
            ui_components = self.create_ui_components(config)
            
            # Perform post-initialization setup
            self._post_init_setup(ui_components)
            
            self.logger.info("✅ Training module initialized successfully")
            
            return ui_components
            
        except Exception as e:
            error_msg = f"Training module initialization failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _post_init_setup(self, ui_components: Dict[str, Any]) -> None:
        """
        Perform post-initialization setup.
        
        Args:
            ui_components: Dictionary of UI components
        """
        try:
            # Validate backend availability
            if 'ui_handler' in ui_components:
                ui_handler = ui_components['ui_handler']
                service = ui_handler.training_service
                
                # Check backend availability
                backend_status = service.validate_backend_availability()
                
                if backend_status.get("available", False):
                    self.logger.info("✅ Backend training components available")
                else:
                    self.logger.warning(f"⚠️ Backend not available: {backend_status.get('message', 'Unknown')}")
                    self.logger.info("🔄 Running in simulation mode")
            
            # Set initial button states
            if 'ui_handler' in ui_components:
                ui_handler = ui_components['ui_handler']
                ui_handler._update_button_states()
            
            # Update initial config summary
            if 'ui_handler' in ui_components:
                ui_handler = ui_components['ui_handler']
                ui_handler._update_config_summary()
            
        except Exception as e:
            self.logger.warning(f"Post-initialization setup warning: {str(e)}")


def _training_initialize_legacy(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Legacy training initialization function.
    
    Args:
        config: Optional configuration override
        
    Returns:
        Dictionary containing UI components
    """
    initializer = TrainingInitializer()
    return initializer.initialize_module(config)


# Create the standardized display function
initialize_training_ui = create_ui_display_function(
    module_name='train',
    parent_module='model',
    initializer_class=TrainingInitializer,
    legacy_function=_training_initialize_legacy
)


# Convenience functions for backward compatibility
def get_training_initializer() -> TrainingInitializer:
    """
    Get training module initializer instance.
    
    Returns:
        TrainingInitializer instance
    """
    return TrainingInitializer()


def create_training_ui_components(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create training UI components without displaying.
    
    Args:
        config: Optional configuration override
        
    Returns:
        Dictionary containing UI components
    """
    initializer = TrainingInitializer()
    return initializer.create_ui_components(config)