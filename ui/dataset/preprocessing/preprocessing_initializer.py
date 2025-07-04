"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Preprocessing initializer dengan ModuleInitializer integration
"""

from typing import Dict, Any, List, Optional
from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from smartcash.ui.dataset.preprocessing.configs import PreprocessingConfigHandler, create_preprocessing_config_handler
from smartcash.ui.dataset.preprocessing.components.ui_components import create_preprocessing_main_ui
from smartcash.ui.dataset.preprocessing.handlers.module_handler import PreprocessingModuleHandler, create_preprocessing_module_handler


class PreprocessingInitializer(ModuleInitializer):
    """Preprocessing initializer dengan ModuleInitializer integration.
    
    Features:
    - 🎯 Complete lifecycle management
    - 📊 UI-Config synchronization
    - 🔧 Handler setup dan management
    - 🔄 Event registration
    """
    
    def __init__(self):
        """Initialize preprocessing initializer."""
        super().__init__(
            module_name='preprocessing',
            parent_module='dataset'
        )
    
    def create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Create UI components untuk preprocessing module.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components
        """
        self.logger.info("🎯 Creating preprocessing UI components")
        
        # Create UI components
        ui_components = create_preprocessing_main_ui(config)
        
        # Add metadata
        ui_components.update({
            'module_name': self.module_name,
            'parent_module': self.parent_module,
            'data_dir': config.get('data', {}).get('dir', 'data')
        })
        
        self.logger.info(f"✅ Created {len(ui_components)} UI components")
        return ui_components
    
    def create_config_handler(self, **kwargs) -> PreprocessingConfigHandler:
        """Create config handler untuk preprocessing module.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            PreprocessingConfigHandler instance
        """
        self.logger.info("🎯 Creating preprocessing config handler")
        
        # Get persistence and sharing settings
        persistence_enabled = kwargs.get('persistence_enabled', True)
        enable_sharing = kwargs.get('enable_sharing', True)
        
        # Create config handler
        config_handler = create_preprocessing_config_handler(
            persistence_enabled=persistence_enabled,
            enable_sharing=enable_sharing
        )
        
        self.logger.info("✅ Created preprocessing config handler")
        return config_handler
    
    def create_module_handler(self, ui_components: Dict[str, Any], **kwargs) -> PreprocessingModuleHandler:
        """Create module handler untuk preprocessing module.
        
        Args:
            ui_components: Dictionary of UI components
            **kwargs: Additional arguments
            
        Returns:
            PreprocessingModuleHandler instance
        """
        self.logger.info("🎯 Creating preprocessing module handler")
        
        # Create module handler
        module_handler = create_preprocessing_module_handler(ui_components)
        
        self.logger.info("✅ Created preprocessing module handler")
        return module_handler
    
    def setup_handlers(self, ui_components: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Setup handlers untuk preprocessing module.
        
        Args:
            ui_components: Dictionary of UI components
            **kwargs: Additional arguments
            
        Returns:
            Updated UI components with handlers
        """
        self.logger.info("🎯 Setting up preprocessing handlers")
        
        # Add config handler to UI components
        ui_components['config_handler'] = self.config_handler
        
        # Create module handler
        module_handler = self.create_module_handler(ui_components)
        
        # Add module handler to UI components
        ui_components['module_handler'] = module_handler
        
        self.logger.info("✅ Preprocessing handlers setup complete")
        return ui_components
    
    def get_critical_components(self) -> List[str]:
        """Get list of critical UI components that must exist.
        
        Returns:
            List of critical component keys
        """
        return [
            'ui', 'preprocess_button', 'check_button', 'cleanup_button',
            'log_output', 'status_panel', 'progress_tracker', 'summary_container'
        ]
    
    def pre_initialize_checks(self, **kwargs) -> None:
        """Perform pre-initialization checks.
        
        Raises:
            Exception: If any pre-initialization check fails
        """
        # Check if we're in a supported environment
        try:
            import IPython
            # Additional checks can be added here
        except ImportError:
            raise RuntimeError("Dataset preprocessing requires IPython environment")


# Global instance
_preprocessing_initializer = PreprocessingInitializer()


def initialize_preprocessing_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Factory function untuk preprocessing UI.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        Dictionary of UI components with 'ui' as the main component
    """
    return _preprocessing_initializer.initialize(config=config, **kwargs)