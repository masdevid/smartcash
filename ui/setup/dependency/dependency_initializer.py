"""
File: smartcash/ui/setup/dependency/dependency_initializer.py
Deskripsi: Dependency initializer using the new CommonInitializer
"""

from typing import Dict, Any, Optional

from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.common.logger import get_logger
from smartcash.ui.setup.dependency.utils import update_status_panel

# Global instance to avoid circular imports
_dependency_initializer = None

class DependencyInitializer(CommonInitializer):
    """Dependency initializer with proper component and handler setup."""
    
    def __init__(self):
        from .handlers.config_handler import DependencyConfigHandler
        super().__init__('dependency', DependencyConfigHandler)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default dependency configuration."""
        try:
            from .handlers.defaults import get_default_dependency_config
            return get_default_dependency_config()
        except ImportError:
            return {
                'module_name': 'dependency',
                'dependencies': {
                    'torch': {'version': 'latest', 'required': True},
                    'torchvision': {'version': 'latest', 'required': True}, 
                    'ultralytics': {'version': 'latest', 'required': True}
                }
            }
    
    def _get_ui_root(self, ui_components: Dict[str, Any]) -> Any:
        """Get the root UI component from components dictionary."""
        return ui_components.get('ui')
    
    def _create_ui_components(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Create and return UI components for dependency management."""
        try:
            from smartcash.ui.setup.dependency.components.ui_components import create_dependency_main_ui
            return create_dependency_main_ui(config)
        except Exception as e:
            self.logger.error(f"Failed to create dependency UI components: {str(e)}")
            raise
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Setup event handlers for dependency management."""
        try:
            from .handlers import setup_dependency_handlers
            setup_dependency_handlers(ui_components, config, self.logger)
            return ui_components
        except Exception as e:
            self.logger.error(f"Failed to setup dependency handlers: {str(e)}")
            raise
    
    def _pre_initialize_checks(self, **kwargs) -> None:
        """Run pre-initialization checks for dependencies including package analysis."""
        logger = self.logger
        ui_components = kwargs.get('ui_components', {})
        
        try:
            logger.info("🔍 Running package analysis as part of initialization...")
            
            # Import analysis handler
            from .handlers.analysis_handler import setup_analysis_handler
            
            # Setup analysis handler
            analysis_handler = setup_analysis_handler(ui_components)
            if not analysis_handler:
                logger.warning("⚠️ Failed to setup analysis handler")
                return
                
            # Run analysis
            analysis_handler()
            logger.info("✅ Package analysis completed during initialization")
            
        except Exception as e:
            logger.error(f"❌ Failed to run package analysis during initialization: {e}", exc_info=True)
            # Don't raise to allow the UI to still load with a warning
            update_status_panel(ui_components, 
                             f"⚠️ Package analysis failed: {str(e)}", 
                             "warning")

def initialize_dependency_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Initialize and return the dependency UI with fallback on error."""
    global _dependency_initializer
    
    try:
        if _dependency_initializer is None:
            _dependency_initializer = DependencyInitializer()
        
        return _dependency_initializer.initialize(config or {})
        
    except Exception as e:
        logger = get_logger("smartcash.ui.setup.dependency")
        logger.error(f"Failed to initialize dependency UI: {e}", exc_info=True)
        
        # Create error UI
        from IPython.display import HTML
        return HTML(f"""
            <div style='color: #dc3545; padding: 1em; border: 1px solid #f5c6cb; 
                        background: #f8d7da; border-radius: 4px;'>
                <h4>❌ Failed to initialize Dependency Manager</h4>
                <p>Error: {str(e)}</p>
            </div>
        """)

__all__ = ['initialize_dependency_ui']