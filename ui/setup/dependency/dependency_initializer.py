"""
File: smartcash/ui/setup/dependency/dependency_initializer.py
Deskripsi: Fixed dependency initializer dengan proper handler instantiation
"""

from typing import Dict, Any, Optional
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.common.logger import get_logger

_dependency_initializer = None  # Global instance to avoid circular imports

class DependencyInitializer(CommonInitializer):
    """Dependency initializer with proper component and handler setup."""
    
    def __init__(self):
        from .handlers.config_handler import DependencyConfigHandler
        super().__init__('dependency', DependencyConfigHandler, 'setup')
        self.logger = get_logger("smartcash.ui.setup.dependency")
    
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
    
    def _get_critical_components(self) -> list:
        """Return list of required component keys."""
        return [
            'ui', 'install_button', 'analyze_button', 'check_button',
            'save_button', 'reset_button', 'log_output', 'status_panel',
            'progress_tracker', 'logger'
        ]
        
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create and map UI components for dependency management.
        
        Note: Component mapping is handled in create_dependency_main_ui
        """
        try:
            from smartcash.ui.setup.dependency.components.ui_components import create_dependency_main_ui
            
            # Create and get pre-mapped UI components
            components = create_dependency_main_ui(config=config or {})
            
            # Ensure logger is set
            components['logger'] = components.get('logger') or get_logger("smartcash.ui.setup.dependency")
            
            # Verify required components
            if missing := [k for k in self._get_critical_components() if not components.get(k)]:
                self.logger.error(f"Missing components: {missing}")
                
            return components
            
        except Exception as e:
            error_msg = f"UI creation failed: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], 
                             env=None, **kwargs) -> Dict[str, Any]:
        """Initialize and configure module handlers with proper error handling."""
        # Ensure logger is available
        ui_components.setdefault('logger', 
            get_logger("smartcash.ui.setup.dependency.handlers"))
        logger = ui_components['logger']
        
        try:
            # Check for required components
            if missing := [
                k for k in self._get_critical_components() 
                if not ui_components.get(k)
            ]:
                error_msg = f"❌ Missing components: {', '.join(missing)}"
                logger.error(error_msg)
                if panel := ui_components.get('status_panel'):
                    panel.value = error_msg
                raise ValueError(error_msg)
            
            # Setup handlers
            from smartcash.ui.setup.dependency.handlers.dependency_handler import setup_dependency_handlers
            handlers = setup_dependency_handlers(ui_components) or {}
            ui_components['handlers'] = handlers
            
            logger.info(f"✅ {len(handlers)} handlers initialized")
            if panel := ui_components.get('status_panel'):
                panel.value = "✅ Handlers ready"
                
            return ui_components
            
        except Exception as e:
            error_msg = f"❌ Handler setup failed: {e}"
            logger.error(error_msg, exc_info=True)
            if panel := ui_components.get('status_panel'):
                panel.value = error_msg
                
            # Fallback to minimal handlers
            ui_components['handlers'] = {k: None for k in 
                ['install', 'analyze', 'check', 'save', 'reset']}
            return ui_components

def initialize_dependency_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Initialize and return the dependency UI with fallback on error."""
    global _dependency_initializer
    
    try:
        _dependency_initializer = _dependency_initializer or DependencyInitializer()
        return _dependency_initializer.initialize(config or {})
        
    except Exception as e:
        logger = get_logger("smartcash.ui.setup.dependency")
        logger.error(f"❌ UI initialization failed: {e}")
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        return create_fallback_ui(str(e), "dependency")

def get_dependency_config() -> Dict[str, Any]:
    """Return current dependency config with safe fallback."""
    try:
        if _dependency_initializer and _dependency_initializer.config_handler:
            return _dependency_initializer.config_handler.get_current_config()
    except Exception:
        pass
    return {
        'module_name': 'dependency',
        'dependencies': {},
        'install_options': {'force_reinstall': False}
    }

def get_dependency_config_handler():
    """Return the dependency config handler instance if available."""
    return getattr(_dependency_initializer, 'config_handler', None)

def update_dependency_config(config: Dict[str, Any]) -> bool:
    """Update dependency config with error handling."""
    try:
        if handler := get_dependency_config_handler():
            return handler.update_config(config)
    except Exception as e:
        get_logger("smartcash.ui.setup.dependency").error(
            f"Config update failed: {e}", exc_info=True)
    return False

def reset_dependency_config() -> bool:
    """Reset dependency config to defaults with error handling."""
    try:
        if handler := get_dependency_config_handler():
            return handler.reset_to_defaults()
    except Exception as e:
        get_logger("smartcash.ui.setup.dependency").error(
            f"Config reset failed: {e}", exc_info=True)
    return False

def validate_dependency_setup() -> Dict[str, Any]:
    """Validate and return dependency setup status."""
    try:
        return {
            'valid': True,
            'config': get_dependency_config(),
            'missing': [],
            'errors': []
        }
    except Exception as e:
        return {
            'valid': False,
            'config': {},
            'missing': [],
            'errors': [f"Validation failed: {e}"]
        }
    if missing_components:
        return {'valid': False, 'error': f'Missing components: {missing_components}'}
    
    return {'valid': True, 'message': 'Dependency setup valid'}

__all__ = [
    'initialize_dependency_ui',
    'get_dependency_config', 
    'get_dependency_config_handler',
    'update_dependency_config',
    'reset_dependency_config',
    'validate_dependency_setup'
]  # For backward compatibility