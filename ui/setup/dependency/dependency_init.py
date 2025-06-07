"""
File: smartcash/ui/setup/dependency/dependency_init.py
Deskripsi: Dependency installer init dengan fixed logger, generator cleanup, dan public API
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
# Import handlers
from smartcash.ui.setup.dependency.handlers.defaults import get_default_dependency_config
from smartcash.ui.setup.dependency.components.ui_components import create_dependency_main_ui
from smartcash.ui.setup.dependency.handlers.dependency_handler import setup_dependency_handlers
from smartcash.ui.setup.dependency.handlers.config_handler import DependencyConfigHandler

class DependencyInitializer(CommonInitializer):
    """Fixed dependency initializer dengan proper handlers"""
    
    def __init__(self):
        super().__init__('dependency', DependencyConfigHandler, 'setup')
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components"""
        ui_components = create_dependency_main_ui(config)
        ui_components.update({
            'dependency_initialized': True,
            'auto_analyze_on_render': config.get('ui_settings', {}).get('auto_analyze_on_render', True)
        })
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers"""
        return setup_dependency_handlers(ui_components, config, env)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default config"""
        return get_default_dependency_config()
    
    def _get_critical_components(self) -> List[str]:
        """Critical components"""
        return ['ui', 'install_button', 'analyze_button', 'check_button', 'save_button', 'reset_button', 'log_output', 'status_panel', 'progress_tracker', 'progress_container', 'show_for_operation', 'update_progress', 'complete_operation', 'error_operation', 'reset_all']
    
    def get_current_config(self) -> Dict[str, Any]:
        """Public API untuk current config"""
        handler = getattr(self, '_config_handler_instance', None)
        return handler.get_current_config() if handler else {}

# Global instance
_dependency_initializer = DependencyInitializer()

def initialize_dependency_ui(env=None, config=None, **kwargs) -> Any:
    """Initialize dependency installer UI"""
    return _dependency_initializer.initialize(env=env, config=config, **kwargs)

def get_dependency_config() -> Dict[str, Any]:
    """Get current dependency config"""
    return _dependency_initializer.get_current_config()

def get_dependency_config_handler() -> DependencyConfigHandler:
    """Get config handler instance"""
    return DependencyConfigHandler('dependency', 'setup')

def update_dependency_config(config: Dict[str, Any]) -> bool:
    """Update dependency config"""
    handler = getattr(_dependency_initializer, '_config_handler_instance', None)
    if handler:
        try:
            handler.update_ui({}, config)
            return True
        except Exception:
            return False
    return False

def reset_dependency_config() -> bool:
    """Reset dependency config"""
    handler = getattr(_dependency_initializer, '_config_handler_instance', None)
    if handler:
        try:
            default_config = handler.get_default_config()
            handler.update_ui({}, default_config)
            return True
        except Exception:
            return False
    return False

def validate_dependency_setup(ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """Validate dependency setup"""
    if ui_components is None:
        return {'valid': False, 'message': 'UI components tidak ditemukan', 'missing_components': ['all']}
    
    critical_components = ['ui', 'install_button', 'analyze_button', 'check_button', 'save_button', 'reset_button', 'log_output', 'status_panel', 
                          'progress_tracker', 'progress_container', 'show_for_operation', 'update_progress', 'complete_operation', 'error_operation', 'reset_all']
    missing_components = [comp for comp in critical_components if comp not in ui_components]
    generator_issues = any(hasattr(v, '__next__') and hasattr(v, 'close') for v in ui_components.values())
    
    return {
        'valid': len(missing_components) == 0 and not generator_issues,
        'message': 'Setup valid' if not missing_components and not generator_issues else f'Issues: {missing_components + (["generators"] if generator_issues else [])}',
        'missing_components': missing_components,
        'has_config_handler': 'config_handler' in ui_components,
        'has_logger': 'logger' in ui_components,
        'module_initialized': ui_components.get('dependency_initialized', False),
        'generator_issues': generator_issues
    }

def get_dependency_status() -> Dict[str, Any]:
    """Get dependency status dengan config info"""
    status = _dependency_initializer.get_module_status()
    status['current_config'] = get_dependency_config()
    status['config_keys'] = list(get_dependency_config().keys())
    return status

def cleanup_dependency_generators(ui_components: Dict[str, Any]) -> int:
    """Cleanup generators"""
    generators = [gen for gen in ui_components.values() if hasattr(gen, 'close') and hasattr(gen, '__next__')]
    [gen.close() for gen in generators]
    return len(generators)

# Public config utilities
def get_selected_packages_count() -> int:
    """Get selected packages count"""
    config = get_dependency_config()
    return len(config.get('selected_packages', []))

def get_installation_settings() -> Dict[str, Any]:
    """Get installation settings"""
    config = get_dependency_config()
    return config.get('installation', {})

def get_analysis_settings() -> Dict[str, Any]:
    """Get analysis settings"""
    config = get_dependency_config()
    return config.get('analysis', {})

def is_auto_analyze_enabled() -> bool:
    """Check auto analyze status"""
    config = get_dependency_config()
    return config.get('auto_analyze', True)
