"""
File: smartcash/ui/setup/dependency/dependency_init.py
Deskripsi: Updated dependency installer initializer menggunakan CommonInitializer pattern
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.utils.logger_bridge import get_logger
from smartcash.ui.utils.ui_logger_namespace import DEPENDENCY_LOGGER_NAMESPACE, KNOWN_NAMESPACES

# Import handlers yang sudah direfaktor
from smartcash.ui.setup.dependency.handlers.config_extractor import extract_dependency_config
from smartcash.ui.setup.dependency.handlers.config_updater import update_dependency_ui
from smartcash.ui.setup.dependency.handlers.defaults import get_default_dependency_config

# Import components dan handlers (unchanged)
from smartcash.ui.setup.dependency.components.ui_components import create_dependency_main_ui
from smartcash.ui.setup.dependency.handlers.dependency_handler import setup_dependency_handlers
from smartcash.ui.handlers.config_handlers import ConfigHandler

MODULE_LOGGER_NAME = KNOWN_NAMESPACES[DEPENDENCY_LOGGER_NAMESPACE]

class DependencyConfigHandler(ConfigHandler):
    """ConfigHandler untuk dependency installer dengan pattern CommonInitializer"""
    
    def __init__(self, module_name: str, parent_module: str = None):
        self.module_name = module_name
        self.parent_module = parent_module
        self.logger = get_logger(MODULE_LOGGER_NAME)
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config menggunakan dedicated extractor - one-liner delegation"""
        return extract_dependency_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI menggunakan dedicated updater - one-liner delegation"""
        update_dependency_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config menggunakan defaults module - one-liner delegation"""
        return get_default_dependency_config()
        

class DependencyInitializer(CommonInitializer):
    """Updated dependency installer initializer dengan CommonInitializer pattern dan built-in logger"""
    
    def __init__(self):
        # CommonInitializer sudah handle logger setup, tidak perlu duplicate
        super().__init__('dependency', DependencyConfigHandler, 'setup')
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components untuk dependency installer dengan config integration"""
        
        # Create main UI menggunakan existing component
        ui_components = create_dependency_main_ui(config)
        
        # Add module-specific flags dan metadata (logger sudah dihandle CommonInitializer)
        ui_components.update({
            'dependency_initialized': True,
            'auto_analyze_on_render': config.get('ui_settings', {}).get('auto_analyze_on_render', True)
        })
        
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan pattern yang konsisten"""
        return setup_dependency_handlers(ui_components, config, env)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default config untuk dependency installer - one-liner delegation"""
        return get_default_dependency_config()
    
    def _get_critical_components(self) -> List[str]:
        """Komponen kritis yang harus ada untuk dependency installer"""
        return [
            'ui', 'install_button', 'analyze_button', 'check_button',
            'save_button', 'reset_button', 'log_output', 'status_panel'
        ]
    
    def _extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config menggunakan dedicated extractor - compatibility method"""
        return extract_dependency_config(ui_components)
    
    def _update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI menggunakan dedicated updater - compatibility method"""
        update_dependency_ui(ui_components, config)

# Global instance dan public API dengan updated pattern
_dependency_initializer = DependencyInitializer()

def initialize_dependency_ui(env=None, config=None, **kwargs) -> Any:
    """Public API untuk initialize dependency installer UI dengan CommonInitializer pattern"""
    return _dependency_initializer.initialize(env=env, config=config, **kwargs)

def get_dependency_config_handler() -> DependencyConfigHandler:
    """Get config handler instance untuk external usage - one-liner factory"""
    return DependencyConfigHandler('dependency', 'setup')

def validate_dependency_setup(ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """Validate dependency installer setup dengan comprehensive check
    
    Args:
        ui_components: Dictionary berisi komponen UI. Jika None, akan mengembalikan
                     status invalid dengan pesan error.
    """
    if ui_components is None:
        return {
            'valid': False,
            'message': 'UI components tidak ditemukan',
            'missing_components': ['all'],
            'has_config_handler': False,
            'has_logger': False,
            'module_initialized': False
        }
    
    critical_components = [
        'ui', 'install_button', 'analyze_button', 'check_button',
        'save_button', 'reset_button', 'log_output', 'status_panel'
    ]
    
    missing_components = [comp for comp in critical_components if comp not in ui_components]
    
    validation_result = {
        'valid': len(missing_components) == 0,
        'message': 'Setup valid' if not missing_components else f'Komponen yang hilang: {missing_components}',
        'missing_components': missing_components,
        'has_config_handler': 'config_handler' in ui_components,
        'has_logger': 'logger' in ui_components,
        'module_initialized': ui_components.get('dependency_initialized', False)
    }
    
    return validation_result

def get_dependency_status() -> Dict[str, Any]:
    """Get status dependency installer module untuk debugging - one-liner"""
    return _dependency_initializer.get_module_status()
