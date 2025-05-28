"""
File: smartcash/ui/setup/dependency_installer/dependency_installer_initializer.py
Deskripsi: Fixed dependency installer menggunakan CommonInitializer pattern
"""

from typing import Dict, Any, List
from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.ui_logger_namespace import DEPENDENCY_INSTALLER_LOGGER_NAMESPACE, KNOWN_NAMESPACES
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[DEPENDENCY_INSTALLER_LOGGER_NAMESPACE]

# Import handlers dan components
from smartcash.ui.setup.dependency_installer.handlers.setup_handlers import setup_dependency_installer_handlers
from smartcash.ui.setup.dependency_installer.components.dependency_installer_component import create_dependency_installer_ui


class DependencyInstallerInitializer(CommonInitializer):
    """Fixed dependency installer menggunakan CommonInitializer pattern"""
    
    def __init__(self):
        super().__init__(MODULE_LOGGER_NAME, DEPENDENCY_INSTALLER_LOGGER_NAMESPACE)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration untuk dependency installer"""
        return {
            'auto_install': False,
            'selected_packages': ['yolov5_req', 'smartcash_req', 'torch_req'],
            'custom_packages': '',
            'validate_after_install': True
        }
    
    def _get_critical_components(self) -> List[str]:
        """Critical component keys yang harus ada"""
        return ['ui', 'install_button', 'status']
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components untuk dependency installer"""
        return create_dependency_installer_ui(env, config)
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers untuk dependency installer"""
        try:
            ui_components = setup_dependency_installer_handlers(ui_components, env, config)
            ui_components['handlers_setup'] = True
        except Exception as e:
            logger = ui_components.get('logger', self.logger)
            logger.error(f"❌ Handlers setup failed: {str(e)}")
            ui_components['handlers_setup'] = False
        
        return ui_components
    
    def _additional_validation(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Validation untuk dependency installer"""
        button_keys = ['install_button']
        functional_buttons = [key for key in button_keys if ui_components.get(key) and hasattr(ui_components[key], 'on_click')]
        
        if not functional_buttons:
            return {'valid': False, 'message': 'Install button tidak functional'}
        
        return {'valid': True, 'functional_buttons': functional_buttons}


# Global instance
_dependency_installer_initializer = DependencyInstallerInitializer()

# Public API
initialize_dependency_installer = lambda env=None, config=None: _dependency_installer_initializer.initialize(env=env, config=config)