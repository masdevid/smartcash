"""
File: smartcash/ui/setup/dependency_installer/dependency_installer_initializer.py
Deskripsi: Initializer dependency installer yang terintegrasi dengan common patterns
"""

from typing import Dict, Any, List
from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.ui_logger_namespace import DEPENDENCY_INSTALLER_LOGGER_NAMESPACE, KNOWN_NAMESPACES

MODULE_LOGGER_NAME = KNOWN_NAMESPACES[DEPENDENCY_INSTALLER_LOGGER_NAMESPACE]

# Import components dan handlers
from smartcash.ui.setup.dependency_installer.components.ui_components import create_dependency_installer_main_ui
from smartcash.ui.setup.dependency_installer.handlers.dependency_installer_handlers import setup_dependency_installer_handlers

class DependencyInstallerInitializer(CommonInitializer):
    """Initializer dependency installer dengan pattern yang konsisten"""
    
    def __init__(self):
        super().__init__(MODULE_LOGGER_NAME, DEPENDENCY_INSTALLER_LOGGER_NAMESPACE)
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components untuk dependency installer"""
        ui_components = create_dependency_installer_main_ui(config)
        ui_components.update({
            'dependency_installer_initialized': True,
            'auto_analyze_on_render': True  # Flag untuk auto-analisis setelah render
        })
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan pattern yang konsisten"""
        return setup_dependency_installer_handlers(ui_components, config, env)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default config untuk dependency installer"""
        return {
            'installation': {
                'parallel_workers': 3,
                'force_reinstall': False,
                'use_cache': True,
                'timeout': 300
            },
            'analysis': {
                'check_compatibility': True,
                'include_dev_deps': False
            }
        }
    
    def _get_critical_components(self) -> List[str]:
        """Komponen kritis yang harus ada"""
        return [
            'ui', 'install_button', 'analyze_button', 'check_button',
            'save_button', 'reset_button', 'log_output', 'status_panel'
        ]

# Global instance dan public API
_dependency_installer_initializer = DependencyInstallerInitializer()
initialize_dependency_installer_ui = lambda env=None, config=None: _dependency_installer_initializer.initialize(env=env, config=config)