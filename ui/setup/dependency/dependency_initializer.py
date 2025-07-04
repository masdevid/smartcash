def setup_operation_handlers(self) -> None:
        """
File: smartcash/ui/setup/dependency/dependency_initializer.py
Deskripsi: Dependency module initializer dengan proper inheritance
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.utils.constants import ICONS

from .components.dependency_tabs import create_dependency_tabs
from .handlers.dependency_ui_handler import DependencyUIHandler
from .configs.dependency_defaults import get_default_dependency_config

class DependencyInitializer(ModuleInitializer):
    """Initializer untuk dependency module dengan proper structure"""
    
    def __init__(self):
        super().__init__(
            module_name='dependency',
            parent_module='setup',
            handler_class=DependencyUIHandler,
            auto_setup_handlers=True
        )
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default dependency configuration"""
        return get_default_dependency_config()
    
    def create_module_ui_components(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Create dependency UI components dengan standard layout"""
        
        current_config = config or self.get_default_config()
        
        # Header container
        header_container = create_header_container(
            title=f"{ICONS.get('package', '📦')} Dependency Manager",
            subtitle="Kelola packages untuk SmartCash dengan interface yang mudah",
            status_message="Siap mengelola dependencies",
            status_type="info"
        )
        
        # Form container (tabs)
        dependency_tabs = create_dependency_tabs(current_config, self.logger)
        
        # Action container
        action_container = create_action_container(
            buttons=[
                {'name': 'install_button', 'text': 'Install Selected', 'icon': '📥', 'variant': 'primary'},
                {'name': 'check_updates_button', 'text': 'Check Updates', 'icon': '🔄', 'variant': 'secondary'},
                {'name': 'uninstall_button', 'text': 'Uninstall Selected', 'icon': '🗑️', 'variant': 'danger'}
            ],
            show_progress_tracker=True,
            show_confirmation=True
        )
        
        # Footer container
        footer_container = create_footer_container(
            logger=self.logger,
            show_info_box=True,
            info_content="💡 Tip: Gunakan tab pertama untuk categories, tab kedua untuk custom packages"
        )
        
        # Main container
        main_container = create_main_container(
            containers=[header_container, dependency_tabs, action_container, footer_container]
        )
        
        return {
            'main_container': main_container,
            'header_container': header_container,
            'dependency_tabs': dependency_tabs,
            'action_container': action_container,
            'footer_container': footer_container,
            'status_panel': header_container.status_panel,
            'install_button': action_container.buttons['install_button'],
            'check_updates_button': action_container.buttons['check_updates_button'],
            'uninstall_button': action_container.buttons['uninstall_button'],
            'progress_tracker': action_container.progress_tracker,
            'confirmation_dialog': action_container.confirmation_dialog,
            'log_accordion': footer_container.log_accordion,
            'logger': self.logger
        }
    
    def setup_operation_handlers(self) -> None:
        """Setup operation handlers untuk package management"""
        from .operations.install_handler import InstallOperationHandler
        from .operations.update_handler import UpdateOperationHandler
        from .operations.uninstall_handler import UninstallOperationHandler
        from .operations.check_status_handler import CheckStatusOperationHandler
        
        self.register_operation_handler('install', InstallOperationHandler(self._ui_components, self.config))
        self.register_operation_handler('update', UpdateOperationHandler(self._ui_components, self.config))
        self.register_operation_handler('uninstall', UninstallOperationHandler(self._ui_components, self.config))
        self.register_operation_handler('check_status', CheckStatusOperationHandler(self._ui_components, self.config))

# Global instance
_dependency_initializer: Optional[DependencyInitializer] = None

def initialize_dependency_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """🚀 Initialize dependency UI - main entry point"""
    global _dependency_initializer
    
    try:
        if _dependency_initializer is None:
            _dependency_initializer = DependencyInitializer()
        
        # Initialize dengan config
        result = _dependency_initializer.initialize(config)
        
        return {
            'ui_components': result.get('ui_components', {}),
            'module_handler': result.get('module_handler'),
            'config_handler': result.get('config_handler'),
            'operation_handlers': result.get('operation_handlers', {})
        }
        
    except Exception as e:
        print(f"❌ Error initializing dependency UI: {e}")
        return {}