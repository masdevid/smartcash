"""
File: smartcash/ui/setup/dependency/dependency_initializer.py
Deskripsi: Dependency initializer dengan implementasi abstract methods yang lengkap
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.setup.dependency.handlers.config_handler import DependencyConfigHandler


class DependencyInitializer(CommonInitializer):
    """Dependency installer initializer dengan proper UI dan handler setup"""
    
    def __init__(self):
        super().__init__(
            module_name='dependency',
            config_handler_class=DependencyConfigHandler,
            parent_module='setup'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Implementasi wajib: Create dependency installer UI components"""
        try:
            from .components.ui_components import create_dependency_main_ui
            from smartcash.ui.utils.ui_logger import create_ui_logger
            
            # Create main UI components
            ui_components = create_dependency_main_ui(config or {})
            
            # Validate UI components
            if not isinstance(ui_components, dict) or not ui_components:
                raise ValueError("create_dependency_main_ui mengembalikan nilai invalid")
            
            # Create logger untuk dependency operations
            logger = create_ui_logger(
                ui_components=ui_components,
                name="dependency_installer",
                log_level=config.get('ui_settings', {}).get('log_level', 'INFO')
            )
            
            # Update dengan metadata dan logger
            ui_components.update({
                'logger': logger,
                'module_name': 'dependency',
                'dependency_initialized': True,
                'auto_analyze_on_render': config.get('ui_settings', {}).get('auto_analyze_on_render', True),
                'package_manager_ready': True
            })
            
            self.logger.debug(f"✅ Dependency UI components created: {list(ui_components.keys())}")
            return ui_components
            
        except ImportError as e:
            self.logger.error(f"❌ Import error: {str(e)}")
            raise ImportError(f"Tidak dapat import dependency UI components: {str(e)}")
        except Exception as e:
            self.logger.error(f"❌ Error creating dependency UI: {str(e)}")
            raise
    
    def _get_critical_components(self) -> List[str]:
        """Implementasi wajib: Komponen kritis untuk dependency installer"""
        return [
            'header',
            'status_panel',
            'log_output',
            'install_button',
            'analyze_button', 
            'check_button',
            'save_button',
            'reset_button',
            'progress_tracker',
            'package_selector'
        ]
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Implementasi wajib: Default config dari handlers/defaults.py"""
        from .handlers.defaults import get_default_dependency_config
        return get_default_dependency_config()
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], 
                             env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers khusus dependency installer"""
        try:
            from .handlers.dependency_handler import setup_dependency_handlers
            
            # Setup handlers dengan config
            handlers = setup_dependency_handlers(ui_components, config)
            ui_components['handlers'] = handlers
            
            # Auto analyze jika enabled
            if config.get('analysis', {}).get('auto_analyze_on_render', True):
                self._trigger_auto_analyze(ui_components)
            
            logger = ui_components.get('logger')
            if logger:
                logger.info(f"🎯 {len(handlers)} dependency handlers berhasil disetup")
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Import error pada dependency handlers: {str(e)}")
        except Exception as e:
            self.logger.warning(f"⚠️ Error setting up dependency handlers: {str(e)}")
        
        return ui_components
    
    def _post_initialization_hook(self, ui_components: Dict[str, Any], config: Dict[str, Any], 
                                env=None, **kwargs) -> None:
        """Hook setelah dependency initialization selesai"""
        try:
            # Update package selections dari config
            self._update_package_selections(ui_components, config)
            
            # Setup periodic status check jika diperlukan
            if config.get('analysis', {}).get('periodic_check', False):
                self._setup_periodic_check(ui_components)
            
            logger = ui_components.get('logger')
            if logger:
                selected_count = len(config.get('packages', {}).get('selected_packages', []))
                logger.info(f"🎉 Dependency installer siap dengan {selected_count} packages terpilih")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error dalam post-initialization: {str(e)}")
    
    def _trigger_auto_analyze(self, ui_components: Dict[str, Any]) -> None:
        """Trigger auto analyze untuk cek status packages"""
        try:
            from .utils.ui_state_utils import create_operation_context
            
            analyze_button = ui_components.get('analyze_button')
            if analyze_button and hasattr(analyze_button, 'click'):
                # Simulate button click untuk auto analyze
                with create_operation_context(ui_components, "Auto analyzing packages..."):
                    analyze_button.click()
                    
        except Exception as e:
            self.logger.debug(f"⚠️ Auto analyze gagal: {str(e)}")
    
    def _update_package_selections(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update package selections dari config"""
        try:
            from .utils.package_selector_utils import update_package_status
            
            selected_packages = config.get('packages', {}).get('selected_packages', [])
            for package in selected_packages:
                update_package_status(ui_components, package, selected=True)
                
        except Exception as e:
            self.logger.debug(f"⚠️ Error updating package selections: {str(e)}")
    
    def _setup_periodic_check(self, ui_components: Dict[str, Any]) -> None:
        """Setup periodic package status check"""
        try:
            # Implementation untuk periodic check jika diperlukan
            # Bisa menggunakan threading.Timer atau background tasks
            pass
        except Exception as e:
            self.logger.debug(f"⚠️ Error setting up periodic check: {str(e)}")


# Entry point function
def initialize_dependency_ui(config=None, env=None, **kwargs):
    """
    Initialize dependency installer UI.
    
    Args:
        config: Konfigurasi awal (optional)
        env: Environment info (optional)
        **kwargs: Parameter tambahan
        
    Returns:
        UI components dict atau fallback UI
    """
    return _dependency_initializer.initialize(config=config, env=env, **kwargs)


# Utility functions untuk compatibility
def get_dependency_config():
    """Get current dependency config"""
    if _dependency_initializer and hasattr(_dependency_initializer, 'config_handler'):
        return _dependency_initializer.config_handler.get_current_config()
    return {}

def get_dependency_status():
    """Get dependency installer status"""
    return {
        'initialized': _dependency_initializer is not None,
        'ready': _dependency_initializer is not None and 
                hasattr(_dependency_initializer, 'config_handler')
    }

def cleanup_dependency_resources():
    """Cleanup dependency resources"""
    global _dependency_initializer
    if _dependency_initializer:
        # Cleanup jika ada resources yang perlu dibersihkan
        _dependency_initializer = None


# Global instance
_dependency_initializer = DependencyInitializer()