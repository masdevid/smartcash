"""
File: smartcash/ui/setup/dependency/dependency_init.py
Deskripsi: Entry point utama untuk dependency installer dengan ui_logger sebagai default
"""

from typing import Dict, Any, Optional
from smartcash.ui.initializers.common_initializer import CommonInitializer

# Global instance untuk menghindari circular imports
_dependency_initializer = None

class DependencyInitializer(CommonInitializer):
    """Dependency initializer dengan proper handler setup"""
    
    def __init__(self):
        from .handlers.config_handler import DependencyConfigHandler
        super().__init__('dependency', DependencyConfigHandler, 'setup')
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config - required abstract method"""
        from .handlers.defaults import get_default_dependency_config
        return get_default_dependency_config()
    
    def _get_critical_components(self) -> list:
        """Get critical components list - required abstract method"""
        return [
            'ui', 'install_button', 'analyze_button', 'check_button', 
            'save_button', 'reset_button', 'log_output', 'status_panel',
            'progress_tracker', 'logger'
        ]
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup module handlers - required abstract method"""
        from .handlers.dependency_handler import setup_dependency_handlers
        
        try:
            handlers = setup_dependency_handlers(ui_components)
            ui_components['handlers'] = handlers
            
            logger = ui_components.get('logger')
            if logger:
                logger.info(f"🎯 {len(handlers)} handlers berhasil disetup")
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Setup handlers failed: {str(e)}")
        
        return ui_components
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan logger integration"""
        from .components.ui_components import create_dependency_main_ui
        from smartcash.ui.utils.ui_logger import create_ui_logger
        
        # Create UI components
        ui_components = create_dependency_main_ui(config)
        
        # Create logger untuk dependency installer
        logger = create_ui_logger(
            ui_components=ui_components,
            name="dependency_installer",
            log_level=config.get('ui_settings', {}).get('log_level', 'INFO')
        )
        
        # Update components dengan logger dan metadata
        ui_components.update({
            'logger': logger,
            'dependency_initialized': True,
            'auto_analyze_on_render': config.get('ui_settings', {}).get('auto_analyze_on_render', True),
            'module_name': 'dependency'
        })
        
        return ui_components
    
    def _setup_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Setup handlers untuk dependency operations"""
        from .handlers.dependency_handler import setup_dependency_handlers
        
        try:
            handlers = setup_dependency_handlers(ui_components)
            ui_components['handlers'] = handlers
            
            logger = ui_components.get('logger')
            if logger:
                logger.info(f"🎯 {len(handlers)} handlers berhasil disetup")
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"❌ Setup handlers failed: {str(e)}")

def initialize_dependency_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Initialize dependency UI dengan shared components"""
    global _dependency_initializer
    
    if _dependency_initializer is None:
        _dependency_initializer = DependencyInitializer()
    
    return _dependency_initializer.initialize(config)

def get_dependency_config() -> Dict[str, Any]:
    """Get current dependency config"""
    global _dependency_initializer
    if _dependency_initializer and _dependency_initializer.config_handler:
        return _dependency_initializer.config_handler.get_current_config()
    return {}

def get_dependency_config_handler():
    """Get dependency config handler instance"""
    global _dependency_initializer
    return _dependency_initializer.config_handler if _dependency_initializer else None

def update_dependency_config(config: Dict[str, Any]) -> bool:
    """Update dependency config"""
    try:
        handler = get_dependency_config_handler()
        if handler and _dependency_initializer.ui_components:
            handler.update_ui(_dependency_initializer.ui_components, config)
            return True
        return False
    except Exception:
        return False

def reset_dependency_config() -> bool:
    """Reset dependency config ke defaults"""
    try:
        handler = get_dependency_config_handler()
        if handler and _dependency_initializer.ui_components:
            handler.reset_ui(_dependency_initializer.ui_components)
            return True
        return False
    except Exception:
        return False

def validate_dependency_setup() -> Dict[str, Any]:
    """Validate dependency setup"""
    global _dependency_initializer
    if not _dependency_initializer:
        return {'valid': False, 'error': 'Not initialized'}
    
    from .handlers.dependency_handler import validate_ui_components
    return validate_ui_components(_dependency_initializer.ui_components or {})

def get_dependency_status() -> Dict[str, Any]:
    """Get dependency installer status"""
    global _dependency_initializer
    
    status = {
        'initialized': _dependency_initializer is not None,
        'ui_components_ready': False,
        'handlers_ready': False,
        'config_handler_ready': False
    }
    
    if _dependency_initializer:
        status['ui_components_ready'] = _dependency_initializer.ui_components is not None
        status['config_handler_ready'] = _dependency_initializer.config_handler is not None
        
        if _dependency_initializer.ui_components:
            status['handlers_ready'] = 'handlers' in _dependency_initializer.ui_components
    
    return status

# Utility functions untuk kompatibilitas
def get_selected_packages_count() -> int:
    """Get jumlah packages yang dipilih"""
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
    """Check if auto analyze is enabled"""
    config = get_dependency_config()
    return config.get('auto_analyze', True)

def cleanup_dependency_generators():
    """Cleanup generators untuk menghindari memory leak"""
    global _dependency_initializer
    if _dependency_initializer and hasattr(_dependency_initializer, 'cleanup'):
        _dependency_initializer.cleanup()