"""
File: smartcash/ui/setup/dependency/dependency_initializer.py
Deskripsi: Fixed dependency initializer dengan proper handler instantiation
"""

from typing import Dict, Any, Optional
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.common.logger import get_logger

# Global instance untuk menghindari circular imports
_dependency_initializer = None

class DependencyInitializer(CommonInitializer):
    """Fixed dependency initializer dengan proper handler setup"""
    
    def __init__(self):
        # Import handler class secara lazy
        from .handlers.config_handler import DependencyConfigHandler
        
        # Pass handler class ke parent dengan proper arguments
        super().__init__('dependency', DependencyConfigHandler, 'setup')
        self.logger = get_logger("smartcash.ui.setup.dependency")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config - required abstract method"""
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
        """Get critical components list - required abstract method"""
        return [
            'ui', 'install_button', 'analyze_button', 'check_button',
            'save_button', 'reset_button', 'log_output', 'status_panel',
            'progress_tracker', 'logger'
        ]
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup module handlers - required abstract method"""
        try:
            from .handlers.dependency_handler import setup_dependency_handlers
            
            handlers = setup_dependency_handlers(ui_components)
            ui_components['handlers'] = handlers
            
            logger = ui_components.get('logger')
            if logger:
                logger.info(f"🎯 {len(handlers)} dependency handlers berhasil disetup")
            
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Setup handlers failed: {str(e)}")
            # Fallback: create minimal handlers
            ui_components['handlers'] = {'install': None, 'check': None}
            return ui_components

def initialize_dependency_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Initialize dependency UI dengan error handling yang robust"""
    global _dependency_initializer
    
    try:
        if _dependency_initializer is None:
            _dependency_initializer = DependencyInitializer()
        
        return _dependency_initializer.initialize(config)
        
    except Exception as e:
        logger = get_logger("smartcash.ui.setup.dependency")
        logger.error(f"❌ Failed to initialize dependency UI: {str(e)}")
        
        # Return existing fallback UI
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        return create_fallback_ui(str(e), "dependency")

def get_dependency_config() -> Dict[str, Any]:
    """Get current dependency config dengan safe fallback"""
    global _dependency_initializer
    
    if _dependency_initializer and _dependency_initializer.config_handler:
        try:
            return _dependency_initializer.config_handler.get_current_config()
        except Exception:
            pass
    
    # Return default config jika tidak ada
    return {
        'module_name': 'dependency',
        'dependencies': {},
        'install_options': {'force_reinstall': False}
    }

def get_dependency_config_handler():
    """Get dependency config handler instance dengan safe access"""
    global _dependency_initializer
    return _dependency_initializer.config_handler if _dependency_initializer else None

def update_dependency_config(config: Dict[str, Any]) -> bool:
    """Update dependency config dengan error handling"""
    try:
        handler = get_dependency_config_handler()
        if handler and _dependency_initializer.ui_components:
            handler.update_ui(_dependency_initializer.ui_components, config)
            return True
        return False
    except Exception as e:
        logger = get_logger("smartcash.ui.setup.dependency")
        logger.warning(f"⚠️ Update config failed: {str(e)}")
        return False

def reset_dependency_config() -> bool:
    """Reset dependency config ke defaults dengan error handling"""
    try:
        handler = get_dependency_config_handler()
        if handler and _dependency_initializer.ui_components:
            default_config = handler.get_default_config()
            handler.update_ui(_dependency_initializer.ui_components, default_config)
            return True
        return False
    except Exception as e:
        logger = get_logger("smartcash.ui.setup.dependency")
        logger.warning(f"⚠️ Reset config failed: {str(e)}")
        return False

def validate_dependency_setup() -> Dict[str, Any]:
    """Validate dependency setup dengan comprehensive check"""
    global _dependency_initializer
    
    if not _dependency_initializer:
        return {'valid': False, 'error': 'Initializer belum dibuat'}
    
    if not _dependency_initializer.ui_components:
        return {'valid': False, 'error': 'UI components belum diinisialisasi'}
    
    if not _dependency_initializer.config_handler:
        return {'valid': False, 'error': 'Config handler belum disetup'}
    
    # Check critical components
    critical_components = _dependency_initializer._get_critical_components()
    missing_components = [comp for comp in critical_components 
                         if comp not in _dependency_initializer.ui_components]
    
    if missing_components:
        return {'valid': False, 'error': f'Missing components: {missing_components}'}
    
    return {'valid': True, 'message': 'Dependency setup valid'}

# Export functions untuk backward compatibility
__all__ = [
    'initialize_dependency_ui',
    'get_dependency_config', 
    'get_dependency_config_handler',
    'update_dependency_config',
    'reset_dependency_config',
    'validate_dependency_setup'
]