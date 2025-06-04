"""
File: smartcash/ui/setup/dependency/dependency_init.py
Deskripsi: Fixed dependency installer dengan generator cleanup dan public config API
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.utils.logger_bridge import get_logger
from smartcash.ui.utils.ui_logger_namespace import DEPENDENCY_LOGGER_NAMESPACE, KNOWN_NAMESPACES

# Import handlers yang sudah direfaktor
from smartcash.ui.setup.dependency.handlers.config_extractor import extract_dependency_config
from smartcash.ui.setup.dependency.handlers.config_updater import update_dependency_ui
from smartcash.ui.setup.dependency.handlers.defaults import get_default_dependency_config

# Import components dan handlers
from smartcash.ui.setup.dependency.components.ui_components import create_dependency_main_ui
from smartcash.ui.setup.dependency.handlers.dependency_handler import setup_dependency_handlers
from smartcash.ui.handlers.config_handlers import ConfigHandler

MODULE_LOGGER_NAME = KNOWN_NAMESPACES[DEPENDENCY_LOGGER_NAMESPACE]

class DependencyConfigHandler(ConfigHandler):
    """Fixed ConfigHandler dengan proper generator cleanup dan public config access"""
    
    def __init__(self, module_name: str, parent_module: str = None):
        self.module_name = module_name
        self.parent_module = parent_module
        self.logger = get_logger(MODULE_LOGGER_NAME)
        self._current_config = {}  # Public config storage
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dengan generator cleanup - one-liner delegation"""
        self._current_config = extract_dependency_config(ui_components)
        return self._current_config
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dengan generator cleanup - one-liner delegation"""
        self._current_config = config
        update_dependency_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan caching - one-liner delegation"""
        return get_default_dependency_config()
    
    def get_current_config(self) -> Dict[str, Any]:
        """Public API untuk mendapatkan current config - one-liner"""
        return self._current_config.copy()
    
    def save_config(self, ui_components: Dict[str, Any], config_name: str = None) -> bool:
        """Fixed save dengan generator cleanup - one-liner override"""
        try:
            # Force close any open generators before save
            [gen.close() for gen in ui_components.values() if hasattr(gen, 'close') and hasattr(gen, '__next__')]
            return super().save_config(ui_components, config_name)
        except RuntimeError as e:
            self.logger.error(f"💥 Generator error in save: {str(e)}")
            return False
    
    def reset_config(self, ui_components: Dict[str, Any], config_name: str = None) -> bool:
        """Fixed reset dengan generator cleanup - one-liner override"""
        try:
            # Force close any open generators before reset
            [gen.close() for gen in ui_components.values() if hasattr(gen, 'close') and hasattr(gen, '__next__')]
            return super().reset_config(ui_components, config_name)
        except RuntimeError as e:
            self.logger.error(f"💥 Generator error in reset: {str(e)}")
            return False

class DependencyInitializer(CommonInitializer):
    """Fixed dependency initializer dengan proper button binding dan public config API"""
    
    def __init__(self):
        super().__init__('dependency', DependencyConfigHandler, 'setup')
        self._config_handler_instance = None
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan fixed button binding"""
        
        # Create main UI
        ui_components = create_dependency_main_ui(config)
        
        # Force rebind buttons jika ada masalah binding
        self._ensure_button_binding(ui_components)
        
        # Add module-specific flags
        ui_components.update({
            'dependency_initialized': True,
            'auto_analyze_on_render': config.get('ui_settings', {}).get('auto_analyze_on_render', True)
        })
        
        return ui_components
    
    def _ensure_button_binding(self, ui_components: Dict[str, Any]) -> None:
        """Ensure semua buttons ter-bind dengan proper - one-liner check dan rebind"""
        critical_buttons = ['install_button', 'analyze_button', 'check_button', 'save_button', 'reset_button']
        [self._create_fallback_button(ui_components, btn_key) for btn_key in critical_buttons if btn_key not in ui_components or not hasattr(ui_components[btn_key], 'on_click')]
    
    def _create_fallback_button(self, ui_components: Dict[str, Any], button_key: str) -> None:
        """Create fallback button jika button tidak ada atau rusak - one-liner factory"""
        import ipywidgets as widgets
        ui_components[button_key] = widgets.Button(
            description=button_key.replace('_', ' ').title(),
            button_style='primary' if 'install' in button_key else 'info' if 'analyze' in button_key else '',
            tooltip=f"Action: {button_key.replace('_', ' ')}",
            layout=widgets.Layout(width='140px', height='35px')
        )
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan fixed button binding"""
        
        # Store config handler instance untuk public access
        self._config_handler_instance = ui_components.get('config_handler')
        
        # Setup handlers dengan error handling
        try:
            return setup_dependency_handlers(ui_components, config, env)
        except Exception as e:
            self.logger.error(f"💥 Error setting up handlers: {str(e)}")
            return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default config - one-liner delegation"""
        return get_default_dependency_config()
    
    def _get_critical_components(self) -> List[str]:
        """Critical components yang harus ada"""
        return [
            'ui', 'install_button', 'analyze_button', 'check_button',
            'save_button', 'reset_button', 'log_output', 'status_panel'
        ]
    
    def get_current_config(self) -> Dict[str, Any]:
        """Public API untuk mendapatkan current config - one-liner"""
        return self._config_handler_instance.get_current_config() if self._config_handler_instance else {}

# Global instance dan public API dengan enhanced config access
_dependency_initializer = DependencyInitializer()

def initialize_dependency_ui(env=None, config=None, **kwargs) -> Any:
    """Public API untuk initialize dependency installer UI"""
    return _dependency_initializer.initialize(env=env, config=config, **kwargs)

def get_dependency_config() -> Dict[str, Any]:
    """Public API untuk mendapatkan current dependency config - one-liner"""
    return _dependency_initializer.get_current_config()

def get_dependency_config_handler() -> DependencyConfigHandler:
    """Get config handler instance - one-liner factory"""
    return DependencyConfigHandler('dependency', 'setup')

def update_dependency_config(config: Dict[str, Any]) -> bool:
    """Public API untuk update dependency config - one-liner"""
    return _dependency_initializer._config_handler_instance.update_ui({}, config) if _dependency_initializer._config_handler_instance else False

def reset_dependency_config() -> bool:
    """Public API untuk reset dependency config - one-liner"""
    return _dependency_initializer._config_handler_instance.reset_config({}) if _dependency_initializer._config_handler_instance else False

def validate_dependency_setup(ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """Enhanced validate dengan generator cleanup check"""
    if ui_components is None:
        return {
            'valid': False,
            'message': 'UI components tidak ditemukan',
            'missing_components': ['all'],
            'has_config_handler': False,
            'has_logger': False,
            'module_initialized': False,
            'generator_issues': False
        }
    
    critical_components = [
        'ui', 'install_button', 'analyze_button', 'check_button',
        'save_button', 'reset_button', 'log_output', 'status_panel'
    ]
    
    missing_components = [comp for comp in critical_components if comp not in ui_components]
    
    # Check untuk generator issues
    generator_issues = any(hasattr(v, '__next__') and hasattr(v, 'close') for v in ui_components.values())
    
    validation_result = {
        'valid': len(missing_components) == 0 and not generator_issues,
        'message': 'Setup valid' if not missing_components and not generator_issues else f'Issues: {missing_components + (["generators"] if generator_issues else [])}',
        'missing_components': missing_components,
        'has_config_handler': 'config_handler' in ui_components,
        'has_logger': 'logger' in ui_components,
        'module_initialized': ui_components.get('dependency_initialized', False),
        'generator_issues': generator_issues
    }
    
    return validation_result

def get_dependency_status() -> Dict[str, Any]:
    """Enhanced status dengan config info - one-liner"""
    status = _dependency_initializer.get_module_status()
    status['current_config'] = get_dependency_config()
    status['config_keys'] = list(get_dependency_config().keys())
    return status

def cleanup_dependency_generators(ui_components: Dict[str, Any]) -> int:
    """Public API untuk cleanup generators - one-liner dengan count"""
    generators = [gen for gen in ui_components.values() if hasattr(gen, 'close') and hasattr(gen, '__next__')]
    [gen.close() for gen in generators]
    return len(generators)

# Enhanced public config utilities
def get_selected_packages_count() -> int:
    """Get count selected packages - one-liner"""
    config = get_dependency_config()
    return len(config.get('selected_packages', []))

def get_installation_settings() -> Dict[str, Any]:
    """Get current installation settings - one-liner"""
    config = get_dependency_config()
    return config.get('installation', {})

def get_analysis_settings() -> Dict[str, Any]:
    """Get current analysis settings - one-liner"""
    config = get_dependency_config()
    return config.get('analysis', {})

def is_auto_analyze_enabled() -> bool:
    """Check if auto analyze enabled - one-liner"""
    config = get_dependency_config()
    return config.get('auto_analyze', True)

# One-liner utilities untuk debugging
debug_generator_count = lambda ui_components: len([v for v in ui_components.values() if hasattr(v, '__next__')])
debug_button_status = lambda ui_components: {k: hasattr(v, 'on_click') for k, v in ui_components.items() if 'button' in k}
debug_config_summary = lambda: f"Config keys: {len(get_dependency_config())} | Selected: {get_selected_packages_count()} | Auto-analyze: {is_auto_analyze_enabled()}"