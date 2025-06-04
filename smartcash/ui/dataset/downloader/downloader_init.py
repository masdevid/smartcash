"""
File: smartcash/ui/dataset/downloader/downloader_init.py
Deskripsi: Updated downloader initializer dengan streamlined handlers dan reduced redundancy
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.utils.logger_bridge import get_logger
from smartcash.ui.utils.ui_logger_namespace import DOWNLOAD_LOGGER_NAMESPACE, KNOWN_NAMESPACES
from smartcash.ui.dataset.downloader.handlers.config_handler import DownloaderConfigHandler
from smartcash.ui.dataset.downloader.handlers.streamlined_download_handler import setup_streamlined_download_handlers
from smartcash.ui.dataset.downloader.components.ui_layout import create_downloader_ui

class StreamlinedDownloaderInitializer(CommonInitializer):
    """Streamlined downloader initializer dengan reduced redundancy dan improved integration."""
    
    def __init__(self):
        super().__init__('downloader', DownloaderConfigHandler, 'dataset')
        self._instance_config = {}
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan existing layout - no changes to form interface."""
        ui_components = create_downloader_ui(config, env)
        ui_components.update({
            'downloader_initialized': True,
            'config': config,
            'env_manager': env
        })
        
        # Store instance config
        self._instance_config = config.copy()
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup streamlined handlers dengan reduced redundancy."""
        # Use streamlined handler instead of multiple separate handlers
        handler_result = setup_streamlined_download_handlers(ui_components, config, env)
        
        # Add config handler instance untuk public access
        ui_components['config_handler'] = self.config_handler
        
        return handler_result
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan YAML structure preservation."""
        from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config
        return get_default_downloader_config()
    
    def _get_critical_components(self) -> List[str]:
        """Critical components yang harus ada."""
        return [
            'ui', 'download_button', 'check_button', 'cleanup_button', 
            'save_button', 'reset_button', 'workspace_field', 'project_field', 
            'version_field', 'api_key_field', 'log_output', 'status_panel'
        ]
    
    def get_current_config(self) -> Dict[str, Any]:
        """Public API untuk current config dengan YAML structure."""
        return self._instance_config.copy() if self._instance_config else self._get_default_config()
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update config dengan YAML structure preservation."""
        try:
            # Merge dengan existing config untuk maintain structure
            merged_config = {**self._instance_config, **new_config}
            self._instance_config = merged_config
            return True
        except Exception:
            return False

# Global instance dengan singleton pattern
_downloader_initializer = StreamlinedDownloaderInitializer()

def initialize_downloader_ui(env=None, config=None, **kwargs) -> Any:
    """Main entry point untuk downloader UI initialization."""
    return _downloader_initializer.initialize(env=env, config=config, **kwargs)

def get_downloader_status() -> Dict[str, Any]:
    """Get comprehensive downloader status."""
    status = _downloader_initializer.get_module_status()
    status.update({
        'current_config_keys': list(_downloader_initializer.get_current_config().keys()),
        'config_structure_preserved': 'roboflow' in _downloader_initializer.get_current_config(),
        'backend_integration': True
    })
    return status

def get_downloader_config() -> Dict[str, Any]:
    """Get current downloader config dengan YAML structure."""
    return _downloader_initializer.get_current_config()

def update_downloader_config(config: Dict[str, Any]) -> bool:
    """Update downloader config dengan structure preservation."""
    return _downloader_initializer.update_config(config)

def reset_downloader_config() -> bool:
    """Reset downloader config ke default dengan YAML structure."""
    try:
        default_config = _downloader_initializer._get_default_config()
        return _downloader_initializer.update_config(default_config)
    except Exception:
        return False

def validate_downloader_setup(ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """Validate downloader setup dengan comprehensive checking."""
    if not ui_components:
        return {'valid': False, 'message': 'No UI components provided', 'missing_components': ['all']}
    
    critical_components = _downloader_initializer._get_critical_components()
    missing_components = [comp for comp in critical_components if comp not in ui_components]
    
    return {
        'valid': len(missing_components) == 0,
        'message': 'Setup valid' if not missing_components else f'Missing: {", ".join(missing_components)}',
        'missing_components': missing_components,
        'has_config_handler': 'config_handler' in ui_components,
        'has_logger': 'logger' in ui_components,
        'has_progress_integrator': 'progress_integrator' in ui_components,
        'backend_integration': ui_components.get('create_download_service') is not None,
        'module_initialized': ui_components.get('downloader_initialized', False)
    }

# Backward compatibility exports
def extract_downloader_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Backward compatibility: extract config using config handler."""
    config_handler = ui_components.get('config_handler')
    return config_handler.extract_config(ui_components) if config_handler else {}

def update_downloader_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Backward compatibility: update UI using config handler."""
    config_handler = ui_components.get('config_handler')
    config_handler and config_handler.update_ui(ui_components, config)

def get_default_downloader_config() -> Dict[str, Any]:
    """Backward compatibility: get default config."""
    return _downloader_initializer._get_default_config()

def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Backward compatibility: setup handlers."""
    return setup_streamlined_download_handlers(ui_components, config, env)