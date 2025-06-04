"""
File: smartcash/ui/dataset/downloader/downloader_init.py
Deskripsi: Downloader initializer yang menggunakan CommonInitializer pattern dengan config inheritance
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.downloader.handlers.config_extractor import DownloaderConfigExtractor
from smartcash.ui.dataset.downloader.handlers.config_updater import DownloaderConfigUpdater
from smartcash.ui.dataset.downloader.components.main_ui import create_downloader_ui
from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers
from smartcash.ui.dataset.downloader.handlers.validation_handler import setup_validation_handlers
from smartcash.ui.dataset.downloader.handlers.progress_handler import setup_progress_handlers

class DownloaderInitializer(CommonInitializer):
    """Downloader initializer dengan CommonInitializer pattern dan config inheritance."""
    
    def __init__(self):
        # Config handler dengan extractor/updater terpisah
        config_handler_class = type('DownloaderConfigHandler', (object,), {
            'extract_config': lambda self, ui: DownloaderConfigExtractor.extract_config(ui),
            'update_ui': lambda self, ui, cfg: DownloaderConfigUpdater.update_ui(ui, cfg),
            'get_default_config': lambda self: self._get_default_config()
        })
        
        super().__init__('downloader', config_handler_class, parent_module='dataset')
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan responsive layout."""
        return create_downloader_ui(config, env)
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan progress callback integration."""
        try:
            # Setup validation handlers
            validation_result = setup_validation_handlers(ui_components, config)
            
            # Setup progress handlers dengan callback
            progress_result = setup_progress_handlers(ui_components)
            
            # Setup download handlers dengan validation dan progress
            download_result = setup_download_handlers(ui_components, env, config)
            
            return {**validation_result, **progress_result, **download_result}
            
        except Exception as e:
            logger = ui_components.get('logger')
            logger and logger.error(f"❌ Handler setup error: {str(e)}")
            return {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config dari defaults.py."""
        try:
            from smartcash.ui.dataset.downloader.handlers.defaults import DEFAULT_CONFIG
            return DEFAULT_CONFIG
        except ImportError:
            return self._fallback_config()
    
    def _fallback_config(self) -> Dict[str, Any]:
        """Fallback config jika defaults.py tidak ada."""
        return {
            'workspace': 'smartcash-wo2us',
            'project': 'rupiah-emisi-2022',
            'version': '3',
            'api_key': '',
            'output_format': 'yolov5pytorch',
            'validate_download': True,
            'organize_dataset': True,
            'backup_existing': False,
            'progress_enabled': True
        }
    
    def _get_critical_components(self) -> List[str]:
        """Critical components yang harus ada."""
        return [
            'ui', 'download_button', 'validate_button', 'workspace_field',
            'project_field', 'version_field', 'api_key_field', 'progress_tracker'
        ]

# Singleton instance dan public API
_downloader_initializer = DownloaderInitializer()

def initialize_downloader_ui(env=None, config=None, **kwargs):
    """Public API untuk initialize downloader UI."""
    return _downloader_initializer.initialize(env=env, config=config, **kwargs)

def get_downloader_status():
    """Get status downloader initializer."""
    return _downloader_initializer.get_module_status()