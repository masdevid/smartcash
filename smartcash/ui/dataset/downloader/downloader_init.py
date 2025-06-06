"""
File: smartcash/ui/dataset/downloader/downloader_init.py
Deskripsi: Minimal initializer untuk Dataset Downloader UI dengan delegasi ke SRP handlers
"""

from typing import Dict, Any
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.downloader.handlers.config_handler import DownloaderConfigHandler
from smartcash.ui.dataset.downloader.components.ui_components import create_downloader_main_ui

class DownloaderInitializer(CommonInitializer):
    """Minimal initializer untuk Dataset Downloader dengan delegasi ke handlers"""
    
    def __init__(self):
        super().__init__(
            module_name='downloader',
            config_handler_class=DownloaderConfigHandler,
            parent_module='dataset'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat UI components dan setup metadata minimal"""
        ui_components = create_downloader_main_ui(config)
        ui_components.update({
            'module_type': 'dataset_downloader',
            'supports_uuid': True,
            'config_file': 'dataset_config.yaml'
        })
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan import dan binding minimal"""
        from smartcash.ui.dataset.downloader.handlers import (
            check_handler, cleanup_handler, download_handler, validation_handler
        )
        
        # Setup handlers dengan one-liner error handling
        handlers = [
            ('check', check_handler.setup_check_handler),
            ('cleanup', cleanup_handler.setup_cleanup_handler),
            ('download', download_handler.setup_download_handler),
            ('validation', validation_handler.setup_validation_handler)
        ]
        
        [self._safe_setup_handler(name, setup_func, ui_components, config) 
         for name, setup_func in handlers]
        
        return ui_components
    
    def _safe_setup_handler(self, name: str, setup_func, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Safe handler setup dengan minimal error handling"""
        try:
            setup_func(ui_components, config, self.logger)
        except Exception as e:
            self.logger.warning(f"⚠️ {name.capitalize()} handler setup warning: {str(e)}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config via config handler"""
        return self.config_handler.get_default_config()
    
    def _get_critical_components(self) -> list:
        """Komponen critical minimal yang harus ada"""
        return ['ui', 'download_button', 'log_output', 'progress_tracker']

def initialize_downloader_ui(env=None, config=None, **kwargs):
    """One-liner initialization untuk downloader UI"""
    return DownloaderInitializer().initialize(env, config, **kwargs)