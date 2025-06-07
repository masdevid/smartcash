"""
File: smartcash/ui/dataset/downloader/downloader_initializer.py
Deskripsi: Downloader initializer 
"""

from typing import Dict, Any, Optional, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.downloader.components.ui_components import create_downloader_main_ui
from smartcash.ui.dataset.downloader.handlers.config_handler import DownloaderConfigHandler

class DownloaderInitializer(CommonInitializer):
    """Downloader initializer"""
    
    def __init__(self):
        super().__init__(
            module_name='downloader',
            config_handler_class=DownloaderConfigHandler,
            parent_module='dataset'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create downloader UI components"""
        self._clear_existing_widgets()

        ui_components = create_downloader_main_ui(config)
        ui_components.update({
            'dataset_downloader_initialized': True,
            'module_name': 'downloader'
        })
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup unified handlers"""
        from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers

        try:
            return setup_download_handlers(ui_components, config, env)
        except Exception as e:
            self.logger.error(f"💥 Error setup handlers: {str(e)}")
            return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config"""
        from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config
        return get_default_downloader_config()

    def _get_critical_components(self) -> List[str]:
        """Get critical components"""
        return [
            'ui', 'download_button', 'check_button', 'cleanup_button',
            'save_button', 'reset_button', 'log_output', 'progress_tracker'
        ]
    
    def _clear_existing_widgets(self) -> None:
        """Clear existing widgets untuk avoid conflicts"""
        try:
            import gc
            from IPython.display import clear_output
            # Force garbage collection
            gc.collect()
            # Clear any existing outputs
            clear_output(wait=True)
        except Exception:
            pass  # Silent fail jika clear tidak berhasil

_downloader_init = DownloaderInitializer()

def initialize_downloader(env=None, config=None, **kwargs) -> Any:
    """Initialize downloader UI"""
    return _downloader_init.initialize(env, config, **kwargs)
