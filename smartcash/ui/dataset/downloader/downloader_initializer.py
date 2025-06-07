"""
File: smartcash/ui/dataset/downloader/downloader_initializer.py
Deskripsi: FIXED downloader initializer dengan proper error handling tanpa fallbacks
"""

from typing import Dict, Any, Optional, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.downloader.components.ui_components import create_downloader_main_ui
from smartcash.ui.dataset.downloader.handlers.config_handler import DownloaderConfigHandler
from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers
from smartcash.ui.utils.logging_utils import setup_ipython_logging
from smartcash.common.logger import get_logger

class DownloaderInitializer(CommonInitializer):
    """FIXED downloader initializer dengan proper error resolution"""
    
    def __init__(self):
        super().__init__(
            module_name='downloader',
            config_handler_class=DownloaderConfigHandler,
            parent_module='dataset'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create downloader UI components dengan error resolution"""
        # Create UI components
        ui_components = create_downloader_main_ui(config)
        
        # Setup logger with UI integration
        logger = setup_ipython_logging(
            ui_components, 
            module_name='smartcash.dataset.downloader',
            log_to_file=False,
            redirect_all_logs=False
        )
        ui_components['logger'] = logger
        ui_components['download_initialized'] = True
        
        # Log successful creation
        logger.success(f"✅ Downloader UI components created successfully")
        
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup unified handlers dengan backend integration"""
        logger = ui_components.get('logger')
        
        # Setup unified handlers
        ui_components = setup_download_handlers(ui_components, config, env)
        
        # Verify handlers setup
        handlers_status = self._verify_handlers_setup(ui_components)
        
        if handlers_status['all_present']:
            logger.success("✅ Semua handlers berhasil di-setup")
        else:
            missing = handlers_status['missing']
            logger.warning(f"⚠️ Beberapa handlers tidak ditemukan: {', '.join(missing)}")
        
        return ui_components
    
    def _verify_handlers_setup(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Verify handlers setup status"""
        expected_handlers = [
            'download_handler', 'check_handler', 'cleanup_handler', 'unified_handler'
        ]
        
        present_handlers = [h for h in expected_handlers if h in ui_components]
        missing_handlers = [h for h in expected_handlers if h not in ui_components]
        
        return {
            'all_present': len(missing_handlers) == 0,
            'present': present_handlers,
            'missing': missing_handlers,
            'total_expected': len(expected_handlers),
            'total_present': len(present_handlers)
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config untuk downloader"""
        from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config
        return get_default_downloader_config()

    def _get_critical_components(self) -> List[str]:
        """Get critical components yang harus ada"""
        return [
            'ui', 'download_button', 'check_button', 'cleanup_button',
            'save_button', 'reset_button', 'log_output', 'progress_tracker'
        ]

def initialize_downloader(env=None, config=None, **kwargs) -> Any:
    """Initialize downloader UI"""
    initializer = DownloaderInitializer()
    return initializer.initialize(env, config, **kwargs)

# Export
__all__ = ['initialize_downloader', 'DownloaderInitializer']