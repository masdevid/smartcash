"""
File: smartcash/ui/dataset/downloader/downloader_initializer.py
Deskripsi: Downloader initializer dengan integrasi handler yang telah direfaktor
"""

from typing import Dict, Any, Optional, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.downloader.components.ui_components import create_downloader_main_ui
from smartcash.ui.dataset.downloader.handlers.config_handler import DownloaderConfigHandler
from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers
from smartcash.ui.dataset.downloader.handlers.check_handler import DatasetCheckHandler
from smartcash.ui.dataset.downloader.handlers.cleanup_handler import DatasetCleanupHandler
from smartcash.ui.dataset.downloader.handlers.button_handler import ButtonHandler
from smartcash.ui.utils.logging_utils import setup_ipython_logging
from smartcash.common.logger import get_logger

class DownloaderInitializer(CommonInitializer):
    """Downloader initializer dengan handler integration"""
    
    def __init__(self):
        super().__init__(
            module_name='downloader',
            config_handler_class=DownloaderConfigHandler,
            parent_module='dataset'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create downloader UI components dengan proper config integration"""
        try:
            ui_components = create_downloader_main_ui(config)
            
            logger = setup_ipython_logging(
                ui_components, 
                module_name='smartcash.dataset.downloader',
                log_to_file=False,
                redirect_all_logs=False
            )
            ui_components['logger'] = logger
            ui_components['download_initialized'] = True
            
            return ui_components
            
        except Exception as e:
            logger = get_logger('downloader.init')
            logger.error(f"❌ Error creating downloader UI: {str(e)}")
            return self._create_fallback_ui(str(e))
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup unified handlers dengan proper binding"""
        try:
            logger = ui_components.get('logger')
            
            # Setup handlers - single call untuk semua handlers
            ui_components = setup_download_handlers(ui_components, config, env)
            
            # Verifikasi setup handler
            download_handler = ui_components.get('download_handler')
            check_handler = ui_components.get('check_handler')
            cleanup_handler = ui_components.get('cleanup_handler')
            
            if download_handler and check_handler and cleanup_handler:
                logger.success("✅ Semua handlers berhasil di-setup")
            else:
                missing = []
                if not download_handler: missing.append('download_handler')
                if not check_handler: missing.append('check_handler')
                if not cleanup_handler: missing.append('cleanup_handler')
                logger.warning(f"⚠️ Beberapa handler tidak ditemukan: {', '.join(missing)}")
            
            return ui_components
                
        except Exception as e:
            logger = ui_components.get('logger') or get_logger('downloader.handlers')
            logger.error(f"❌ Error setup handlers: {str(e)}")
            return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config untuk downloader"""
        from smartcash.ui.dataset.downloader.handlers.defaults import get_default_downloader_config
        return get_default_downloader_config()

    def _get_critical_components(self) -> List[str]:
        """Get critical components yang harus ada"""
        return [
            'ui', 'download_button', 'check_button', 'cleanup_button',
            'save_button', 'reset_button', 'log_output', 'progress_tracker',
            'download_handler', 'check_handler', 'cleanup_handler'  # Handler yang direfaktor
        ]


def initialize_downloader(env=None, config=None, **kwargs) -> Any:
    """Initialize downloader UI dengan handlers yang telah direfaktor"""
    try:
        initializer = DownloaderInitializer()
        return initializer.initialize(env, config, **kwargs)
        
    except Exception as e:
        logger = get_logger('downloader.factory')
        logger.error(f"❌ Error initializing downloader: {str(e)}")
        
        import ipywidgets as widgets
        return widgets.HTML(f"""
        <div style="padding: 15px; background: #f8d7da; border-radius: 5px; color: #721c24;">
            <h4>❌ Downloader Initialization Failed</h4>
            <p>Error: {str(e)}</p>
        </div>
        """)
