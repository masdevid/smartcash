"""
File: smartcash/ui/dataset/downloader/downloader_initializer.py
Deskripsi: FIXED downloader initializer dengan unified handlers dan environment fix
"""

from typing import Dict, Any, Optional, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.downloader.components.ui_components import create_downloader_main_ui
from smartcash.ui.dataset.downloader.handlers.config_handler import DownloaderConfigHandler
from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers
from smartcash.ui.utils.logging_utils import setup_ipython_logging
from smartcash.common.logger import get_logger

class DownloaderInitializer(CommonInitializer):
    """FIXED downloader initializer dengan unified handlers dan environment fix"""
    
    def __init__(self):
        super().__init__(
            module_name='downloader',
            config_handler_class=DownloaderConfigHandler,
            parent_module='dataset'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create downloader UI components dengan proper environment"""
        try:
            # Log environment info at start
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            env_info = env_manager.get_system_info()
            temp_logger = get_logger('downloader.init')
            temp_logger.info(f"🌍 Environment: {env_info['environment']}")
            temp_logger.info(f"📂 Dataset path: {env_manager.get_dataset_path()}")
            temp_logger.info(f"💾 Drive mounted: {env_info['drive_mounted']}")
            
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
            
        except Exception as e:
            logger = get_logger('downloader.init')
            logger.error(f"❌ Error creating downloader UI: {str(e)}")
            return self._create_fallback_ui(str(e))
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup unified handlers dengan backend integration"""
        try:
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
                
        except Exception as e:
            logger = ui_components.get('logger') or get_logger('downloader.handlers')
            logger.error(f"❌ Error setup handlers: {str(e)}")
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
    
    def _create_fallback_ui(self, error_message: str) -> Dict[str, Any]:
        """Create fallback UI untuk error cases"""
        import ipywidgets as widgets
        
        error_widget = widgets.HTML(f"""
        <div style="padding: 15px; background: #f8d7da; border-radius: 5px; color: #721c24; margin: 10px 0;">
            <h4>❌ Downloader Initialization Failed</h4>
            <p>Error: {error_message}</p>
            <small>💡 Try restarting cell atau check environment setup</small>
        </div>
        """)
        
        return {
            'ui': error_widget,
            'main_container': error_widget,
            'error': True,
            'error_message': error_message
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
    """Initialize downloader UI dengan FIXED environment dan unified handlers"""
    try:
        # Pre-initialize environment manager
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        
        # Log environment status
        temp_logger = get_logger('downloader.factory')
        temp_logger.info(f"🌍 Initializing in: {env_manager.get_system_info()['environment']}")
        temp_logger.info(f"📂 Dataset path: {env_manager.get_dataset_path()}")
        
        # Create initializer
        initializer = DownloaderInitializer()
        result = initializer.initialize(env, config, **kwargs)
        
        temp_logger.success("✅ Downloader initialization completed")
        return result
        
    except Exception as e:
        logger = get_logger('downloader.factory')
        logger.error(f"❌ Critical error initializing downloader: {str(e)}")
        
        import ipywidgets as widgets
        return widgets.HTML(f"""
        <div style="padding: 15px; background: #f8d7da; border-radius: 5px; color: #721c24;">
            <h4>❌ Critical Downloader Error</h4>
            <p>Error: {str(e)}</p>
            <small>Check environment setup dan restart kernel jika diperlukan</small>
        </div>
        """)

# Export
__all__ = ['initialize_downloader', 'DownloaderInitializer']