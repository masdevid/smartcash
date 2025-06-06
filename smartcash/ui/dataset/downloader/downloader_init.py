"""
File: smartcash/ui/dataset/downloader/downloader_initializer.py
Deskripsi: Fixed downloader initializer dengan proper action button binding dan config persistence
"""

from typing import Dict, Any, Optional, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.downloader.components.ui_components import create_downloader_main_ui
from smartcash.ui.dataset.downloader.handlers.config_handler import create_downloader_config_handler
from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers
from smartcash.ui.utils.logging_utils import setup_ipython_logging
from smartcash.common.logger import get_logger

class DownloaderInitializer(CommonInitializer):
    """Fixed downloader initializer dengan proper action button binding dan config persistence - CommonInitializer"""
    
    def __init__(self):
        super().__init__(
            module_name='downloader',
            config_handler_class=create_downloader_config_handler,
            parent_module='dataset'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create downloader UI components dengan proper config integration"""
        try:
            # Create main UI components
            ui_components = create_downloader_main_ui(config)
            
            # Setup logging dengan namespace yang benar
            logger = setup_ipython_logging(
                ui_components, 
                module_name='smartcash.dataset.downloader',
                log_to_file=False,
                redirect_all_logs=False
            )
            ui_components['logger'] = logger
            
            # Set initialization flag untuk namespace detection
            ui_components['enhanced_download_initialized'] = True
            
            return ui_components
            
        except Exception as e:
            logger = get_logger('downloader.init')
            logger.error(f"❌ Error creating downloader UI: {str(e)}")
            return self._create_fallback_ui(str(e))
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup module-specific handlers untuk action buttons dengan proper binding"""
        try:
            logger = ui_components.get('logger')
            
            # Setup download handlers dengan proper binding - update ui_components in place
            updated_components = setup_download_handlers(ui_components, config, env)
            
            # Update ui_components dengan hasil dari setup
            ui_components.update(updated_components)
            
            # Verify handlers setup
            if self._verify_handlers_setup(ui_components, logger):
                logger.success("✅ Action buttons dan handlers berhasil di-setup")
                # Log button status untuk debugging
                button_status = self._get_button_status(ui_components)
                logger.debug(f"🔧 Button Status: {button_status}")
            else:
                logger.warning("⚠️ Beberapa handlers mungkin tidak ter-setup dengan benar")
                self._debug_handler_setup(ui_components, logger)
            
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
            'save_button', 'reset_button', 'log_output', 'progress_tracker'
        ]


# Factory function untuk easy initialization
def create_downloader_initializer() -> DownloaderInitializer:
    """Factory function untuk membuat downloader initializer"""
    return DownloaderInitializer()


def initialize_downloader(env=None, config=None, **kwargs) -> Any:
    """
    Initialize downloader UI dengan fixed action buttons dan config persistence.
    
    Args:
        env: Environment context (optional)
        config: Custom config override (optional)
        **kwargs: Additional initialization parameters
        
    Returns:
        Initialized downloader UI
    """
    try:
        initializer = create_downloader_initializer()
        return initializer.initialize(env, config, **kwargs)
        
    except Exception as e:
        logger = get_logger('downloader.factory')
        logger.error(f"❌ Factory error: {str(e)}")
        
        # Return simple error display
        import ipywidgets as widgets
        return widgets.HTML(f"""
        <div style="padding: 15px; background: #f8d7da; border-radius: 5px; color: #721c24;">
            <h4>❌ Downloader Initialization Failed</h4>
            <p>Error: {str(e)}</p>
        </div>
        """)


# One-liner utilities untuk debugging
get_downloader_status = lambda ui: f"📊 Buttons: {len([k for k in ui.keys() if k.endswith('_button')])} | Handlers: {len([k for k in ui.keys() if k.endswith('_handler')])} | Config: {'✅' if 'config_handler' in ui else '❌'}"
validate_button_bindings = lambda ui: all(btn in ui and hasattr(ui[btn], '_model_id') for btn in ['download_button', 'check_button', 'cleanup_button'])
check_config_persistence = lambda ui: bool(ui.get('config_handler')) and hasattr(ui.get('config_handler'), 'save_config')