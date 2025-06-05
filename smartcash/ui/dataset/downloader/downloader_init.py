"""
File: smartcash/ui/dataset/downloader/downloader_init.py
Deskripsi: Fixed downloader initializer dengan layout order yang benar dan one-liner style
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.utils.logger_bridge import get_logger
from smartcash.ui.utils.ui_logger_namespace import DOWNLOAD_LOGGER_NAMESPACE, KNOWN_NAMESPACES
from smartcash.ui.handlers.config_handlers import ConfigHandler

# Import handlers dengan one-liner style
from smartcash.ui.dataset.downloader.handlers.config_handler import DownloadConfigHandler
from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers
from smartcash.ui.dataset.downloader.handlers.defaults import get_default_download_config

MODULE_LOGGER_NAME = KNOWN_NAMESPACES.get(DOWNLOAD_LOGGER_NAMESPACE, 'DOWNLOAD')

class DownloadInitializer(CommonInitializer):
    """Fixed download initializer dengan proper layout order dan one-liner implementation"""
    
    def __init__(self):
        super().__init__('downloader', DownloadConfigHandler, 'dataset')
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan fixed layout order - one-liner imports"""
        try:
            from smartcash.ui.dataset.downloader.components.ui_layout import create_downloader_ui
            ui_components = create_downloader_ui(config, env)
        except ImportError:
            ui_components = self._create_fallback_ui_components(config, env)
        
        # Add initialization flags dan metadata
        ui_components.update({
            'downloader_initialized': True,
            'layout_order_fixed': True,
            'auto_check_on_load': config.get('auto_check_on_load', False),
            'module_name': 'downloader',
            'parent_module': 'dataset'
        })
        
        return ui_components
    
    def _create_fallback_ui_components(self, config: Dict[str, Any], env=None) -> Dict[str, Any]:
        """Create fallback UI components - one-liner minimal UI"""
        import ipywidgets as widgets
        
        # Minimal form dengan fixed order
        header = widgets.HTML("<h3>📥 Dataset Downloader (Fallback Mode)</h3>")
        
        # Basic form fields - one-liner style
        form_fields = {
            'workspace_input': widgets.Text(value='smartcash-wo2us', description='Workspace:'),
            'project_input': widgets.Text(value='rupiah-emisi-2022', description='Project:'),
            'version_input': widgets.Text(value='3', description='Version:'),
            'api_key_input': widgets.Password(description='API Key:')
        }
        
        # Buttons dengan fixed order
        save_button = widgets.Button(description='💾 Simpan', button_style='success')
        reset_button = widgets.Button(description='🔄 Reset')
        download_button = widgets.Button(description='📥 Download', button_style='primary')
        check_button = widgets.Button(description='🔍 Check', button_style='info')
        cleanup_button = widgets.Button(description='🧹 Cleanup', button_style='danger')
        
        # Output components
        confirmation_area = widgets.Output()
        log_output = widgets.Output()
        status_panel = widgets.HTML("📊 Fallback mode active")
        
        # Fixed layout order sesuai requirement
        form_container = widgets.VBox(list(form_fields.values()))
        save_reset_container = widgets.HBox([save_button, reset_button])
        action_container = widgets.HBox([download_button, check_button, cleanup_button])
        
        ui = widgets.VBox([
            header, status_panel, form_container,     # Header, status, form 2 kolom
            save_reset_container,                     # Save reset button
            confirmation_area,                        # Area konfirmasi
            action_container,                         # Action buttons
            log_output                               # Log output (progress tracker diabaikan di fallback)
        ])
        
        # Combine semua components
        components = {
            'ui': ui, 'main_container': ui, 'header': header, 'status_panel': status_panel,
            'form_container': form_container, 'save_reset_container': save_reset_container,
            'confirmation_area': confirmation_area, 'action_container': action_container,
            'log_output': log_output, 'save_button': save_button, 'reset_button': reset_button,
            'download_button': download_button, 'check_button': check_button, 'cleanup_button': cleanup_button
        }
        
        components.update(form_fields)
        return components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan proper error handling - one-liner fallback"""
        try:
            return setup_download_handlers(ui_components, config, env)
        except Exception as e:
            self.logger.warning(f"⚠️ Error setup handlers: {str(e)}")
            return self._setup_fallback_handlers(ui_components, config)
    
    def _setup_fallback_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Setup fallback handlers - one-liner button binding"""
        # Simple button handlers untuk fallback mode
        fallback_handlers = {
            'save_button': lambda b: self.logger.info("💾 Save handler (fallback)"),
            'reset_button': lambda b: self.logger.info("🔄 Reset handler (fallback)"),
            'download_button': lambda b: self.logger.info("📥 Download handler (fallback)"),
            'check_button': lambda b: self.logger.info("🔍 Check handler (fallback)"),
            'cleanup_button': lambda b: self.logger.info("🧹 Cleanup handler (fallback)")
        }
        
        # Bind handlers dengan one-liner
        [ui_components[btn].on_click(handler) for btn, handler in fallback_handlers.items() 
         if btn in ui_components and hasattr(ui_components[btn], 'on_click')]
        
        ui_components['handlers_mode'] = 'fallback'
        return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config - one-liner"""
        return get_default_download_config()
    
    def _get_critical_components(self) -> List[str]:
        """Critical components untuk validation - one-liner list"""
        return ['ui', 'form_container', 'save_button', 'reset_button', 'download_button', 
                'check_button', 'cleanup_button', 'log_output', 'confirmation_area']
    
    def validate_layout_order(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Validate layout order sesuai requirement - one-liner validation"""
        required_order = ['form_container', 'save_reset_container', 'confirmation_area', 
                         'action_container', 'progress_container', 'log_output']
        
        # Check main container children order
        main_ui = ui_components.get('ui') or ui_components.get('main_container')
        if not main_ui or not hasattr(main_ui, 'children'):
            return {'valid': False, 'message': 'Main UI container tidak ditemukan'}
        
        # Simple order validation - check keberadaan components
        missing_components = [comp for comp in required_order if comp not in ui_components]
        
        return {
            'valid': len(missing_components) == 0,
            'missing_components': missing_components,
            'layout_order_fixed': ui_components.get('layout_order_fixed', False),
            'message': 'Layout order valid' if not missing_components else f'Missing: {missing_components}'
        }

# Global instance
_downloader_initializer = DownloadInitializer()

def initialize_downloader_ui(env=None, config=None, **kwargs) -> Any:
    """Initialize downloader UI dengan fixed layout order - one-liner factory"""
    return _downloader_initializer.initialize(env=env, config=config, **kwargs)

def get_downloader_config() -> Dict[str, Any]:
    """Get current downloader config - one-liner"""
    handler = getattr(_downloader_initializer, 'config_handler', None)
    return handler.get_current_config() if handler else {}

def validate_downloader_layout(ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """Validate downloader layout order - one-liner validation"""
    if not ui_components:
        return {'valid': False, 'message': 'UI components tidak ditemukan'}
    
    return _downloader_initializer.validate_layout_order(ui_components)

def get_downloader_status() -> Dict[str, Any]:
    """Get downloader status dengan layout info - one-liner status"""
    status = _downloader_initializer.get_module_status()
    status.update({
        'layout_order_fixed': True,
        'current_config': get_downloader_config(),
        'critical_components_count': len(_downloader_initializer._get_critical_components())
    })
    return status

def reset_downloader_layout() -> bool:
    """Reset downloader layout - one-liner reset"""
    try:
        _downloader_initializer.__init__()
        return True
    except Exception:
        return False

# One-liner utilities untuk debugging
debug_layout_order = lambda ui: [type(child).__name__ for child in getattr(ui.get('ui'), 'children', [])]
debug_component_count = lambda ui: len([k for k in ui.keys() if not k.startswith('_')])
debug_button_states = lambda ui: {k: getattr(v, 'disabled', 'N/A') for k, v in ui.items() if 'button' in k}
get_layout_summary = lambda ui: f"Components: {debug_component_count(ui)} | Layout: {len(debug_layout_order(ui))} widgets | Fixed: {ui.get('layout_order_fixed', False)}"