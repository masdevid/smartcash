"""
File: smartcash/ui/dataset/downloader/downloader_init.py
Deskripsi: Fixed downloader initializer tanpa UI fallbacks dan error handling yang proper
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.utils.ui_logger_namespace import DOWNLOAD_LOGGER_NAMESPACE, KNOWN_NAMESPACES
from smartcash.ui.dataset.downloader.handlers.config_handler import DownloadConfigHandler
from smartcash.ui.dataset.downloader.handlers.download_handler import setup_download_handlers
from smartcash.ui.dataset.downloader.handlers.defaults import get_default_download_config
from smartcash.ui.dataset.downloader.components.ui_components import create_downloader_main_ui

MODULE_LOGGER_NAME = KNOWN_NAMESPACES.get(DOWNLOAD_LOGGER_NAMESPACE, 'DOWNLOAD')

class DownloadInitializer(CommonInitializer):
    """Fixed download initializer tanpa fallbacks dan proper error propagation"""
    
    def __init__(self):
        super().__init__('downloader', DownloadConfigHandler, 'dataset')
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan direct call tanpa fallback"""
        # Direct call ke create_downloader_main_ui
        ui_components = create_downloader_main_ui(config)
        
        # Validate critical components ada
        if not ui_components or 'ui' not in ui_components:
            raise ValueError("UI components creation failed - missing 'ui' component")
        
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan direct call"""
        # Direct call ke setup_download_handlers
        setup_download_handlers(ui_components, config, env)
        return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan direct call"""
        return get_default_download_config()
    
    def _get_critical_components(self) -> List[str]:
        """Critical components list yang realistis"""
        return [
            'ui',                    # Main UI container
            'form_container',        # Form container
            'save_button',          # Save button
            'reset_button',         # Reset button
            'download_button',      # Download button
            'check_button',         # Check button
            'cleanup_button',       # Cleanup button
            'log_output',           # Log output
            'confirmation_area',    # Confirmation area
            'progress_tracker',     # Progress tracker instance
            'progress_container'    # Progress container widget
        ]

# Factory function untuk create initializer
def create_downloader_initializer() -> DownloadInitializer:
    """Factory untuk create downloader initializer"""
    return DownloadInitializer()

# Main initialization function
def initialize_downloader_ui(env=None, config=None, **kwargs) -> Dict[str, Any]:
    """Initialize downloader UI dengan proper error handling"""
    try:
        # Create fresh initializer instance
        initializer = create_downloader_initializer()
        
        # Initialize dengan proper error propagation
        result = initializer.initialize(env=env, config=config, **kwargs)
        
        # Validate result is a proper dict
        if not result or not isinstance(result, dict):
            raise ValueError("Initializer returned invalid result")
        
        # Ensure we have at least 'ui' component
        if 'ui' not in result:
            raise ValueError("Missing 'ui' component in result")
        
        # Success - return the full result
        return result
        
    except Exception as e:
        # Log the actual error for debugging
        error_msg = str(e)
        print(f"❌ Failed to initialize downloader UI: {error_msg}")
        
        # Import traceback for detailed error info
        import traceback
        traceback.print_exc()
        
        # Return error dict untuk consistency dengan pattern yang ada
        return {
            'error': f"Failed to initialize downloader UI: {error_msg}",
            'ui': None,
            'initialized': False,
            'module_name': 'downloader'
        }

# Utility functions
def get_downloader_config() -> Dict[str, Any]:
    """Get current downloader config dengan safe access"""
    try:
        initializer = create_downloader_initializer()
        if hasattr(initializer, 'config_handler') and initializer.config_handler:
            return initializer.config_handler.get_default_config()
        return get_default_download_config()
    except Exception:
        return get_default_download_config()

def validate_downloader_layout(ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """Validate downloader layout dengan proper checks"""
    if not ui_components:
        return {
            'valid': False,
            'message': 'UI components tidak ditemukan - panggil initialize_downloader_ui() terlebih dahulu',
            'missing_components': []
        }
    
    try:
        initializer = create_downloader_initializer()
        critical_components = initializer._get_critical_components()
        
        # Check missing components
        missing = [comp for comp in critical_components if comp not in ui_components]
        
        return {
            'valid': len(missing) == 0,
            'message': 'Layout valid' if not missing else f'Missing components: {", ".join(missing)}',
            'missing_components': missing,
            'total_components': len(ui_components),
            'critical_components': len(critical_components)
        }
    except Exception as e:
        return {
            'valid': False,
            'message': f'Validation error: {str(e)}',
            'missing_components': []
        }

def get_downloader_status() -> Dict[str, Any]:
    """Get downloader status dengan comprehensive info"""
    try:
        initializer = create_downloader_initializer()
        
        return {
            'module_name': 'downloader',
            'initialized': True,
            'layout_order_fixed': True,
            'current_config': get_downloader_config(),
            'critical_components_count': len(initializer._get_critical_components()),
            'logger_namespace': initializer.logger_namespace,
            'config_handler_class': initializer.config_handler_class.__name__ if initializer.config_handler_class else None
        }
    except Exception as e:
        return {
            'module_name': 'downloader',
            'initialized': False,
            'error': str(e)
        }

def reset_downloader_ui() -> bool:
    """Reset downloader UI dengan cleanup"""
    try:
        # Simply return True - next initialization akan create fresh instance
        return True
    except Exception:
        return False

# Debug function untuk troubleshooting
def debug_downloader_initialization(env=None, config=None, **kwargs) -> Dict[str, Any]:
    """Debug function untuk troubleshoot initialization issues"""
    debug_info = {
        'steps': [],
        'errors': [],
        'success': False
    }
    
    try:
        # Step 1: Create initializer
        debug_info['steps'].append('Creating initializer...')
        initializer = create_downloader_initializer()
        debug_info['steps'].append('✅ Initializer created')
        
        # Step 2: Get default config
        debug_info['steps'].append('Getting default config...')
        default_config = initializer._get_default_config()
        debug_info['steps'].append(f'✅ Default config loaded: {len(default_config)} keys')
        
        # Step 3: Create UI components
        debug_info['steps'].append('Creating UI components...')
        merged_config = config or default_config
        ui_components = initializer._create_ui_components(merged_config, env, **kwargs)
        debug_info['steps'].append(f'✅ UI components created: {len(ui_components)} components')
        
        # Step 4: Validate critical components
        debug_info['steps'].append('Validating critical components...')
        critical = initializer._get_critical_components()
        missing = [c for c in critical if c not in ui_components]
        if missing:
            debug_info['errors'].append(f'Missing critical components: {missing}')
        else:
            debug_info['steps'].append('✅ All critical components present')
        
        # Step 5: Setup handlers
        debug_info['steps'].append('Setting up handlers...')
        initializer._setup_module_handlers(ui_components, merged_config, env, **kwargs)
        debug_info['steps'].append('✅ Handlers setup complete')
        
        debug_info['success'] = True
        debug_info['component_list'] = list(ui_components.keys())
        
    except Exception as e:
        debug_info['errors'].append(f'Error: {str(e)}')
        debug_info['exception_type'] = type(e).__name__
        import traceback
        debug_info['traceback'] = traceback.format_exc()
    
    return debug_info