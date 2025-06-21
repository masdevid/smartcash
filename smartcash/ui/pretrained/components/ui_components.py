# File: smartcash/ui/pretrained/components/ui_components.py
"""
File: smartcash/ui/pretrained/components/ui_components.py
Deskripsi: UI components untuk pretrained models menggunakan reusable shared components dari ui/components/
"""

# Standard library imports
import sys
import traceback
from typing import Dict, Any, Optional, Tuple, Callable

# Third-party imports
import ipywidgets as widgets

# SmartCash imports
from smartcash.common.logger import get_logger
from smartcash.ui.components.header import create_header
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.progress_tracker.factory import create_dual_progress_tracker
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.components.dialog import (
    show_confirmation_dialog,
    show_info_dialog,
    clear_dialog_area,
    is_dialog_visible
)
from smartcash.ui.utils.fallback_utils import (
    create_fallback_ui as _create_fallback_ui,
    FallbackConfig,
    try_operation_safe
)

logger = get_logger(__name__)

def create_pretrained_ui_components(env=None, config: Optional[Dict] = None, **kwargs) -> Dict:
    """🎯 Create pretrained UI menggunakan shared reusable components dengan error handling yang lebih baik
    
    Args:
        env: Environment configuration (optional)
        config: Configuration dictionary (optional)
        **kwargs: Additional keyword arguments
            - exc_info: Optional exception info tuple (type, value, traceback)
            - module_name: Nama modul untuk logging
            
    Returns:
        Dictionary berisi komponen UI atau fallback UI jika terjadi error
    """
    module_name = kwargs.get('module_name', 'pretrained_ui')
    logger = get_logger(module_name)
    
    def _create_ui() -> Dict:
        """Fungsi pembantu untuk membuat UI dengan error handling"""
        try:
            # Handle None config
            config = config or {}
                
            # Ensure config is a dictionary
            if not isinstance(config, dict):
                config = {}
                
            # Ensure pretrained_models exists and is a dictionary
            if 'pretrained_models' not in config:
                config['pretrained_models'] = {}
                
            if not isinstance(config['pretrained_models'], dict):
                config['pretrained_models'] = {}
            
            # Get pretrained_config with type safety
            pretrained_config = config.get('pretrained_models', {})
            
            # Create input options
            input_options = _create_pretrained_input_options(pretrained_config)
            
            # Create UI components
            ui_components = {
                'header': create_header(
                    title="Pretrained Models",
                    subtitle="Pilih dan konfigurasi model yang telah dilatih sebelumnya",
                    icon='model'
                ),
                'status': create_status_panel(
                    message="Siap digunakan",
                    status="success"
                ),
                'progress': create_dual_progress_tracker(
                    operation="Downloading Model",
                    primary_label="Download",
                    secondary_label="Extraction"
                ),
                'action_buttons': create_action_buttons({
                    'primary': {
                        'label': 'Download & Sync',
                        'icon': 'download',
                        'style': 'primary'
                    },
                    'secondary': {
                        'label': 'Refresh',
                        'icon': 'refresh'
                    }
                }),
                'log_output': create_log_accordion(
                    module_name='pretrained_models',
                    height='200px',
                    width='100%'
                ),
                # Dialog handlers
                'show_confirmation_dialog': lambda **kwargs: show_confirmation_dialog(
                    ui_components=ui_components,
                    **{
                        'title': 'Konfirmasi',
                        'message': 'Apakah Anda yakin?',
                        'confirm_text': 'Ya',
                        'cancel_text': 'Batal',
                        **kwargs
                    }
                ),
                'show_info_dialog': lambda **kwargs: show_info_dialog(
                    ui_components=ui_components,
                    **{
                        'title': 'Informasi',
                        'message': '',
                        'close_text': 'Tutup',
                        'dialog_type': 'info',
                        **kwargs
                    }
                ),
                # Error dialog menggunakan show_info_dialog dengan type error
                'show_error_dialog': lambda **kwargs: show_info_dialog(
                    ui_components=ui_components,
                    **{
                        'title': 'Error',
                        'message': 'Terjadi kesalahan',
                        'close_text': 'Tutup',
                        'dialog_type': 'error',
                        **{k: v for k, v in kwargs.items() if k != 'dialog_type'}
                    }
                ),
                # Fungsi untuk membersihkan dialog
                'clear_dialog_area': lambda: clear_dialog_area(ui_components),
                # Fungsi untuk mengecek visibilitas dialog
                'is_dialog_visible': lambda: is_dialog_visible(ui_components),
                'input_options': input_options,
                'module_name': 'pretrained_models',
                'config': config,
                'ui_initialized': True
            }
            
            return ui_components
            
        except Exception as e:
            error_msg = f"Gagal membuat komponen UI: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return _create_fallback_ui(
                error_msg=error_msg,
                exc_info=sys.exc_info(),
                module_name=module_name
            )
    
    # Gunakan try_operation_safe untuk error handling yang lebih baik
    return try_operation_safe(
        operation=_create_ui,
        fallback_value={},
        logger=logger,
        operation_name="create_pretrained_ui_components",
        exc_info=True,
        on_error=lambda e, **kwargs: _create_fallback_ui(
            error_msg=f"Gagal membuat UI components: {str(e)}",
            exc_info=kwargs.get('exc_info'),
            module_name=module_name
        )
    )

def _create_pretrained_input_options(pretrained_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create minimal input options khusus pretrained (module-specific minimal UI)"""
    # Ensure pretrained_config is a dictionary
    if not isinstance(pretrained_config, dict):
        pretrained_config = {}  # Ensure it's a dict to prevent attribute errors
    
    def safe_get_config_value(config: Dict[str, Any], key: str, default: Any) -> Any:
        """Safely get value from config with type checking"""
        if not isinstance(config, dict):
            return default
        return config.get(key, default)
    
    def safe_convert_to_string(value: Any, default: str = 'yolov5s', context: str = '') -> str:
        """
        Safely convert any value to string, handling lists, tuples, and other types
        
        Args:
            value: The value to convert
            default: Default value if conversion fails
            context: Optional context for error messages (unused in this version)
            
        Returns:
            str: The converted string or default value
        """
        try:
            if value is None:
                return default
                
            if isinstance(value, (list, tuple)):
                if not value:
                    return default
                for item in value:
                    if item is not None:
                        return safe_convert_to_string(item, default, context)
                return default
            
            result = str(value).strip()
            return result if result else default
            
        except Exception:
            return default
    
    def safe_bool(value: Any, default: bool = False) -> bool:
        """Safely convert any value to boolean"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 't', 'y', 'yes')
        if isinstance(value, (int, float)):
            return bool(value)
        return default
    
    try:
        models_dir = safe_convert_to_string(safe_get_config_value(pretrained_config, 'models_dir', '/content/models'))
        drive_models_dir = safe_convert_to_string(safe_get_config_value(pretrained_config, 'drive_models_dir', '/content/drive/MyDrive/SmartCash/models'))
        
        models_dir_input = widgets.Text(
            value=models_dir,
            description='Models Dir:',
            placeholder='Path untuk menyimpan models',
            layout=widgets.Layout(width='100%'),
            style={'description_width': '120px'}
        )
        
        drive_models_dir_input = widgets.Text(
            value=drive_models_dir,
            description='Drive Dir:',
            placeholder='Path Google Drive untuk sync',
            layout=widgets.Layout(width='100%'),
            style={'description_width': '120px'}
        )
        
        allowed_models = ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
        default_model = 'yolov5s'
        final_model = default_model
        
        raw_pretrained_type = safe_get_config_value(pretrained_config, 'pretrained_type', default_model)
        pretrained_type = safe_convert_to_string(raw_pretrained_type, default=default_model)
        
        if pretrained_type.lower() in [m.lower() for m in allowed_models]:
            final_model = pretrained_type
        
        # Final model selection is complete
        
        pretrained_type_dropdown = widgets.Dropdown(
            options=allowed_models,
            value=final_model,
            description='Model Type:',
            style={'description_width': '120px'},
            disabled=False
        )
        
        def on_model_type_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                new_value = change['new']
                if new_value in allowed_models:
                    pretrained_config['pretrained_type'] = new_value
        
        pretrained_type_dropdown.observe(on_model_type_change)
        
        auto_download = safe_bool(pretrained_config.get('auto_download'), False)
        sync_drive = safe_bool(pretrained_config.get('sync_drive'), True)
        
        
        auto_download_checkbox = widgets.Checkbox(
            value=auto_download,
            description='Auto Download',
            tooltip='Download otomatis jika model tidak ditemukan'
        )
        
        sync_drive_checkbox = widgets.Checkbox(
            value=sync_drive,
            description='Sync to Drive',
            tooltip='Sinkronisasi dengan Google Drive'
        )
        
        # Simple container
        container = widgets.VBox([
            widgets.HTML("<h4>⚙️ Configuration</h4>"),
            models_dir_input,
            drive_models_dir_input,
            widgets.HBox([
                pretrained_type_dropdown,
                auto_download_checkbox,
                sync_drive_checkbox
            ])
        ], layout=widgets.Layout(
            padding='10px',
            border='1px solid #ddd',
            border_radius='5px',
            margin='10px 0'
        ))
        
        return {
            'container': container,
            'models_dir_input': models_dir_input,
            'drive_models_dir_input': drive_models_dir_input,
            'pretrained_type_dropdown': pretrained_type_dropdown,
            'auto_download_checkbox': auto_download_checkbox,
            'sync_drive_checkbox': sync_drive_checkbox
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"❌ ❌ Error creating input options: {str(e)}")
        logger.error(f"[DEBUG] Error details: {error_trace}")
        
        # Try to get more context about the error
        if hasattr(e, '__traceback__'):
            tb = e.__traceback__
            while tb.tb_next:
                tb = tb.tb_next
            frame = tb.tb_frame
            logger.error(f"[DEBUG] Error in {frame.f_code.co_filename} at line {frame.f_lineno}")
            logger.error(f"[DEBUG] Local variables: {frame.f_locals}")
        
        return {
            'container': widgets.HTML(f"<div style='color: red;'>Input options error: {str(e)}\n\n{error_trace}</div>")
        }

def _create_fallback_ui(
    error_msg: str, 
    exc_info: Optional[Tuple[type, BaseException, Any]] = None,
    module_name: str = "pretrained_ui",
    show_traceback: bool = True,
    retry_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """Create fallback UI with detailed error information and traceback
    
    Args:
        error_msg: Error message to display
        exc_info: Optional exception info tuple (type, value, traceback)
        module_name: Name of the module where the error occurred
        show_traceback: Whether to show the full traceback
        retry_callback: Optional callback function for retry button
        
    Returns:
        Dictionary containing the fallback UI components
    """
    logger = get_logger(module_name)
    
    # Log the error with traceback
    if exc_info:
        logger.error(f"Error in {module_name}: {error_msg}", exc_info=exc_info)
    else:
        logger.error(f"Error in {module_name}: {error_msg}")
    
    # Format traceback if available
    tb_msg = ""
    if exc_info and show_traceback:
        try:
            tb_msg = "".join(traceback.format_exception(*exc_info))
        except Exception as e:
            tb_msg = f"Error getting traceback: {str(e)}"
    
    # Create fallback configuration
    config = FallbackConfig(
        title=f"⚠️ Error in {module_name}",
        message=error_msg,
        traceback=tb_msg,
        module_name=module_name,
        show_traceback=show_traceback,
        show_retry=retry_callback is not None,
        retry_callback=retry_callback,
        container_style={
            'border': '1px solid #f5c6cb',
            'border_radius': '8px',
            'padding': '15px',
            'margin': '10px 0',
            'background': '#f8d7da',
            'color': '#721c24'
        }
    )
    
    # Create help tips
    help_tips = [
        "Restart the kernel and try again",
        "Check if all required dependencies are installed",
        "Verify your configuration settings",
        "Check the logs for more details"
    ]
    
    # Add help tips to the message
    if show_traceback:
        config.message += "\n\nTroubleshooting tips:\n" + "\n".join(f"• {tip}" for tip in help_tips)
    
    # Create and return the fallback UI
    return _create_fallback_ui(
        error_message=config.message,
        module_name=module_name,
        exc_info=exc_info,
        config=config
    )