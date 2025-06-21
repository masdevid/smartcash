# File: smartcash/ui/pretrained/components/ui_components.py
"""
File: smartcash/ui/pretrained/components/ui_components.py
Deskripsi: UI components untuk pretrained models menggunakan reusable shared components dari ui/components/
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
import traceback

logger = get_logger(__name__)

def create_pretrained_ui_components(env=None, config: Optional[Dict] = None, **kwargs) -> Dict:
    """🎯 Create pretrained UI menggunakan shared reusable components
    
    Args:
        env: Environment configuration (optional)
        config: Configuration dictionary (optional)
        **kwargs: Additional keyword arguments
            - exc_info: Optional exception info tuple (type, value, traceback)
    """
    exc_info = kwargs.pop('exc_info', None)
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
        
        # Import shared reusable components
        from smartcash.ui.utils.header_utils import create_header
        from smartcash.ui.components.status_panel import create_status_panel
        from smartcash.ui.components.progress_tracker.factory import create_dual_progress_tracker
        from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
        from smartcash.ui.components.log_accordion import create_log_accordion
        from smartcash.ui.components.action_buttons import create_action_buttons
        
        # 1. 📊 Header - shared component
        header = create_header(
            "🤖 Pretrained Models Manager",
            "Download dan sync pretrained models untuk YOLOv5 + EfficientNet-B4",
            "🚀"
        )
        
        # 2. 📋 Status Panel - shared component
        status_panel = create_status_panel(
            "🔄 Ready untuk setup pretrained models",
            "info"
        )
        
        # 3. 📈 Progress Tracker - shared component
        progress_tracker = create_dual_progress_tracker(
            operation="Pretrained Models Setup",
            auto_hide=True
        )
        
        # 4. 🔧 Input Options (module-specific minimal)
        input_options = _create_pretrained_input_options(pretrained_config)
        
        # 5. 🎯 Action Buttons - shared component
        download_sync_button = widgets.Button(
            description="📥 Download & Sync Models",
            button_style='primary',
            icon='download',
            layout=widgets.Layout(width='200px', height='35px')
        )
        action_buttons = create_action_buttons([download_sync_button])
        
        # 6. 💾 Save/Reset Buttons - shared component
        save_reset_buttons = create_save_reset_buttons()
        
        # 7. 📝 Log Output - shared component
        log_output = widgets.Output(
            layout=widgets.Layout(
                width='100%',
                min_height='100px',
                max_height='300px',
                border='1px solid #ddd',
                padding='10px'
            )
        )
        log_accordion = create_log_accordion(
            log_output=log_output,
            title="📋 Pretrained Setup Logs",
            selected_index=None
        )
        
        # 8. 💬 Dialog Area (minimal)
        confirmation_area = widgets.Output(layout=widgets.Layout(
            border='1px solid #ddd',
            padding='10px',
            margin='10px 0',
            display='none'
        ))
        
        # 9. 📦 Main Container Assembly
        main_container = widgets.VBox([
            header,
            status_panel,
            progress_tracker,
            input_options['container'],
            action_buttons,
            save_reset_buttons['container'],
            log_accordion,
            confirmation_area
        ], layout=widgets.Layout(padding='10px', width='100%'))
        
        # 10. 🔧 Return Complete UI Components
        ui_components = {
            # Core UI
            'ui': main_container,
            'main_container': main_container,
            
            # Shared components
            'header': header,
            'status_panel': status_panel,
            'progress_tracker': progress_tracker,
            'log_output': log_output,
            'log_accordion': log_accordion,
            'confirmation_area': confirmation_area,
            'dialog_area': confirmation_area,
            
            # Action buttons
            'download_sync_button': download_sync_button,
            'action_buttons': action_buttons,
            
            # Save/Reset (from shared component)
            'save_button': save_reset_buttons['save_button'],
            'reset_button': save_reset_buttons['reset_button'],
            'save_reset_buttons': save_reset_buttons,
            
            # Input widgets (module-specific)
            'models_dir_input': input_options.get('models_dir_input'),
            'drive_models_dir_input': input_options.get('drive_models_dir_input'),
            'pretrained_type_dropdown': input_options.get('pretrained_type_dropdown'),
            'auto_download_checkbox': input_options.get('auto_download_checkbox'),
            'sync_drive_checkbox': input_options.get('sync_drive_checkbox'),
            'input_options': input_options,
            
            # Metadata
            'module_name': 'pretrained_models',
            'config': config,
            'ui_initialized': True,
            'api_integration': True,
            'dialog_support': True,
            'progress_tracking': True
        }
        
        return ui_components
        
    except Exception as e:
        import sys
        logger.error("❌ Error creating pretrained UI", exc_info=True)
        return _create_fallback_ui(
            f"Error creating pretrained UI: {str(e)}",
            exc_info=sys.exc_info()
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

def _create_fallback_ui(error_msg: str, exc_info: tuple = None) -> Dict[str, Any]:
    """Create fallback UI with detailed error information
    
    Args:
        error_msg: Error message to display
        exc_info: Optional exception info tuple (type, value, traceback)
    """
    import traceback
    import html
    import sys
    
    # Format the error message and traceback
    error_type = "Error"
    error_traceback = ""
    
    if exc_info:
        ex_type, ex_value, ex_traceback = exc_info
        error_type = ex_type.__name__
        error_traceback = "".join(traceback.format_exception(ex_type, ex_value, ex_traceback))
    
    # Escape HTML special characters
    safe_error_msg = html.escape(str(error_msg))
    safe_traceback = html.escape(error_traceback) if error_traceback else "No traceback available"
    
    # Create collapsible error details
    error_widget = widgets.HTML(f"""
        <div style='padding:20px;border:2px solid #dc3545;border-radius:8px;background:#f8d7da;color:#721c24;'>
            <h4 style='margin-top:0;'>⚠️ Pretrained UI Creation Error: {error_type}</h4>
            <p><strong>Error:</strong> {safe_error_msg}</p>
            
            <details style='margin-top:10px;'>
                <summary style='cursor:pointer;color:#007bff;font-weight:bold;'>
                    Show error details
                </summary>
                <div style='margin-top:10px;padding:10px;background:rgba(0,0,0,0.05);border-radius:4px;'>
                    <pre style='white-space:pre-wrap;margin:0;overflow:auto;'>{safe_traceback}</pre>
                </div>
            </details>
            
            <div style='margin-top:15px;padding:10px;background:rgba(0,0,0,0.03);border-radius:4px;'>
                <p style='margin:5px 0;'>💡 <strong>Tips:</strong></p>
                <ul style='margin:5px 0 0 20px;padding:0;'>
                    <li>Restart the kernel and try again</li>
                    <li>Check if all required dependencies are installed</li>
                    <li>Verify the configuration values</li>
                    <li>Check the logs for more details</li>
                </ul>
            </div>
        </div>
    """)
    
    return {
        'ui': widgets.VBox([error_widget]),
        'main_container': widgets.VBox([error_widget]),
        'status': widgets.HTML(f"<div style='color:#dc3545;'>❌ {error_msg}</div>"),
        'log_output': widgets.Output(),
        'confirmation_area': widgets.Output(),
        'dialog_area': widgets.Output(),
        'download_sync_button': widgets.Button(description="Error", disabled=True),
        'save_button': widgets.Button(description="Error", disabled=True),
        'reset_button': widgets.Button(description="Error", disabled=True),
        'progress_tracker': None,
        'module_name': 'pretrained_models',
        'ui_initialized': False,
        'error': error_msg,
        'fallback_mode': True
    }