# File: smartcash/ui/pretrained/components/ui_components.py
"""
File: smartcash/ui/pretrained/components/ui_components.py
Deskripsi: UI components untuk pretrained models - Fixed version dengan complete implementation
"""

# Standard library imports
import sys
import traceback
from typing import Dict, Any, Optional, List

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
    """🎯 Create pretrained UI menggunakan shared reusable components
    
    Args:
        env: Environment configuration (optional)
        config: Configuration dictionary (optional)
        **kwargs: Additional keyword arguments
            
    Returns:
        Dictionary berisi komponen UI atau fallback UI jika terjadi error
    """
    module_name = kwargs.get('module_name', 'pretrained_ui')
    logger = get_logger(module_name)
    
    def _create_ui() -> Dict:
        """🛠️ Fungsi internal untuk membuat UI components"""
        try:
            # 🔧 Safe config handling
            if config is None:
                config = {}
            if not isinstance(config, dict):
                config = {}
            
            # Ensure pretrained_models section exists
            if 'pretrained_models' not in config:
                config['pretrained_models'] = {}
            
            pretrained_config = config.get('pretrained_models', {})
            
            # 📝 Create input options
            input_options = _create_pretrained_input_options(pretrained_config)
            
            # 🎨 Create UI components dengan error handling per component
            ui_components = {}
            
            # Header
            try:
                ui_components['header'] = create_header(
                    title="🤖 Pretrained Models Configuration",
                    subtitle="Setup YOLOv5 dan EfficientNet-B4 untuk deteksi mata uang"
                )
            except Exception as e:
                logger.warning(f"Header creation failed: {e}")
                ui_components['header'] = widgets.HTML("<h3>🤖 Pretrained Models</h3>")
            
            # Status panel
            try:
                ui_components['status'] = create_status_panel()
            except Exception as e:
                logger.warning(f"Status panel creation failed: {e}")
                ui_components['status'] = widgets.HTML("<div>Status: Ready</div>")
            
            # Progress tracker
            try:
                ui_components['progress_tracker'] = create_dual_progress_tracker(
                    primary_label="Model Download",
                    secondary_label="Drive Sync"
                )
            except Exception as e:
                logger.warning(f"Progress tracker creation failed: {e}")
                ui_components['progress_tracker'] = widgets.Output()
            
            # Action buttons
            try:
                button_configs = [
                    {'name': 'download_sync', 'label': '📥 Download & Sync Models', 'style': 'primary'},
                    {'name': 'save', 'label': '💾 Simpan Config', 'style': 'success'},
                    {'name': 'reset', 'label': '🔄 Reset', 'style': 'warning'}
                ]
                buttons = create_action_buttons(button_configs)
                for button_config in button_configs:
                    ui_components[f"{button_config['name']}_button"] = buttons[button_config['name']]
            except Exception as e:
                logger.warning(f"Action buttons creation failed: {e}")
                # Fallback manual buttons
                ui_components['download_sync_button'] = widgets.Button(description='📥 Download & Sync')
                ui_components['save_button'] = widgets.Button(description='💾 Simpan')
                ui_components['reset_button'] = widgets.Button(description='🔄 Reset')
            
            # Log accordion
            try:
                ui_components['log_accordion'] = create_log_accordion()
                ui_components['log_output'] = ui_components['log_accordion']['log_output']
            except Exception as e:
                logger.warning(f"Log accordion creation failed: {e}")
                ui_components['log_output'] = widgets.Output()
                ui_components['log_accordion'] = {'accordion': widgets.Accordion([ui_components['log_output']])}
            
            # Dialog area (for confirmations)
            ui_components['confirmation_area'] = widgets.VBox(
                layout=widgets.Layout(display='none')
            )
            
            # 📋 Create main layout
            try:
                # Input form section
                input_form = widgets.VBox([
                    widgets.HTML("<h4>📝 Konfigurasi Model</h4>"),
                    input_options['models_dir_text'],
                    input_options['drive_models_dir_text'],
                    input_options['pretrained_type_dropdown'],
                    input_options['auto_download_checkbox'],
                    input_options['sync_drive_checkbox']
                ])
                
                # Action section
                action_section = widgets.HBox([
                    ui_components['download_sync_button'],
                    ui_components['save_button'],
                    ui_components['reset_button']
                ])
                
                # Main container
                ui_components['main_container'] = widgets.VBox([
                    ui_components['header'],
                    ui_components['status'],
                    input_form,
                    action_section,
                    ui_components['progress_tracker'],
                    ui_components['confirmation_area'],
                    ui_components['log_accordion'].get('accordion', ui_components['log_output'])
                ])
                
            except Exception as e:
                logger.error(f"Layout creation failed: {e}")
                # Simple fallback layout
                ui_components['main_container'] = widgets.VBox([
                    widgets.HTML("<h3>🤖 Pretrained Models</h3>"),
                    ui_components.get('download_sync_button', widgets.Button(description='Download')),
                    ui_components.get('log_output', widgets.Output())
                ])
            
            # 🔗 Add dialog helper functions
            ui_components.update({
                'show_confirmation_dialog': lambda **kwargs: show_confirmation_dialog(
                    ui_components=ui_components,
                    **{
                        'title': 'Konfirmasi',
                        'message': 'Yakin ingin melanjutkan?',
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
                'clear_dialog_area': lambda: clear_dialog_area(ui_components),
                'is_dialog_visible': lambda: is_dialog_visible(ui_components),
                'input_options': input_options,
                'module_name': 'pretrained_models',
                'config': config,
                'ui_initialized': True
            })
            
            logger.info("✅ UI components berhasil dibuat")
            return ui_components
            
        except Exception as e:
            error_msg = f"❌ Gagal membuat komponen UI: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return _create_fallback_ui(
                error_msg=error_msg,
                exc_info=sys.exc_info(),
                module_name=module_name
            )
    
    # 🛡️ Use try_operation_safe untuk robust error handling
    return try_operation_safe(
        operation=_create_ui,
        fallback_value={},
        logger=logger,
        operation_name="create_pretrained_ui_components",
        exc_info=True,
        on_error=lambda e, *args: logger.error(f"🚨 Critical error in UI creation: {str(e)}")
    )

def _create_pretrained_input_options(pretrained_config: Dict[str, Any]) -> Dict[str, widgets.Widget]:
    """📝 Create input form widgets untuk pretrained configuration
    
    Args:
        pretrained_config: Konfigurasi pretrained models dari config
        
    Returns:
        Dictionary berisi widget input forms
    """
    try:
        # Safe defaults
        models_dir = pretrained_config.get('models_dir', '/content/models')
        drive_models_dir = pretrained_config.get('drive_models_dir', '/content/drive/MyDrive/SmartCash/models')
        pretrained_type = pretrained_config.get('pretrained_type', 'yolov5s')
        auto_download = pretrained_config.get('auto_download', False)
        sync_drive = pretrained_config.get('sync_drive', True)
        
        return {
            'models_dir_text': widgets.Text(
                value=models_dir,
                description='📁 Models Dir:',
                style={'description_width': '120px'},
                layout=widgets.Layout(width='500px')
            ),
            'drive_models_dir_text': widgets.Text(
                value=drive_models_dir,
                description='☁️ Drive Models:',
                style={'description_width': '120px'},
                layout=widgets.Layout(width='500px')
            ),
            'pretrained_type_dropdown': widgets.Dropdown(
                options=[
                    ('YOLOv5s (Ringan)', 'yolov5s'),
                    ('YOLOv5m (Medium)', 'yolov5m'),
                    ('YOLOv5l (Besar)', 'yolov5l'),
                    ('YOLOv5x (Extra Besar)', 'yolov5x')
                ],
                value=pretrained_type,
                description='🎯 Model Type:',
                style={'description_width': '120px'},
                layout=widgets.Layout(width='300px')
            ),
            'auto_download_checkbox': widgets.Checkbox(
                value=auto_download,
                description='🔄 Auto Download',
                style={'description_width': '120px'}
            ),
            'sync_drive_checkbox': widgets.Checkbox(
                value=sync_drive,
                description='☁️ Sync ke Drive',
                style={'description_width': '120px'}
            )
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating input options: {str(e)}")
        # Return minimal fallback
        return {
            'models_dir_text': widgets.Text(value='/content/models', description='Models Dir:'),
            'drive_models_dir_text': widgets.Text(value='/content/drive/MyDrive/SmartCash/models', description='Drive Dir:'),
            'pretrained_type_dropdown': widgets.Dropdown(options=[('YOLOv5s', 'yolov5s')], description='Type:'),
            'auto_download_checkbox': widgets.Checkbox(value=False, description='Auto Download'),
            'sync_drive_checkbox': widgets.Checkbox(value=True, description='Sync Drive')
        }