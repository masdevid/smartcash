"""
File: smartcash/ui/pretrained/components/ui_components.py
Deskripsi: UI components untuk pretrained models dengan konsistensi module preprocessing
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from smartcash.ui.components import (
    create_header, create_action_buttons, create_status_panel,
    create_log_accordion, create_save_reset_buttons,
    create_dual_progress_tracker, create_confirmation_area
)

def create_pretrained_main_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Create pretrained models UI dengan konsistensi module preprocessing"""
    config = config or {}
    ui_components = {
        'ui': widgets.VBox([]),
        'ui_initialized': False
    }
    
    # Initialize UI components
    from smartcash.ui.components.action_section import create_action_section
    
    # Create header and status
    header = create_header(
        "Pretrained Models",
        "Manajemen model pretrained untuk deteksi mata uang",
        "🤖"
    )
    
    status_panel = create_status_panel(
        "🚀 Siap mengelola model pretrained",
        "info"
    )
    
    # Create input options
    input_components = create_pretrained_input_options(config.get('pretrained_models', {}))
    
    # Create action buttons
    action_components = create_action_buttons(
        primary_button={
            "label": "📥 Download Model",
            "style": "success",
            "width": "180px",
            "key": "download_button"
        },
        secondary_buttons=[
            {
                "label": "🔄 Sync Model", 
                "style": "info", 
                "width": "150px",
                "key": "sync_button"
            }
        ]
    )
    
    # Initialize progress tracker
    progress_tracker = create_dual_progress_tracker(
        operation="Pretrained Models",
        auto_hide=False
    )
    
    # Create log components
    log_components = create_log_accordion(
        module_name='pretrained',
        height='200px'
    )
    
    # Create confirmation area
    confirmation_area = create_confirmation_area(ui_components=ui_components)
    
    # Create save/reset buttons
    save_reset_components = create_save_reset_buttons(
        save_label="Simpan",
        reset_label="Reset"
    )
    
    # Create main UI layout
    main_ui = widgets.VBox([
        header,
        status_panel,
        input_components['ui'],
        widgets.VBox([
            widgets.Box([save_reset_components['container']], 
                layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%'))
        ], layout=widgets.Layout(margin='8px 0')),
        widgets.VBox([action_components.get('container', widgets.VBox([]))]),
        progress_tracker.container if hasattr(progress_tracker, 'container') else widgets.VBox([]),
        log_components.get('log_accordion', widgets.VBox([]))
    ], layout=widgets.Layout(
        width='100%',
        max_width='1280px',
        margin='0 auto',
        padding='15px',
        border='1px solid #e0e0e0',
        border_radius='8px',
        box_shadow='0 2px 4px rgba(0,0,0,0.05)'
    ))
    
    # Update UI components with references to all interactive elements
    ui_components.update({
        'ui': main_ui,
        'header': header,
        'status_panel': status_panel,
        'progress_tracker': progress_tracker,
        'log_output': log_components.get('log_output'),
        'download_button': action_components.get('primary_button'),
        'sync_button': action_components.get('secondary_buttons', [{}])[0].get('button') if action_components.get('secondary_buttons') else None,
        'progress_tracker': progress_tracker,
        'confirmation_area': confirmation_area,
        'input_options': input_components,
        'save_button': save_reset_components.get('save_button'),
        'reset_button': save_reset_components.get('reset_button'),
        'download_btn': action_components.get('primary'),
        'sync_btn': action_components.get('secondary_0'),
        'cleanup_btn': action_components.get('secondary_1'),
        'ui_initialized': True
    })
    
    return ui_components


def create_pretrained_input_options(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create input options untuk konfigurasi pretrained models"""
    # Default values
    default_values = {
        'model_dir': '/data/pretrained',
        'model_type': 'yolov5s'  # Hardcoded to yolov5s
    }
    
    # Apply config values
    config = {**default_values, **config}
    
    from smartcash.ui.pretrained.handlers.defaults import DEFAULT_MODEL_URLS
    
    # Create input components
    model_dir_input = widgets.Text(
        value=config.get('models_dir', '/data/pretrained'),
        description='Model Directory:',
        placeholder='/data/pretrained',
        style={'description_width': '120px'},
        layout={'width': '100%'}
    )
    
    # URL Inputs
    yolo_url_input = widgets.Text(
        value=config.get('model_urls', {}).get('yolov5s', DEFAULT_MODEL_URLS['yolov5s']),
        description='YOLOv5 URL:',
        placeholder='https://...',
        style={'description_width': '120px'},
        layout={'width': '100%'}
    )
    
    efficientnet_url_input = widgets.Text(
        value=config.get('model_urls', {}).get('efficientnet', DEFAULT_MODEL_URLS['efficientnet']),
        description='EfficientNet URL:',
        placeholder='https://...',
        style={'description_width': '120px'},
        layout={'width': '100%'}
    )
    
    # Create layout
    input_ui = widgets.VBox([
        widgets.HTML("<h4>📁 Model Configuration</h4>"),
        model_dir_input,
        widgets.HTML(
            "<div style='margin: 10px 0 5px 8px; color: #666;'>"
            "Model Type: <b>yolov5s</b></div>"
        ),
        widgets.HTML(
            "<div style='margin: 15px 0 5px 0; font-weight: bold;'>"
            "Custom Download URLs:</div>"
        ),
        yolo_url_input,
        efficientnet_url_input,
        widgets.HTML(
            "<div style='margin-top: 5px; font-size: 0.9em; color: #666;'>"
            "Biarkan kosong untuk menggunakan URL default"
            "</div>"
        )
    ])
    
    return {
        'ui': input_ui,
        'model_dir_input': model_dir_input,
        'yolo_url_input': yolo_url_input,
        'efficientnet_url_input': efficientnet_url_input
    }

