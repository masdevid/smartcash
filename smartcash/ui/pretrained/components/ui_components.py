"""
File: smartcash/ui/pretrained/components/ui_components.py
Deskripsi: UI components untuk pretrained models dengan shared container components
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets

# Import standard container components
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components import create_log_accordion, create_save_reset_buttons
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
from smartcash.ui.components.progress_tracker.progress_config import ProgressLevel

def create_pretrained_main_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """🎨 Create pretrained models UI using shared container components"""
    config = config or {}
    
    # Initialize UI components dictionary
    ui_components = {}
    
    # === CORE COMPONENTS ===
    
    # 1. Create Header Container
    header_container = create_header_container(
        title="Pretrained Models",
        subtitle="Manajemen model pretrained untuk deteksi mata uang",
        icon="🤖"
    )
    ui_components['header_container'] = header_container.container
    
    # 2. Create Form Container
    form_container = create_form_container()
    
    # Create input options
    input_components = create_pretrained_input_options(config.get('pretrained_models', {}))
    
    # Place input options in the form container
    form_container['form_container'].children = (input_components['ui'],)
    ui_components['form_container'] = form_container['container']
    
    # 3. Create Footer Container with Log Accordion
    log_components = create_log_accordion('pretrained', '200px')
    footer_container = create_footer_container(
        log_output=log_components['log_output'],
        info_box=widgets.HTML(
            """
            <div class="alert alert-info" style="font-size: 0.9em; padding: 8px 12px;">
                <strong>Tips Model Pretrained:</strong>
                <ul style="margin: 5px 0 0 15px; padding: 0;">
                    <li>Gunakan model yang sesuai dengan kebutuhan deteksi</li>
                    <li>Pastikan direktori model valid dan dapat diakses</li>
                    <li>Sinkronisasi model untuk memastikan versi terbaru</li>
                </ul>
            </div>
            """
        )
    )
    ui_components['footer_container'] = footer_container.container
    
    # 4. Create Action Container with standard approach
    action_container = create_action_container(
        buttons=[
            {
                "button_id": "download",
                "text": "📥 Download Model",
                "style": "primary",
                "order": 1
            },
            {
                "button_id": "sync",
                "text": "🔄 Sync Model",
                "style": "info",
                "order": 2
            }
        ],
        title="🤖 Model Operations",
        alignment="left"
    )
    ui_components['action_container'] = action_container.container
    
    # 5. Create Progress Tracker
    progress_tracker = ProgressTracker(
        title="Pretrained Models",
        levels=[ProgressLevel.OVERALL, ProgressLevel.CURRENT],
        auto_hide=False
    )
    ui_components['progress_tracker'] = progress_tracker
    
    # 6. Create save/reset buttons
    save_reset_components = create_save_reset_buttons(
        save_label="Simpan",
        reset_label="Reset"
    )
    
    # Create config buttons container with proper styling
    config_buttons_container = widgets.Box(
        [save_reset_components['container']], 
        layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%', margin='8px 0')
    )
    
    # 7. Assemble the main UI using shared container approach
    main_container = create_main_container(
        header_container=ui_components['header_container'],
        form_container=ui_components['form_container'],
        footer_container=ui_components['footer_container'],
        additional_components=[
            config_buttons_container,
            ui_components['action_container'],
            progress_tracker.container
        ]
    )
    
    # Set main UI container
    ui_components['ui'] = main_container
    
    # Update UI components with references to all interactive elements
    ui_components.update({
        # Standard container references
        'main_container': main_container,
        
        # Button references with standard approach
        'download_button': action_container.get_button('download'),
        'sync_button': action_container.get_button('sync'),
        
        # For backward compatibility
        'download_btn': action_container.get_button('download'),
        'sync_btn': action_container.get_button('sync'),
        
        # Config buttons
        'save_button': save_reset_components.get('save_button'),
        'reset_button': save_reset_components.get('reset_button'),
        
        # Input components
        'input_options': input_components,
        
        # Log and progress
        'log_output': log_components.get('log_output'),
        'log_accordion': log_components.get('log_accordion'),
        
        # Metadata
        'ui_initialized': True,
        'module_name': 'pretrained'
    })
    
    # Log any missing components for debugging
    from smartcash.ui.utils.logging_utils import log_missing_components
    log_missing_components(ui_components)
    
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

