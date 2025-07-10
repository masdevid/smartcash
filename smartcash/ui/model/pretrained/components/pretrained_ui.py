"""
File: smartcash/ui/model/pretrained/components/pretrained_ui.py
UI components for pretrained models module following container standards.
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets

from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.header_container import create_header_container  
from smartcash.ui.components.form_container import create_form_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container

from ..constants import UI_CONFIG, DEFAULT_CONFIG, PretrainedModelType, DEFAULT_MODEL_URLS


def create_pretrained_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create pretrained models UI using standard container components.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing UI components
    """
    config = config or DEFAULT_CONFIG.copy()
    ui_components = {}
    
    # 1. Header Container
    header_container = create_header_container(
        title=UI_CONFIG["title"],
        subtitle=UI_CONFIG["subtitle"],
        icon=UI_CONFIG["icon"]
    )
    ui_components['header_container'] = header_container.container
    
    # 2. Form Container with input options
    form_container = create_form_container()
    input_components = create_input_form(config)
    form_container['add_item'](input_components['ui'])  # Use add_item method to add content
    ui_components['form_container'] = form_container['container']
    ui_components['input_options'] = input_components
    
    # 3. Action Container with default primary button for download
    action_container = create_action_container(
        buttons=[],  # No additional buttons, use default primary
        title="🤖 Model Operations",
        alignment="center",
        show_save_reset=True  # Keep default save/reset buttons
    )
    ui_components['action_container'] = action_container['container']
    
    # Configure the primary button for download
    primary_button = action_container['primary_button']
    if primary_button:
        primary_button.description = "📥 Download Models"
        primary_button.tooltip = "Download selected pretrained models"
    ui_components['download_button'] = primary_button
    
    # 4. Summary Container
    summary_container = create_summary_container(
        title="📊 Models Status",
        theme="info"
    )
    # Set content after creation
    status_content = create_models_status_widget(config)
    summary_container.set_content(status_content.value if hasattr(status_content, 'value') else str(status_content))
    ui_components['summary_container'] = summary_container.container
    ui_components['models_status'] = summary_container._ui_components.get('content')
    
    # 5. Operation Container with progress and logging
    operation_container = create_operation_container(
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        log_module_name="pretrained"
    )
    ui_components['operation_container'] = operation_container['container']
    ui_components['progress_tracker'] = operation_container['progress_tracker']
    ui_components['log_output'] = operation_container['log_accordion']  # Use log_accordion as log_output
    ui_components['log_accordion'] = operation_container['log_accordion']
    
    # 6. Footer Container with info box
    info_box = create_info_box()
    footer_container = widgets.VBox([info_box], layout=widgets.Layout(
        width='100%',
        margin='15px 0 0 0'
    ))
    ui_components['footer_container'] = footer_container
    
    # 7. Main Container Assembly - Simple VBox layout
    main_container = widgets.VBox([
        ui_components['header_container'],
        ui_components['form_container'],
        ui_components['action_container'],
        ui_components['summary_container'],
        ui_components['operation_container'],
        ui_components['footer_container']
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    ui_components['ui'] = main_container
    ui_components['main_container'] = main_container
    
    # Metadata
    ui_components.update({
        'ui_initialized': True,
        'module_name': 'pretrained',
        'parent_module': 'model'
    })
    
    return ui_components


def create_input_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create input form for pretrained models configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing input components
    """
    # Models directory input
    models_dir_input = widgets.Text(
        value=config.get('models_dir', DEFAULT_CONFIG['models_dir']),
        description='Models Directory:',
        placeholder='/data/pretrained',
        style={'description_width': '140px'},
        layout={'width': '100%'}
    )
    
    # YOLOv5s URL input  
    yolo_url_input = widgets.Text(
        value=config.get('model_urls', {}).get(PretrainedModelType.YOLOV5S.value, ''),
        description='YOLOv5s URL:',
        placeholder='Leave empty for default GitHub URL',
        style={'description_width': '140px'},
        layout={'width': '100%'}
    )
    
    # EfficientNet-B4 URL input
    efficientnet_url_input = widgets.Text(
        value=config.get('model_urls', {}).get(PretrainedModelType.EFFICIENTNET_B4.value, ''),
        description='EfficientNet URL:',
        placeholder='Leave empty to use timm library',
        style={'description_width': '140px'},
        layout={'width': '100%'}
    )
    
    # Create form layout
    form_ui = widgets.VBox([
        widgets.HTML("<h4>📁 Download Configuration</h4>"),
        models_dir_input,
        widgets.HTML(
            "<div style='margin: 15px 0 8px 0; font-weight: bold; color: #666;'>"
            "🔗 Custom Download URLs (Optional):</div>"
        ),
        yolo_url_input,
        efficientnet_url_input,
        widgets.HTML(
            "<div style='margin-top: 8px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 0.9em; color: #666;'>"
            "<strong>📝 Notes:</strong><br>"
            "• YOLOv5s: Direct download from GitHub release<br>"
            "• EfficientNet-B4: Downloaded via timm library (recommended)<br>"
            "• Leave URLs empty to use default/recommended sources"
            "</div>"
        )
    ])
    
    return {
        'ui': form_ui,
        'model_dir_input': models_dir_input,
        'yolo_url_input': yolo_url_input,
        'efficientnet_url_input': efficientnet_url_input
    }


def create_models_status_widget(config: Dict[str, Any]) -> widgets.Widget:
    """
    Create models status widget for summary container.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Widget showing models status
    """
    models_dir = config.get('models_dir', DEFAULT_CONFIG['models_dir'])
    
    status_html = f"""
    <div style="padding: 10px;">
        <div style="margin-bottom: 10px;">
            <strong>📁 Directory:</strong> <code>{models_dir}</code>
        </div>
        <div style="margin-bottom: 10px;">
            <strong>🎯 Target Models:</strong>
        </div>
        <div style="margin-left: 15px;">
            <div style="margin-bottom: 5px;">
                • <strong>YOLOv5s:</strong> Fast object detection (~14MB)
            </div>
            <div style="margin-bottom: 5px;">
                • <strong>EfficientNet-B4:</strong> Efficient CNN backbone (~75MB)
            </div>
        </div>
        <div style="margin-top: 10px; padding: 8px; background: #e3f2fd; border-radius: 4px; font-size: 0.9em;">
            <strong>ℹ️ Status:</strong> Use the download button to check and download missing models
        </div>
    </div>
    """
    
    return widgets.HTML(status_html)


def create_info_box() -> widgets.Widget:
    """
    Create info box for footer container.
    
    Returns:
        Widget containing helpful information
    """
    info_html = """
    <div class="alert alert-info" style="font-size: 0.9em; padding: 12px; margin: 0;">
        <strong>🤖 Pretrained Models Information:</strong>
        <ul style="margin: 8px 0 0 15px; padding: 0;">
            <li><strong>YOLOv5s:</strong> Optimized for real-time object detection with good accuracy</li>
            <li><strong>EfficientNet-B4:</strong> Provides efficient feature extraction for classification tasks</li>
            <li><strong>Storage:</strong> Models are saved to the specified directory for reuse across sessions</li>
            <li><strong>Performance:</strong> Downloaded models reduce initialization time in training workflows</li>
        </ul>
        <div style="margin-top: 8px; padding: 6px; background: rgba(0,0,0,0.05); border-radius: 3px;">
            <strong>💡 Tip:</strong> Download models once and reuse them across multiple training sessions
        </div>
    </div>
    """
    
    return widgets.HTML(info_html)