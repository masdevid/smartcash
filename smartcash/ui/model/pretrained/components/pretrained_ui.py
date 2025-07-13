"""
File: smartcash/ui/model/pretrained/components/pretrained_ui.py
Description: Pretrained models UI following SmartCash standardized template.

This module provides the user interface for downloading and managing
pretrained models (YOLOv5s and EfficientNet-B4) for the SmartCash system.

Container Order:
1. Header Container (Title, Status)
2. Form Container (Download Configuration)
3. Action Container (Download/Validate/Refresh/Cleanup Buttons)
4. Summary Container (Models Status)
5. Operation Container (Progress + Logs)
6. Footer Container (Tips and Info)
"""

from typing import Optional, Dict, Any
import ipywidgets as widgets

# Core container imports - standardized across all modules
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container, LayoutType
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.core.errors.handlers import handle_ui_errors

# Module imports
from ..constants import UI_CONFIG, BUTTON_CONFIG, DEFAULT_CONFIG, PretrainedModelType

# Module constants (for validator compliance)
UI_CONFIG = UI_CONFIG
BUTTON_CONFIG = BUTTON_CONFIG


@handle_ui_errors(error_component_title="Pretrained Models UI Error")
def create_pretrained_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create the pretrained models UI following SmartCash standards.
    
    This function creates a complete UI for managing pretrained models
    with the following sections:
    - Download configuration options
    - Model management operations
    - Status and progress tracking
    - Detailed model information
    
    Args:
        config: Optional configuration dictionary for the UI
        **kwargs: Additional keyword arguments passed to UI components
        
    Returns:
        Dictionary containing all UI components and their references with 'ui_components' key
        
    Example:
        >>> ui = create_pretrained_ui()
        >>> display(ui['ui'])  # Display the UI
    """
    # Initialize configuration and components dictionary
    current_config = config or DEFAULT_CONFIG.copy()
    ui_components = {
        'config': current_config,
        'containers': {},
        'widgets': {}
    }
    
    # === 1. Create Header Container ===
    header_container = create_header_container(
        title=f"{UI_CONFIG['icon']} {UI_CONFIG['title']}",
        subtitle=UI_CONFIG['subtitle'],
        status_message="Ready to manage pretrained models",
        status_type="info"
    )
    # Store both the container object and its widget
    ui_components['containers']['header'] = {
        'container': header_container.container,
        'widget': header_container
    }
    
    # === 2. Create Form Container ===
    # Create form widgets
    form_widgets = _create_module_form_widgets(current_config)
    
    # Create form container with the widgets
    form_container = create_form_container(
        layout_type=LayoutType.COLUMN,
        container_margin="0",
        container_padding="16px",
        gap="12px"
    )
    
    # Add form rows to the container
    for row in form_widgets['form_rows']:
        for widget in row:
            form_container['add_item'](widget)
    
    # Store references
    ui_components['containers']['form'] = form_container
    ui_components['widgets'].update(form_widgets['widgets'])
    
    # === 3. Create Action Container ===
    # Create action buttons from BUTTON_CONFIG (matching backbone module format)
    action_buttons = []
    for button_id, btn_config in BUTTON_CONFIG.items():
        action_buttons.append({
            'id': button_id,
            'text': btn_config['text'],
            'style': btn_config['style'],
            'tooltip': btn_config['tooltip'],
            'disabled': False
        })
    
    action_container = create_action_container(
        buttons=action_buttons,
        show_save_reset=True
    )
    
    # Store references (matching backbone module format)
    ui_components['action_container'] = action_container
    ui_components['containers']['action'] = action_container
    
    # === 4. Create Summary Container ===
    summary_content = _create_module_summary_content(current_config)
    summary_container = create_summary_container(
        title="📊 Models Status",
        theme="info",
        icon="🤖"
    )
    summary_container.set_content(summary_content)
    
    ui_components['containers']['summary'] = summary_container
    
    # === 5. Create Operation Container ===
    operation_container = create_operation_container(
        title="📥 Download Progress",
        show_progress=True,
        show_logs=True,
        collapsible=True,
        collapsed=False
    )
    ui_components['containers']['operation'] = operation_container
    
    # === 6. Create Footer Container ===
    footer_container = create_footer_container(
        info_box=_create_module_info_box(),
        show_tips=True,
        show_version=True
    )
    # Store both the container object and its widget
    ui_components['containers']['footer'] = {
        'container': footer_container.container,
        'widget': footer_container
    }
    
    # === 7. Create Main Container ===
    # Combine form, action, summary, and operation into the form container slot
    combined_body = widgets.VBox([
        ui_components['containers']['form']['container'],
        ui_components['action_container']['container'],
        ui_components['containers']['summary'].container,
        ui_components['containers']['operation']['container']
    ])
    
    main_container = create_main_container(
        header_container=ui_components['containers']['header']['container'],
        form_container=combined_body,
        footer_container=ui_components['containers']['footer']['container'],
        margin='0 auto',
        max_width='1200px',
        padding='10px',
        border='1px solid #e0e0e0',
        border_radius='5px',
        box_shadow='0 1px 3px rgba(0,0,0,0.1)'
    )
    
    # Store main UI references
    ui_components['ui'] = main_container.container
    ui_components['main_container'] = main_container
    
    result = {
        'ui_components': ui_components,
        'ui': ui_components['ui']
    }
    
    # Add all components to the root for backward compatibility
    result.update(ui_components['containers'])
    result.update(ui_components['widgets'])
    
    # Add legacy compatibility
    result.update({
        'ui_initialized': True,
        'module_name': 'pretrained',
        'parent_module': 'model'
    })
    
    return result


def _create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create module-specific form widgets for pretrained models configuration.
    
    Args:
        config: Configuration dictionary for the form widgets
        
    Returns:
        Dictionary containing the form UI and widget references
    """
    # Models directory input
    models_dir_input = widgets.Text(
        value=config.get('models_dir', DEFAULT_CONFIG['models_dir']),
        description='Models Directory:',
        placeholder='/data/pretrained',
        style={'description_width': '140px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # YOLOv5s URL input  
    yolo_url_input = widgets.Text(
        value=config.get('model_urls', {}).get(PretrainedModelType.YOLOV5S.value, ''),
        description='YOLOv5s URL:',
        placeholder='Leave empty for default GitHub URL',
        style={'description_width': '140px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # EfficientNet-B4 URL input
    efficientnet_url_input = widgets.Text(
        value=config.get('model_urls', {}).get(PretrainedModelType.EFFICIENTNET_B4.value, ''),
        description='EfficientNet URL:',
        placeholder='Leave empty to use timm library',
        style={'description_width': '140px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Auto download checkbox
    auto_download_checkbox = widgets.Checkbox(
        value=config.get('auto_download', False),
        description='Auto Download Missing Models',
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Validate downloads checkbox
    validate_checkbox = widgets.Checkbox(
        value=config.get('validate_downloads', True),
        description='Validate Downloaded Models',
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Create form rows
    form_rows = [
        [widgets.HTML("<h4>📁 Download Configuration</h4>")],
        [models_dir_input],
        [widgets.HTML("<h4>🔗 Custom Download URLs (Optional)</h4>")],
        [yolo_url_input],
        [efficientnet_url_input],
        [widgets.HTML("<h4>⚙️ Download Options</h4>")],
        [widgets.HBox([auto_download_checkbox, validate_checkbox])],
        [widgets.HTML("""
            <div style='margin-top: 8px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 0.9em; color: #666;'>
                <strong>📝 Notes:</strong><br>
                • YOLOv5s: Direct download from GitHub release (~14MB)<br>
                • EfficientNet-B4: Downloaded via timm library (~75MB)<br>
                • Leave URLs empty to use default/recommended sources
            </div>
        """)]
    ]
    
    return {
        'form_rows': form_rows,
        'widgets': {
            'models_dir_input': models_dir_input,
            'yolo_url_input': yolo_url_input,
            'efficientnet_url_input': efficientnet_url_input,
            'auto_download_checkbox': auto_download_checkbox,
            'validate_checkbox': validate_checkbox
        }
    }


def _create_module_summary_content(config: Dict[str, Any]) -> str:
    """
    Create summary content for the module.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        HTML string containing the summary content
    """
    models_dir = config.get('models_dir', DEFAULT_CONFIG['models_dir'])
    
    return f"""
    <div style="padding: 10px;">
        <h5>🤖 Pretrained Models Overview</h5>
        <div style="margin-bottom: 10px;">
            <strong>📁 Directory:</strong> <code>{models_dir}</code>
        </div>
        <div style="margin-bottom: 10px;">
            <strong>🎯 Target Models:</strong>
        </div>
        <ul style="margin-left: 15px;">
            <li><strong>YOLOv5s:</strong> Fast object detection (~14MB)</li>
            <li><strong>EfficientNet-B4:</strong> Efficient CNN backbone (~75MB)</li>
        </ul>
        <div style="margin-top: 10px; padding: 8px; background: #e3f2fd; border-radius: 4px; font-size: 0.9em;">
            <strong>ℹ️ Status:</strong> Use the buttons above to download and manage models
        </div>
    </div>
    """


def _create_module_info_box() -> widgets.Widget:
    """
    Create the info box content for the footer.
    
    Returns:
        Widget containing the info box content
    """
    return widgets.HTML(
        value="""
        <div style="padding: 12px; background: #e3f2fd; border-radius: 4px; margin: 8px 0;">
            <h4 style="margin-top: 0; color: #0d47a1;">🤖 Pretrained Models Guide</h4>
            <p>This module helps you download and manage pretrained models for SmartCash.</p>
            <ol style="margin: 8px 0 0 16px; padding-left: 8px;">
                <li>Configure download directory and options</li>
                <li>Click 'Download Models' to get YOLOv5s and EfficientNet-B4</li>
                <li>Use 'Validate Models' to check model integrity</li>
                <li>Use 'Refresh Status' to update model information</li>
                <li>Use 'Clean Up' to remove corrupted files</li>
            </ol>
            <div style="margin-top: 8px; padding: 6px; background: rgba(0,0,0,0.05); border-radius: 3px;">
                <strong>💡 Tip:</strong> Download models once and reuse them across multiple training sessions
            </div>
        </div>
        """
    )