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
from smartcash.ui.core.decorators import handle_ui_errors

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
    # Create form widgets with the new layout
    form_widgets = _create_module_form_widgets(current_config)
    
    # Create form container with consistent styling
    form_container = create_form_container(
        layout_type=LayoutType.COLUMN,
        container_margin="0 0 20px 0",
        container_padding="0",
        gap="15px",
        layout_kwargs={
            'width': '100%',
            'max_width': '100%',
            'margin': '0',
            'padding': '0',
            'justify_content': 'flex-start',
            'align_items': 'flex-start'
        }
    )
    
    # Add form rows to the container
    for row in form_widgets['form_rows']:
        for widget in row:
            form_container['add_item'](widget, width='100%')
    
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
        show_progress=True,
        show_logs=True,
        log_module_name=UI_CONFIG['title'],
        log_height="200px",
        log_entry_style='compact'  # Ensure consistent hover behavior with downloader
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
    Matches the preprocess UI style with consistent layout and removes redundant notes.
    
    Args:
        config: Configuration dictionary for the form widgets
        
    Returns:
        Dictionary containing the form UI and widget references
    """
    # Common layout settings
    input_layout = widgets.Layout(
        width='90%',
        margin='5px 0',
        padding='5px 0'
    )
    
    checkbox_layout = widgets.Layout(
        width='100%',
        margin='8px 0',
        padding='5px 0'
    )
    
    # Models directory input
    models_dir_input = widgets.Text(
        value=config.get('models_dir', DEFAULT_CONFIG['models_dir']),
        description='Models Directory:',
        placeholder='/data/pretrained',
        style={'description_width': '140px'},
        layout=input_layout
    )
    
    # YOLOv5s URL input  
    yolo_url_input = widgets.Text(
        value=config.get('model_urls', {}).get(PretrainedModelType.YOLOV5S.value, ''),
        description='YOLOv5s URL:',
        placeholder='Leave empty for default GitHub URL',
        style={'description_width': '140px'},
        layout=input_layout
    )
    
    # EfficientNet-B4 URL input
    efficientnet_url_input = widgets.Text(
        value=config.get('model_urls', {}).get(PretrainedModelType.EFFICIENTNET_B4.value, ''),
        description='EfficientNet URL:',
        placeholder='Leave empty to use timm library',
        style={'description_width': '140px'},
        layout=input_layout
    )
    
    # Auto download checkbox
    auto_download_checkbox = widgets.Checkbox(
        value=config.get('auto_download', False),
        description='Auto Download Missing Models',
        layout=checkbox_layout,
        style={'description_width': 'initial'}
    )
    
    # Validate downloads checkbox
    validate_checkbox = widgets.Checkbox(
        value=config.get('validate_downloads', True),
        description='Validate Downloaded Models',
        layout=checkbox_layout,
        style={'description_width': 'initial'}
    )
    
    # Create form sections
    config_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 10px 0 5px 0;'>📁 Download Configuration</h4>"),
        models_dir_input,
        widgets.HTML("<div style='margin: 15px 0 5px 0; border-top: 1px solid #eee;'></div>"),
        widgets.HTML("<h4 style='margin: 5px 0;'>⚙️ Download Options</h4>"),
        auto_download_checkbox,
        validate_checkbox
    ], layout=widgets.Layout(width='100%', margin='0 0 15px 0'))
    
    urls_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 5px 0;'>🔗 Custom Download URLs</h4>"),
        yolo_url_input,
        efficientnet_url_input,
        widgets.HTML("<div style='font-size: 0.85em; color: #666; margin: 5px 0;'>Leave URLs empty to use default sources</div>")
    ], layout=widgets.Layout(width='100%', margin='0 0 10px 0'))
    
    # Combine sections into a single form
    form_ui = widgets.VBox([
        config_section,
        urls_section
    ], layout=widgets.Layout(
        width='100%',
        padding='10px 15px',
        border='1px solid #e0e0e0',
        border_radius='4px',
        margin='5px 0'
    ))
    
    return {
        'form_rows': [[form_ui]],  # Single row with the combined form
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