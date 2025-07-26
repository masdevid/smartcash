"""
Pretrained UI Components Creation

This module contains the UI component creation logic for the pretrained models interface.
"""

from typing import Dict, Any
import ipywidgets as widgets
from smartcash.ui.logger import get_module_logger

def create_pretrained_ui_components(module_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create all UI components for pretrained models management.
    
    Args:
        module_config: Module configuration containing model settings
        
    Returns:
        Dictionary containing all UI components
    """
    try:
        from smartcash.ui.components.header_container import create_header_container
        from smartcash.ui.components.form_container import create_form_container, LayoutType
        from smartcash.ui.components.action_container import create_action_container
        from smartcash.ui.components.operation_container import create_operation_container
        from smartcash.ui.components.footer_container import create_footer_container, PanelConfig, PanelType
        from smartcash.ui.components.main_container import create_main_container
        
        # 1. Header Container
        header_container = create_header_container(
            title="Pretrained Models",
            subtitle="Download and manage YOLOv5s and EfficientNet-B4 models",
            icon="🤖",
            show_environment=True,
            environment='local',
            config_path='pretrained_config.yaml'
        )
        
        # 2. Form Container with Model Configuration
        form_container = create_form_container(
            layout_type=LayoutType.COLUMN,
            container_margin="0",
            container_padding="10px",
            gap="10px"
        )
        
        # Create form widgets
        form_widgets = _create_model_form_widgets(module_config)
        
        # Add items to form
        config_label = widgets.HTML("<h3 style='margin: 10px 0;'>📁 Model Configuration</h3>")
        options_label = widgets.HTML("<h4>⚙️ Download Options</h4>")
        
        form_items = [config_label, form_widgets['config_section'], options_label, form_widgets['options_section']]
        
        for item in form_items:
            form_container['add_item'](item, height="auto")
        
        # 3. Action Container with one-click setup button
        action_container = create_action_container(
            buttons=[
                {
                    'id': 'oneclick_setup',
                    'text': '🚀 One-Click Setup',
                    'style': 'success',
                    'tooltip': 'Complete setup: check existing → download missing → validate → cleanup invalid files'
                },
                {
                    'id': 'refresh',
                    'text': '🔄 Refresh Status',
                    'style': 'info',
                    'tooltip': 'Refresh model status and directory contents'
                }
            ],
            title="🤖 Model Operations",
            show_save_reset=True
        )
        
        # 4. Operation Container with dual progress tracking
        operation_container = create_operation_container(
            show_progress=True,
            show_dialog=False,  # Disable dialog to reduce clutter
            show_logs=True,
            log_module_name="Pretrained Models",
            progress_levels='dual'  # Enable dual progress tracking
        )
        
        # 5. Footer Container
        footer_container = create_footer_container(
            panels=[
                PanelConfig(
                    panel_type=PanelType.INFO_ACCORDION,
                    title="💡 Pretrained Models Guide",
                    content="""
                    <div style="padding: 10px;">
                        <h4>🚀 One-Click Setup</h4>
                        <p>Automatically handles the complete model setup workflow:</p>
                        <ol>
                            <li><strong>Check Existing:</strong> Scan for already downloaded models</li>
                            <li><strong>Download Missing:</strong> Download YOLOv5s and EfficientNet-B4 if needed</li>
                            <li><strong>Validate All:</strong> Check integrity and file sizes</li>
                            <li><strong>Cleanup Invalid:</strong> Remove corrupted or invalid files</li>
                        </ol>
                        <p><strong>Refresh Status:</strong> Update model status and directory contents manually</p>
                    </div>
                    """,
                    style="info",
                    open_by_default=False
                )
            ]
        )
        
        # 6. Main Container
        components = [
            {'type': 'header', 'component': header_container.container, 'order': 0},
            {'type': 'form', 'component': form_container['container'], 'order': 1},
            {'type': 'action', 'component': action_container['container'], 'order': 2},
            {'type': 'operation', 'component': operation_container['container'], 'order': 3},
            {'type': 'footer', 'component': footer_container.container, 'order': 4}
        ]
        
        main_container = create_main_container(components=components)
        
        # Return UI components dictionary
        return {
            # Container components
            'main_container': main_container.container,
            'header_container': header_container,
            'form_container': form_container,
            'action_container': action_container,
            'footer_container': footer_container,
            'operation_container': operation_container,
            
            # Form elements
            'models_dir_input': form_widgets['models_dir_input'],
            'yolo_url_input': form_widgets['yolo_url_input'],
            'efficientnet_url_input': form_widgets['efficientnet_url_input'],
            'auto_download_checkbox': form_widgets['auto_download_checkbox'],
            'validate_checkbox': form_widgets['validate_checkbox'],
            
            # Button references from action container
            'oneclick_setup': action_container.get('buttons', {}).get('oneclick_setup'),
            'refresh': action_container.get('buttons', {}).get('refresh'),
            'save': action_container.get('buttons', {}).get('save'),
            'reset': action_container.get('buttons', {}).get('reset'),
            
            # Operation components (widgets for direct access)
            'operation_container_widget': operation_container.get('container'),
            'progress_tracker': operation_container.get('progress_tracker'),
            'log_accordion': operation_container.get('log_accordion')
        }
        
    except Exception as e:
        get_module_logger("smartcash.ui.model.pretrained.components.pretrained_ui").error(f"❌ Failed to create pretrained UI components: {e}")
        raise


def _create_model_form_widgets(module_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create model form widgets for configuration.
    
    Args:
        module_config: Module configuration containing model settings
        
    Returns:
        Dictionary containing form widgets
    """
    from smartcash.ui.model.pretrained.constants import DEFAULT_CONFIG, PretrainedModelType
    
    # Common layout settings
    input_layout = widgets.Layout(
        width='auto',
        margin='5px 0',
        padding='5px 0'
    )
    
    checkbox_layout = widgets.Layout(
        width='auto',
        margin='8px 0',
        padding='5px 0'
    )
    
    # Models directory input
    models_dir_input = widgets.Text(
        value=module_config.get('models_dir', DEFAULT_CONFIG['models_dir']),
        description='Models Directory:',
        placeholder='/data/pretrained',
        style={'description_width': '140px'},
        layout=input_layout
    )
    
    # YOLOv5s URL input  
    yolo_url_input = widgets.Text(
        value=module_config.get('model_urls', {}).get(PretrainedModelType.YOLOV5S.value, ''),
        description='YOLOv5s URL:',
        placeholder='Leave empty for default GitHub URL',
        style={'description_width': '140px'},
        layout=input_layout
    )
    
    # EfficientNet-B4 URL input
    efficientnet_url_input = widgets.Text(
        value=module_config.get('model_urls', {}).get(PretrainedModelType.EFFICIENTNET_B4.value, ''),
        description='EfficientNet URL:',
        placeholder='Leave empty to use timm library',
        style={'description_width': '140px'},
        layout=input_layout
    )
    
    # Auto download checkbox
    auto_download_checkbox = widgets.Checkbox(
        value=module_config.get('auto_download', False),
        description='Auto Download Missing Models',
        layout=checkbox_layout,
        style={'description_width': 'initial'}
    )
    
    # Validate downloads checkbox
    validate_checkbox = widgets.Checkbox(
        value=module_config.get('validate_downloads', True),
        description='Validate Downloaded Models',
        layout=checkbox_layout,
        style={'description_width': 'initial'}
    )
    
    # Create form sections
    config_section = widgets.VBox([
        models_dir_input,
        widgets.HTML("<h4 style='margin: 5px 0;'>🔗 Custom Download URLs</h4>"),
        yolo_url_input,
        efficientnet_url_input,
        widgets.HTML("<div style='font-size: 0.85em; color: #666; margin: 5px 0;'>Leave URLs empty to use default sources</div>")
    ], layout=widgets.Layout(width='100%', margin='0 0 15px 0'))
    
    options_section = widgets.VBox([
        auto_download_checkbox,
        validate_checkbox
    ], layout=widgets.Layout(width='100%', margin='0 0 10px 0'))
    
    return {
        'config_section': config_section,
        'options_section': options_section,
        'models_dir_input': models_dir_input,
        'yolo_url_input': yolo_url_input,
        'efficientnet_url_input': efficientnet_url_input,
        'auto_download_checkbox': auto_download_checkbox,
        'validate_checkbox': validate_checkbox
    }


# Legacy compatibility function (if needed)
def create_pretrained_ui(config=None, **kwargs):
    """Legacy compatibility function."""
    # Merge kwargs into config if provided
    final_config = config or {}
    if kwargs:
        final_config.update(kwargs)
    return create_pretrained_ui_components(final_config)