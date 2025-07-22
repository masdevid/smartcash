"""
File: smartcash/ui/model/backbone/components/backbone_ui.py
Description: Backbone models UI following SmartCash standardized template.

This module provides the user interface for configuring backbone models
with config summary panel moved to summary_container as requested.

Container Order:
1. Header Container (Title, Status)
2. Form Container (Backbone Configuration)
3. Action Container (Validate/Build/Load/Summary Buttons)
4. Summary Container (Configuration Summary Panel)
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
from ..constants import UI_CONFIG, BUTTON_CONFIG, DEFAULT_CONFIG
from ..configs.backbone_defaults import get_available_backbones, get_detection_layers_config

# Module constants (for validator compliance)
UI_CONFIG = UI_CONFIG
BUTTON_CONFIG = BUTTON_CONFIG


@handle_ui_errors(error_component_title="Backbone Models UI Error")
def create_backbone_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create the backbone models UI following SmartCash standards.
    
    This function creates a complete UI for backbone model configuration
    with the following sections:
    - Backbone selection and configuration options
    - Model parameters and optimization settings
    - Configuration summary panel (moved to summary_container)
    - Status and progress tracking
    - Detailed operation logging
    
    Args:
        config: Optional configuration dictionary for the UI
        **kwargs: Additional keyword arguments passed to UI components
        
    Returns:
        Dictionary containing all UI components and their references
        
    Example:
        >>> ui = create_backbone_ui()
        >>> display(ui['main_container'])  # Display the UI
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
        title=UI_CONFIG['title'],
        subtitle=UI_CONFIG['subtitle'],
        icon='🧠',  # Brain icon for backbone models
        show_environment=True,
        environment='local',
        config_path='backbone_config.yaml'
    )
    ui_components['header_container'] = header_container.container
    ui_components['containers']['header'] = header_container
    
    # === 2. Create Form Container ===
    form_widgets = _create_backbone_form_widgets(current_config)
    form_container = create_form_container(
        title=f"⚙️ {UI_CONFIG['title']} Configuration",
        layout_type=LayoutType.COLUMN,
        container_margin="0",
        container_padding="16px",
        gap="12px",
        layout_kwargs={
            'width': '100%',
            'max_width': '100%',
            'margin': '0',
            'padding': '0',
            'justify_content': 'flex-start',
            'align_items': 'flex-start'
        }
    )
    
    # Add form widgets to the container
    for widget in form_widgets['widgets']:
        form_container['add_item'](widget)
    
    ui_components['form_container'] = form_container['container']
    ui_components['containers']['form'] = form_container
    ui_components['widgets'].update(form_widgets['widget_refs'])
    
    # === 3. Create Action Container ===
    action_buttons = []
    for button_id, button_config in BUTTON_CONFIG.items():
        action_buttons.append({
            'id': button_id,
            'text': button_config['text'],
            'style': button_config['style'],
            'tooltip': button_config['tooltip'],
            'disabled': False
        })
    
    action_container = create_action_container(
        buttons=action_buttons,
        show_save_reset=True
    )
    ui_components['action_container'] = action_container
    ui_components['containers']['action'] = action_container
    
    # Extract and store button references with consistent naming (no _button suffix)
    action_buttons = action_container.get('buttons', {})
    
    # Map button IDs to their references
    button_mapping = {
        'validate': 'validate',
        'build': 'build',
        'rescan_models': 'rescan_models',
        'save': 'save',
        'reset': 'reset'
    }
    
    # Store buttons in both the root and widgets dict for backward compatibility
    for src_id, target_id in button_mapping.items():
        button_ref = action_buttons.get(src_id)
        if button_ref is not None:
            ui_components[target_id] = button_ref
            ui_components['widgets'][target_id] = button_ref
    
    # === 4. Create Summary Container (Model Summary Panel) ===
    summary_content = _generate_model_summary_content(current_config)
    summary_container = create_summary_container(
        title="Model Summary",
        theme="info"
    )
    # Set the content after creation
    if hasattr(summary_container, 'update_content'):
        summary_container.update_content(summary_content)
    ui_components['summary_container'] = summary_container
    ui_components['containers']['summary'] = summary_container
    
    # === 5. Create Operation Container ===
    operation_container = create_operation_container(
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        progress_levels='dual',
        log_module_name=UI_CONFIG['module_name'],
        # log_namespace_filter='backbone',  # Temporarily disabled
        log_height="150px",
        collapsible=True,
        collapsed=False
    )
    ui_components['operation_container'] = operation_container  # Store full container object
    ui_components['containers']['operation'] = operation_container
    
    # === 6. Create Footer Container ===
    footer_tips = [
        "💡 EfficientNet-B4 is recommended for higher accuracy",
        "⚡ CSPDarknet provides faster inference with lower memory usage", 
        "🔧 Feature optimization improves model performance for currency detection",
        "📥 Pretrained weights are automatically loaded during build process"
    ]
    
    footer_container = create_footer_container(
        tips=footer_tips,
        module_info={
            'name': UI_CONFIG['title'],
            'version': UI_CONFIG['version'],
            'description': UI_CONFIG['description']
        }
    )
    ui_components['footer_container'] = footer_container.container
    ui_components['containers']['footer'] = footer_container
    
    # === 7. Create Main Container ===
    main_container = create_main_container(
        header_container=ui_components['header_container'],
        form_container=ui_components['form_container'],
        action_container=ui_components['action_container']['container'],
        summary_container=ui_components['summary_container'].container,
        operation_container=ui_components['operation_container']['container'],
        footer_container=ui_components['footer_container'],
    )
    
    # Store main UI references
    ui_components['ui'] = main_container.container
    ui_components['main_container'] = main_container.container
    
    ui_components['main_layout'] = main_container.container
    return ui_components


def _create_backbone_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create form widgets for backbone configuration with two-column layout based on MODEL_ARC_README.md."""
    backbone_config = config.get('backbone', {})
    
    # Get current values
    current_backbone = backbone_config.get('model_type', 'efficientnet_b4')
    current_pretrained = backbone_config.get('pretrained', True)
    current_feature_opt = backbone_config.get('feature_optimization', True)
    current_layer_mode = backbone_config.get('layer_mode', 'multi')
    current_save_path = backbone_config.get('save_path', 'data/models')
    # Remove fixed values (input_size and num_classes are now fixed as per MODEL_ARC_README.md)
    
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
    
    # ===== Left Column: Architecture =====
    # Backbone Selection Dropdown
    backbone_dropdown = widgets.Dropdown(
        options=[
            ('EfficientNet-B4 (Recommended)', 'efficientnet_b4'),
            ('CSPDarknet (YOLOv5 Default)', 'cspdarknet')
        ],
        value=current_backbone,
        description='Backbone:',
        style={'description_width': '140px'},
        layout=input_layout
    )
    
    # Layer Mode Selection
    layer_mode_dropdown = widgets.Dropdown(
        options=[
            ('Multi-Layer Detection (Recommended)', 'multi'),
            ('Single Layer (Legacy)', 'single')
        ],
        value=current_layer_mode,
        description='Detection Mode:',
        style={'description_width': '140px'},
        layout=input_layout
    )
    
    # Model Save Path
    save_path_text = widgets.Text(
        value=current_save_path,
        description='Save Path:',
        style={'description_width': '140px'},
        layout=input_layout,
        placeholder='data/models'
    )
    
    # Architecture options (no training options as requested)
    pretrained_checkbox = widgets.Checkbox(
        value=current_pretrained,
        description='Use Pretrained Weights',
        layout=checkbox_layout,
        style={'description_width': 'initial'}
    )
    
    feature_opt_checkbox = widgets.Checkbox(
        value=current_feature_opt,
        description='Enable Feature Optimization',
        layout=checkbox_layout,
        style={'description_width': 'initial'}
    )
    
    # Left column container
    left_column = widgets.VBox([
        widgets.HTML("<h4 style='margin: 10px 0 5px 0;'>🧬 Architecture</h4>"),
        backbone_dropdown,
        layer_mode_dropdown,
        widgets.HTML("<div style='margin: 15px 0 5px 0; border-top: 1px solid #eee;'></div>"),
        widgets.HTML("<h4 style='margin: 5px 0;'>⚙️ Model Configuration</h4>"),
        save_path_text,
        pretrained_checkbox,
        feature_opt_checkbox
    ], layout=widgets.Layout(width='48%', margin='0 1% 0 0'))
    
    # ===== Right Column: Multi-Layer Detection Info =====
    # Fixed parameters info (no longer editable as per development logs)
    fixed_params_info = widgets.HTML("""
        <div style='margin-top: 10px; padding: 12px; background: #e8f4fd; border-radius: 4px; font-size: 0.9em; border-left: 4px solid #2196F3;'>
            <h4 style='margin: 0 0 8px 0; font-size: 1em; color: #1976D2;'>📏 Fixed Parameters</h4>
            <div style='line-height: 1.5;'>
                <div><strong>• Input Size:</strong> 640×640</div>
                <div><strong>• Classes Layer 1:</strong> 7 (IDR denominations)</div>
                <div><strong>• Classes Layer 2:</strong> 7 (Denomination features)</div>
                <div><strong>• Classes Layer 3:</strong> 3 (Common features)</div>
            </div>
        </div>
    """)
    
    # Multi-layer detection info
    detection_info = widgets.HTML("""
        <div style='margin-top: 10px; padding: 12px; background: #f8f9fa; border-radius: 4px; font-size: 0.9em;'>
            <h4 style='margin: 0 0 8px 0; font-size: 1em;'>🎯 Multi-Layer Detection System</h4>
            <div style='line-height: 1.5;'>
                <div><strong>• Layer 1:</strong> Full banknote bounding boxes</div>
                <div><strong>• Layer 2:</strong> Denomination-specific markers</div>
                <div><strong>• Layer 3:</strong> Common security features</div>
                <div style='margin-top: 8px; color: #666;'><em>Total: 17 classes across 3 detection layers</em></div>
            </div>
        </div>
    """)
    
    # Model status indicator with rescan button
    model_status_info = widgets.HTML("""
        <div style='margin-top: 10px; padding: 12px; background: #fff3cd; border-radius: 4px; font-size: 0.9em; border-left: 4px solid #ffc107;'>
            <h4 style='margin: 0 0 8px 0; font-size: 1em; color: #856404;'>🔍 Model Status</h4>
            <div style='line-height: 1.5;'>
                <div><strong>• Status:</strong> <span id='model-status'>Not Built</span></div>
                <div><strong>• Last Built:</strong> <span id='last-built'>Never</span></div>
                <div><strong>• Available Models:</strong> <span id='available-models'>Checking...</span></div>
            </div>
        </div>
    """)
    
    # Right column container
    right_column = widgets.VBox([
        widgets.HTML("<h4 style='margin: 10px 0 5px 0;'>📊 Model Information</h4>"),
        fixed_params_info,
        detection_info,
        model_status_info
    ], layout=widgets.Layout(width='48%', margin='0 0 0 1%'))
    
    # Create main form container with two columns
    form_ui = widgets.HBox(
        [left_column, right_column],
        layout=widgets.Layout(
            width='100%',
            justify_content='space-between',
            margin='0',
            padding='0'
        )
    )
    
    # Wrap in a styled container
    form_container = widgets.VBox(
        [form_ui],
        layout=widgets.Layout(
            width='100%',
            padding='15px',
            border='1px solid #e0e0e0',
            border_radius='4px',
            margin='5px 0 15px 0'
        )
    )
    
    return {
        'widgets': [form_container],  # Single widget containing the two-column layout
        'widget_refs': {
            'backbone_dropdown': backbone_dropdown,
            'layer_mode_dropdown': layer_mode_dropdown,
            'save_path_text': save_path_text,
            'pretrained_checkbox': pretrained_checkbox,
            'feature_opt_checkbox': feature_opt_checkbox,
            'fixed_params_info': fixed_params_info,
            'detection_info': detection_info,
            'model_status_info': model_status_info
        }
    }


def _generate_model_summary_content(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate model summary content for the summary container based on MODEL_ARC_README.md."""
    backbone_config = config.get('backbone', {})
    available_backbones = get_available_backbones()
    
    # Get backbone information
    backbone_type = backbone_config.get('model_type', 'efficientnet_b4')
    backbone_info = available_backbones.get(backbone_type, {})
    layer_mode = backbone_config.get('layer_mode', 'multi')
    
    # Generate summary sections for model status
    summary_sections = {
        'Model Status': {
            'Status': '⚠️ Not Built',
            'Message': 'Build model to see detailed summary',
            'Backbone': backbone_info.get('display_name', backbone_type),
            'Detection Mode': 'Multi-Layer' if layer_mode == 'multi' else 'Single Layer',
            'Pretrained': '✅ Auto-loaded from timm/drive' if backbone_config.get('pretrained') else '❌ Disabled'
        },
        'Architecture Configuration': {
            'Input Size': '640×640 (Fixed)',
            'Detection Layers': '3 (Layer 1 + Layer 2 + Layer 3)' if layer_mode == 'multi' else '1 (Layer 1 only)',
            'Total Classes': '17 (7+7+3)' if layer_mode == 'multi' else '7',
            'Feature Optimization': '✅ Enabled' if backbone_config.get('feature_optimization') else '❌ Disabled',
            'Save Path': backbone_config.get('save_path', 'data/models')
        }
    }
    
    # Add multi-layer specific information
    if layer_mode == 'multi':
        summary_sections['Detection Layers'] = {
            'Layer 1 (Full Notes)': '7 classes - IDR denominations',
            'Layer 2 (Features)': '7 classes - Denomination-specific markers',
            'Layer 3 (Common)': '3 classes - Security features shared across notes',
            'Training Strategy': 'Two-phase with uncertainty-based multi-task loss'
        }
    
    return {
        'title': 'Model Summary',
        'sections': summary_sections,
        'style': 'info'
    }


def update_model_summary(summary_container: Any, model_info: Dict[str, Any]) -> None:
    """
    Update the model summary in the summary container.
    
    Args:
        summary_container: The summary container widget
        model_info: Model information dictionary from API
    """
    try:
        if summary_container and hasattr(summary_container, 'update_content'):
            new_content = _generate_model_summary_from_info(model_info)
            summary_container.update_content(new_content)
    except Exception as e:
        # Silently handle update errors to avoid UI disruption
        pass

def _generate_model_summary_from_info(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """Generate model summary content from model API info."""
    status = model_info.get('status', 'not_built')
    
    if status == 'built':
        # Model is built - show detailed info
        summary_sections = {
            'Model Status': {
                'Status': '✅ Built Successfully',
                'Model Name': model_info.get('model_name', 'N/A'),
                'Backbone': model_info.get('backbone', 'N/A'),
                'Device': model_info.get('device', 'N/A')
            },
            'Model Parameters': {
                'Total Parameters': f"{model_info.get('total_parameters', 0):,}",
                'Trainable Parameters': f"{model_info.get('trainable_parameters', 0):,}",
                'Memory Usage': model_info.get('memory_usage', 'N/A'),
                'Input Size': f"{model_info.get('img_size', 640)}px"
            },
            'Configuration': {
                'Detection Layers': ', '.join(model_info.get('detection_layers', ['banknote'])),
                'Layer Mode': model_info.get('layer_mode', 'single').title(),
                'Number of Classes': model_info.get('num_classes', 7),
                'Feature Optimization': '✅ Enabled' if model_info.get('feature_optimization') else '❌ Disabled'
            }
        }
    elif status == 'error':
        # Error state
        summary_sections = {
            'Model Status': {
                'Status': '❌ Error',
                'Message': model_info.get('message', 'Unknown error occurred'),
                'Backbone': model_info.get('backbone', 'N/A')
            }
        }
    else:
        # Not built yet
        summary_sections = {
            'Model Status': {
                'Status': '⚠️ Not Built',
                'Message': model_info.get('message', 'Build model to see detailed summary'),
                'Backbone': model_info.get('backbone', 'N/A'),
                'Pretrained': '✅ Auto-loaded from drive' if model_info.get('pretrained') else '❌ Disabled'
            }
        }
    
    return {
        'title': 'Model Summary',
        'sections': summary_sections,
        'style': 'success' if status == 'built' else 'warning' if status == 'not_built' else 'error'
    }


# ==================== HELPER FUNCTIONS ====================

def get_backbone_form_values(form_container: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract form values from backbone form container.
    
    Args:
        form_container: Form container dictionary
        
    Returns:
        Dictionary of form values
    """
    try:
        if form_container and 'get_form_values' in form_container:
            return form_container['get_form_values']()
        return {}
    except Exception:
        return {}


def update_backbone_form(form_container: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Update backbone form with configuration values.
    
    Args:
        form_container: Form container dictionary
        config: Configuration dictionary
    """
    try:
        if form_container and 'update_from_config' in form_container:
            form_container['update_from_config'](config)
    except Exception:
        # Silently handle update errors
        pass