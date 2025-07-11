"""
File: smartcash/ui/model/backbone/components/ui_components.py
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

# Standard container imports
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container, LayoutType
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.core.errors.handlers import handle_ui_errors

# Local imports
from .model_form import create_model_form, update_form_values
from .config_summary import create_config_summary, update_config_summary

# Module metadata
MODULE_METADATA = {
    'module_name': 'backbone',
    'parent_module': 'model',
    'ui_initialized': True,
    'config': {}
}

# UI Configuration
UI_CONFIG = {
    'module_name': 'Backbone Configuration',
    'parent_module': 'model',
    'version': '1.0.0',
    'title': '🧬 Model Backbone',
    'subtitle': 'Configure the base architecture for your model',
    'icon': '🧬',
    'description': 'Set up the backbone architecture that will be used for feature extraction.'
}

# Action Buttons Configuration
# These are the action buttons that will be shown in the action container
# along with save/reset buttons

ACTION_BUTTONS = [
    {
        'id': 'initialize',
        'text': '🚀 Initialize',
        'style': 'success',
        'icon': 'rocket',
        'tooltip': 'Initialize the backbone architecture',
        'order': 1
    },
    {
        'id': 'validate',
        'text': '🔍 Validate',
        'style': 'info',
        'icon': 'check',
        'tooltip': 'Validate the configuration',
        'order': 2
    },
    {
        'id': 'load',
        'text': '📥 Load',
        'style': 'info',
        'icon': 'download',
        'tooltip': 'Load a saved configuration',
        'order': 3
    },
    {
        'id': 'build',
        'text': '🏗️ Build',
        'style': 'success',
        'icon': 'wrench',
        'tooltip': 'Build the model architecture',
        'order': 4
    },
    {
        'id': 'summary',
        'text': '📊 Summary', 
        'style': 'warning',
        'icon': 'list',
        'tooltip': 'Show model summary',
        'order': 5
    }
]

@handle_ui_errors(error_component_title="Backbone UI Creation Error")
def create_backbone_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Create and initialize the Backbone Configuration UI components.
    
    This function creates a standardized UI for the backbone configuration module,
    following the SmartCash UI component architecture.
    
    Args:
        config: Optional configuration dictionary to initialize the UI
        **kwargs: Additional keyword arguments passed to component creators
        
    Returns:
        Dict containing all UI components and their references
    """
    if config is None:
        config = {}
    
    # Initialize components dictionary
    components = {
        'ui_initialized': False,
        'module_name': UI_CONFIG['module_name'],
        'parent_module': UI_CONFIG['parent_module'],
        'version': UI_CONFIG['version'],
        'config': config.copy()
    }
    
    # 1. Create Header Container
    header_container = create_header_container(
        title=UI_CONFIG['title'],
        subtitle=UI_CONFIG['subtitle'],
        status_text="Ready",
        icon=UI_CONFIG['icon']
    )
    
    # 2. Create Form Container
    form_container = create_form_container(
        layout_type=LayoutType.COLUMN,
        container_padding="0",
        gap="12px"
    )
    
    # Add form widgets (full width)
    model_form = create_model_form(config)
    form_container['add_item'](model_form, "backbone_form")
    
    # Store form widgets for easy access
    components['model_form'] = model_form
    
    # 3. Create Action Container with action buttons and save/reset
    action_container = create_action_container(
        title="Backbone Operations",
        buttons=ACTION_BUTTONS,
        show_save_reset=True,
        alignment="left"
    )
    
    # 4. Create Operation Container
    operation_container = create_operation_container(
        show_progress=True,
        show_logs=True,
        log_module_name=UI_CONFIG['title'],
        log_height="200px"
    )
    
    # 5. Create Summary Container
    summary_content = create_config_summary(config)
    summary_container = create_summary_container(
        title="📋 Ringkasan Konfigurasi",
        theme="info"
    )
    
    # Initialize the container if needed
    if hasattr(summary_container, 'initialize') and not getattr(summary_container, '_initialized', True):
        summary_container.initialize()
    
    # Set the content
    if hasattr(summary_container, 'set_content'):
        summary_container.set_content(summary_content)
    
    # Store summary container and content for updates
    components['summary_container'] = summary_container
    components['summary_content'] = summary_content
    
    # 6. Create Footer Container
    footer_container = create_footer_container(
        info_items=[_create_module_info_box()],
        tips=[
            "💡 Pilih arsitektur backbone yang sesuai dengan kebutuhan model Anda",
            "🔍 Selalu validasi konfigurasi sebelum melanjutkan"
        ]
    )
    
    # Store summary container for updates
    components['summary_container'] = summary_container
    
    # 6. Create Main Container
    main_container = create_main_container(
        components=[
            {'component': header_container.container, 'type': 'header'},
            {'component': form_container['container'], 'type': 'form'},
            {'component': summary_container, 'type': 'summary'},
            {'component': action_container['container'], 'type': 'action'},
            {'component': operation_container['container'], 'type': 'operation'},
            {'component': footer_container.container, 'type': 'footer'}
        ]
    )
    
    # 7. Assemble UI components
    ui_components = {
        'main_container': main_container,
        'header_container': header_container,
        'form_container': form_container,
        'action_container': action_container,
        'operation_container': operation_container,
        'footer_container': footer_container,
        'ui': main_container.container  # For backward compatibility
    }
    
    # Add action buttons for easy access
    action_buttons = {}
    for btn in ACTION_BUTTONS:
        btn_id = btn['id']
        button_ref = action_container['action_container'].get_button(btn_id)
        if button_ref is not None:
            action_buttons[f"{btn_id}_btn"] = button_ref
    
    ui_components.update(action_buttons)
    ui_components.update(components)
    ui_components['ui_initialized'] = True
    
    return ui_components

def create_backbone_child_components(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create child components untuk backbone configuration dengan shared container components
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary of child components
    """
    config = config or {}
    child_components = {}
    
    # === 1. HEADER CONTAINER ===
    
    # Create header container
    header_container = create_header_container(
        title="Model Configuration",
        subtitle="Konfigurasi backbone model YOLOv5 dengan EfficientNet-B4",
        icon="🤖"
    )
    child_components['header_container'] = header_container.container
    
    # === 2. FORM CONTAINER WITH TWO-COLUMN LAYOUT ===
    
    # Create form container to hold the two-column layout
    form_container = create_form_container()
    
    # Model form (left column)
    model_form = create_model_form(config)
    child_components['model_form'] = model_form
    
    # Config summary (right column)
    config_summary = create_config_summary(config)
    child_components['config_summary'] = config_summary
    
    # Create two-column layout with optimized spacing
    two_column_layout = widgets.HBox([
        widgets.Box(
            [model_form],
            layout=widgets.Layout(width='65%', padding='0 5px 0 0')
        ),
        widgets.Box(
            [config_summary],
            layout=widgets.Layout(width='35%', padding='0 0 0 5px')
        )
    ], layout=widgets.Layout(
        display='flex',
        gap='10px',
        width='100%',
        align_items='flex-start',
        margin='0',
        padding='0'
    ))
    
    # Add the two-column layout to the form container
    form_container['add_item'](two_column_layout, width='100%')
    child_components['form_container'] = form_container['container']
    
    # Store the two-column layout for reference
    child_components['two_column_layout'] = two_column_layout
    
    # === 2. ACTION CONTAINER WITH CONSOLIDATED BUTTONS ===
    
    # Create action container with multiple operations and default save/reset
    action_container = create_action_container(
        buttons=[
            {
                "id": "validate",
                "text": "🔍 Validate",
                "style": "info",
                "order": 1,
                "tooltip": "Validate backbone configuration"
            },
            {
                "id": "load",
                "text": "📥 Load Model",
                "style": "primary",
                "order": 2,
                "tooltip": "Load backbone model with current configuration"
            },
            {
                "id": "build",
                "text": "🏗️ Build",
                "style": "success",
                "order": 3,
                "tooltip": "Build backbone architecture"
            },
            {
                "id": "summary",
                "text": "📊 Summary", 
                "style": "warning",
                "order": 4,
                "tooltip": "Generate model summary and statistics"
            }
        ],
        title="🚀 Backbone Operations",
        alignment="left",
        show_save_reset=True  # Use default save/reset buttons
    )
    child_components['action_container'] = action_container['container']
    
    # Get action container and buttons
    action_container_obj = action_container['action_container']
    
    # Define button mappings
    button_mappings = {
        'validate_btn': 'validate',
        'load_btn': 'load',
        'build_btn': 'build',
        'summary_btn': 'summary'
    }
    
    # Add action buttons
    child_components.update({
        name: action_container_obj.get_button(btn_id)
        for name, btn_id in button_mappings.items()
    })
    
    # Add save/reset buttons
    child_components.update({
        'save_button': action_container_obj.save_button,
        'reset_button': action_container_obj.reset_button,
        'save_reset_buttons': {
            'save_button': action_container_obj.save_button,
            'reset_button': action_container_obj.reset_button
        }
    })
    
    # === 4. OPERATION CONTAINER ===
    
    # Create operation container
    operation_container = create_operation_container(
        title="Model Building",
        show_progress=True,
        show_logs=True,
        collapsible=True
    )
    child_components['operation_container'] = operation_container
    
    # Create footer container with log accordion and info box
    footer_container = create_footer_container(
        log_output=widgets.Output(),
        info_box=widgets.HTML(
            """
            <div class="alert alert-info" style="font-size: 0.9em; padding: 8px 12px;">
                <strong>Tips Model Backbone:</strong>
                <ul style="margin: 5px 0 0 15px; padding: 0;">
                    <li>Pilih backbone yang sesuai dengan kebutuhan deteksi</li>
                    <li>Layer mode yang tepat dapat meningkatkan performa model</li>
                    <li>Feature optimization membantu pada perangkat dengan memori terbatas</li>
                </ul>
            </div>
            """
        )
    )
    child_components['footer_container'] = footer_container.container
    
    # === 5. EXTRACT INDIVIDUAL COMPONENTS ===
    
    # Extract form widgets dari model_form for direct access
    if hasattr(model_form, 'backbone_dropdown'):
        child_components['backbone_dropdown'] = model_form.backbone_dropdown
    if hasattr(model_form, 'detection_layers_select'):
        child_components['detection_layers_select'] = model_form.detection_layers_select
    if hasattr(model_form, 'layer_mode_dropdown'):
        child_components['layer_mode_dropdown'] = model_form.layer_mode_dropdown
    if hasattr(model_form, 'feature_optimization_checkbox'):
        child_components['feature_optimization_checkbox'] = model_form.feature_optimization_checkbox
    if hasattr(model_form, 'mixed_precision_checkbox'):
        child_components['mixed_precision_checkbox'] = model_form.mixed_precision_checkbox
    
    # Extract buttons from action container
    if 'action_container' in action_container:
        action_ctrl = action_container['action_container']
        child_components['validate_btn'] = action_ctrl.get_button('validate')
        child_components['load_btn'] = action_ctrl.get_button('load')
        child_components['build_btn'] = action_ctrl.get_button('build')
        child_components['summary_btn'] = action_ctrl.get_button('summary')
    
    # Extract save/reset buttons
    if 'save_reset_buttons' in child_components:
        save_reset = child_components['save_reset_buttons']
        child_components['save_button'] = save_reset.get('save_button')
        child_components['reset_button'] = save_reset.get('reset_button')
    
    # === 6. ASSEMBLE MAIN CONTAINER ===
    
    # Create main container with all components using the new flexible component system
    main_container = create_main_container(
        components=[
            {'type': 'header', 'component': child_components['header_container'], 'order': 0},
            {'type': 'form', 'component': child_components['form_container'], 'order': 1},
            {'type': 'action', 'component': child_components['action_container'], 'order': 2},
            {'type': 'operation', 'component': child_components['operation_container']['container'], 'order': 3},
            {'type': 'footer', 'component': child_components['footer_container'], 'order': 4}
        ]
    )
    child_components['main_container'] = main_container
    
    return child_components

def _create_module_form_widgets(config: Dict[str, Any]) -> widgets.Widget:
    """Create module-specific form widgets.
    
    Args:
        config: Configuration dictionary for the form widgets
        
    Returns:
        Widget containing the form UI
    """
    return create_model_form(config)


def _create_module_summary_content(config: Dict[str, Any]) -> widgets.Widget:
    """Create summary content for the module.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Widget containing the summary content
    """
    summary = widgets.HTML(
        value="<h4>Backbone Configuration Summary</h4>"
             "<p>No configuration summary available yet.</p>"
    )
    return summary


def _create_module_info_box() -> widgets.Widget:
    """Create the info box content for the footer.
    
    Returns:
        Widget containing the info box content
    """
    info_content = """
    <div style='padding: 10px;'>
        <h4>About Backbone Configuration</h4>
        <p>Configure the base architecture for your model's feature extraction.</p>
        <p>Supported architectures: EfficientNet, ResNet, etc.</p>
    </div>
    """
    return widgets.HTML(info_content)


def get_layout_sections(child_components: Dict[str, Any]) -> list:
    """Get ordered layout sections untuk main container
    
    Args:
        child_components: Dictionary of child components
        
    Returns:
        List of widgets in display order
    """
    # With the new shared container approach, we only need to return the main container
    if 'main_container' in child_components:
        return [child_components['main_container']]
    
    # Fallback to legacy approach if main_container is not available
    sections = []
    
    # Add form container if available
    if 'form_container' in child_components:
        sections.append(child_components['form_container'])
    
    # Add config buttons container if available
    if 'config_buttons_container' in child_components:
        sections.append(child_components['config_buttons_container'])
    
    # Add action container if available
    if 'action_container' in child_components:
        sections.append(child_components['action_container'])
    
    # Add footer container if available
    if 'footer_container' in child_components:
        sections.append(child_components['footer_container'])
    
    return sections