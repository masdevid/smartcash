"""
File: smartcash/ui/model/evaluate/components/evaluation_ui.py
Description: Model evaluation UI following SmartCash standardized template.

This module provides the user interface for evaluating trained models
with scenario-based testing, checkpoint management, and comprehensive metrics.

Container Order:
1. Header Container (Title, Status)
2. Form Container (Scenario and Model Selection)
3. Action Container (Evaluate/Load/Compare/Export Buttons)
4. Summary Container (Evaluation Results)
5. Operation Container (Progress + Logs)
6. Footer Container (Tips and Info)
"""

from typing import Optional, Dict, Any, List
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
from ..constants import UI_CONFIG, BUTTON_CONFIG

# Module constants (for validator compliance)
UI_CONFIG = UI_CONFIG
BUTTON_CONFIG = BUTTON_CONFIG


@handle_ui_errors(error_component_title="Evaluation UI Error")
def create_evaluation_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create the model evaluation UI following SmartCash standards.
    
    This function creates a complete UI for evaluating trained models
    with the following sections:
    - Scenario and model selection
    - Evaluation configuration options
    - Checkpoint management
    - Results comparison and export
    
    Args:
        config: Optional configuration dictionary for the UI
        **kwargs: Additional keyword arguments passed to UI components
        
    Returns:
        Dictionary containing all UI components and their references with 'ui_components' key
        
    Example:
        >>> ui = create_evaluation_ui()
        >>> display(ui['ui'])  # Display the UI
    """
    # Initialize configuration and components dictionary
    current_config = config or {}
    ui_components = {
        'config': current_config,
        'containers': {},
        'widgets': {}
    }
    
    # === 1. Create Header Container ===
    header_container = create_header_container(
        title=f"{UI_CONFIG['icon']} {UI_CONFIG['title']}",
        subtitle=UI_CONFIG['subtitle'],
        status_message="Ready to evaluate models",
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
        form_rows=form_widgets['form_rows'],
        layout_type=LayoutType.COLUMN,
        container_margin="0",
        container_padding="16px",
        gap="12px"
    )
    
    # Store references
    ui_components['containers']['form'] = form_container
    ui_components['widgets'].update(form_widgets['widgets'])
    
    # === 3. Create Action Container ===
    # Create action buttons from BUTTON_CONFIG
    action_buttons = []
    for button_id, btn_config in BUTTON_CONFIG.items():
        action_buttons.append({
            'name': button_id,
            'label': btn_config['text'],
            'button_style': btn_config['style'],
            'tooltip': btn_config['tooltip'],
            'icon': 'play' if button_id == 'evaluate' else 'folder' if button_id == 'load_checkpoint' else 'chart-bar' if button_id == 'compare' else 'download'
        })
    
    action_container = create_action_container(
        buttons=action_buttons,
        title="🎯 Evaluation Actions",
        alignment="left"
    )
    
    # Store references
    ui_components['containers']['actions'] = action_container
    if hasattr(action_container, 'get'):
        for btn in action_buttons:
            ui_components['widgets'][f'{btn["name"]}_button'] = action_container.get(btn['name'])
    
    # === 4. Create Summary Container ===
    summary_content = _create_module_summary_content(current_config)
    summary_container = create_summary_container(
        title="📊 Evaluation Results",
        theme="success",
        icon="🎯"
    )
    summary_container.set_content(summary_content)
    
    ui_components['containers']['summary'] = summary_container
    
    # === 5. Create Operation Container ===
    operation_container = create_operation_container(
        title="🔄 Evaluation Progress",
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
    main_container = create_main_container(
        header=ui_components['containers']['header']['container'],
        body=widgets.VBox([
            ui_components['containers']['form']['container'],
            ui_components['containers']['actions']['container'],
            ui_components['containers']['summary'].container,
            ui_components['containers']['operation']['container']
        ]),
        footer=ui_components['containers']['footer']['container'],
        container_config={
            'margin': '0 auto',
            'max_width': '1200px',
            'padding': '10px',
            'border': '1px solid #e0e0e0',
            'border_radius': '5px',
            'box_shadow': '0 1px 3px rgba(0,0,0,0.1)'
        }
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
    
    return result


def _create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create module-specific form widgets for evaluation configuration.
    
    Args:
        config: Configuration dictionary for the form widgets
        
    Returns:
        Dictionary containing the form UI and widget references
    """
    from ..constants import SCENARIO_CONFIGS, MODEL_CONFIGS, AVAILABLE_METRICS, METRIC_CONFIGS
    
    # Scenario Selection
    scenario_checkboxes = []
    scenario_options = [(f"{SCENARIO_CONFIGS[key]['icon']} {SCENARIO_CONFIGS[key]['name']}", key) 
                       for key in SCENARIO_CONFIGS.keys()]
    
    position_checkbox = widgets.Checkbox(
        value=config.get('position_variation', True),
        description='📐 Position Variation',
        layout=widgets.Layout(width='50%', margin='5px 0')
    )
    
    lighting_checkbox = widgets.Checkbox(
        value=config.get('lighting_variation', True),
        description='💡 Lighting Variation',
        layout=widgets.Layout(width='50%', margin='5px 0')
    )
    
    # Model Selection
    cspdarknet_checkbox = widgets.Checkbox(
        value=config.get('cspdarknet', True),
        description='🌙 CSPDarknet',
        layout=widgets.Layout(width='50%', margin='5px 0')
    )
    
    efficientnet_checkbox = widgets.Checkbox(
        value=config.get('efficientnet_b4', True),
        description='⚡ EfficientNet-B4',
        layout=widgets.Layout(width='50%', margin='5px 0')
    )
    
    # Evaluation Settings
    confidence_slider = widgets.FloatSlider(
        value=config.get('confidence_threshold', 0.25),
        min=0.0,
        max=1.0,
        step=0.05,
        description='Confidence:',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    iou_slider = widgets.FloatSlider(
        value=config.get('iou_threshold', 0.45),
        min=0.0,
        max=1.0,
        step=0.05,
        description='IoU Threshold:',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    variations_slider = widgets.IntSlider(
        value=config.get('num_variations', 5),
        min=1,
        max=10,
        step=1,
        description='Variations:',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    # Metrics Selection
    metric_checkboxes = []
    for metric_key in AVAILABLE_METRICS:
        metric_config = METRIC_CONFIGS[metric_key]
        checkbox = widgets.Checkbox(
            value=metric_config.get('default_enabled', False),
            description=f"{metric_config['icon']} {metric_config['name']}",
            layout=widgets.Layout(width='33%', margin='5px 0')
        )
        metric_checkboxes.append(checkbox)
    
    # Create form rows
    form_rows = [
        [widgets.HTML("<h4>📐 Test Scenarios</h4>")],
        [widgets.HBox([position_checkbox, lighting_checkbox])],
        [widgets.HTML("<h4>🤖 Model Selection</h4>")],
        [widgets.HBox([cspdarknet_checkbox, efficientnet_checkbox])],
        [widgets.HTML("<h4>⚙️ Evaluation Settings</h4>")],
        [confidence_slider],
        [iou_slider],
        [variations_slider],
        [widgets.HTML("<h4>📊 Metrics to Calculate</h4>")],
        [widgets.HBox(metric_checkboxes[:3]) if len(metric_checkboxes) >= 3 else widgets.HBox(metric_checkboxes)],
        [widgets.HBox(metric_checkboxes[3:]) if len(metric_checkboxes) > 3 else widgets.HBox()]
    ]
    
    return {
        'form_rows': form_rows,
        'widgets': {
            'position_checkbox': position_checkbox,
            'lighting_checkbox': lighting_checkbox,
            'cspdarknet_checkbox': cspdarknet_checkbox,
            'efficientnet_checkbox': efficientnet_checkbox,
            'confidence_slider': confidence_slider,
            'iou_slider': iou_slider,
            'variations_slider': variations_slider,
            **{f'{AVAILABLE_METRICS[i]}_checkbox': checkbox for i, checkbox in enumerate(metric_checkboxes)}
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
    return """
    <div style="padding: 10px;">
        <h5>🎯 Evaluation Overview</h5>
        <p>Model evaluation results and metrics will be displayed here after running tests.</p>
        <ul>
            <li>Scenarios tested: <span id="scenarios-tested">-</span></li>
            <li>Models evaluated: <span id="models-evaluated">-</span></li>
            <li>Best mAP score: <span id="best-map">-</span></li>
            <li>Average inference time: <span id="avg-inference">-</span></li>
        </ul>
        <div style="margin-top: 10px; padding: 8px; background: #e7f3ff; border-radius: 4px;">
            <strong>Status:</strong> <span id="evaluation-status">Ready to start evaluation</span>
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
            <h4 style="margin-top: 0; color: #0d47a1;">🎯 Evaluation Guide</h4>
            <p>This module helps you evaluate your trained models with scenario-based testing.</p>
            <ol style="margin: 8px 0 0 16px; padding-left: 8px;">
                <li>Select test scenarios (position/lighting variations)</li>
                <li>Choose models to evaluate (CSPDarknet/EfficientNet)</li>
                <li>Configure evaluation settings and metrics</li>
                <li>Click 'Start Evaluation' to run tests</li>
                <li>Use 'Compare Results' to analyze performance</li>
            </ol>
        </div>
        """
    )