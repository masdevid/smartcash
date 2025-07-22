"""
Evaluation UI Components - Using Standard Container Components
Creates comprehensive UI for 2×4 evaluation matrix (2 scenarios × 4 models = 8 tests)
"""

from typing import Dict, Any, List
import ipywidgets as widgets
from smartcash.ui.logger import get_module_logger
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.model.evaluation.constants import UI_CONFIG

def create_evaluation_ui(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create comprehensive evaluation UI components using standard containers.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        Dictionary containing all UI components
    """
    try:
        logger = get_module_logger("smartcash.ui.model.evaluation.components")
        
        # Create header with title and subtitle
        header_container = create_header_container(
            title=UI_CONFIG['title'],
            subtitle=UI_CONFIG['subtitle'],
            icon='🎯',  # Chart emoji for evaluation
            show_environment=True,
            environment='local',
            config_path='evaluation_config.yaml'
        )
        
        # Create comprehensive 2-column layout: 
        # Left: Scenario and Metrics checkboxes
        # Right: Available best models selection
        main_form_row = _create_main_form_layout(config)
        
        # Create action container with evaluation buttons
        action_container = create_action_container(
            buttons=[
                {
                    'id': 'run_evaluation',
                    'text': '🚀 Run Evaluation',
                    'tooltip': 'Execute evaluation based on selected configuration',
                    'style': 'success'
                },
                {
                    'id': 'refresh_models',
                    'text': '🔄 Refresh Models',
                    'tooltip': 'Refresh available model checkpoints',
                    'style': 'info'
                }
            ],
            title="Evaluation Actions",
            show_save_reset=True
        )
        
        # Create operation container for progress and logging (following backbone pattern)
        operation_container = create_operation_container(
            show_progress=True,
            show_dialog=True,
            show_logs=True,
            progress_levels='dual',
            log_module_name=UI_CONFIG['module_name'],
            # log_namespace_filter='evaluation',  # Temporarily disabled
            log_height="150px",
            collapsible=True,
            collapsed=False
        )
        
        # Create summary container for results
        summary_container = create_summary_container(
            theme="info",
            title="📊 Evaluation Results",
            icon="📊"
        )
        
        # Extract widget components from container objects (following backbone pattern)
        header_widget = header_container.container if hasattr(header_container, 'container') else header_container
        action_widget = action_container.get('container') if isinstance(action_container, dict) else action_container
        operation_widget = operation_container['container'] if isinstance(operation_container, dict) else operation_container
        summary_widget = summary_container.container if hasattr(summary_container, 'container') else summary_container
        
        # Create main container with compact layout
        main_container = create_main_container(
            components=[
                {'type': 'header', 'component': header_widget, 'order': 0},
                {'type': 'form', 'component': main_form_row, 'order': 1},
                {'type': 'action', 'component': action_widget, 'order': 2},
                {'type': 'operation', 'component': operation_widget, 'order': 3},
                {'type': 'custom', 'component': summary_widget, 'order': 4}
            ],
            title=UI_CONFIG['title'],
            description=UI_CONFIG['description']
        )
        
        logger.debug(f"🎨 Created evaluation UI with standard containers")
        
        # Extract the actual widget from main container
        main_widget = main_container.container if hasattr(main_container, 'container') else main_container
        
        # Extract and validate action buttons
        action_button_widgets = action_container.get('buttons', {})
        
        return {
            'main_container': main_widget,  # Use the actual widget
            'header_container': header_container,
            'main_form_row': main_form_row,
            'action_container': action_container,
            'operation_container': operation_container,  # Store full container object like backbone
            'summary_container': summary_container,
            'components_count': 5,
            # Button references from action container
            'run_evaluation': action_container.get('buttons', {}).get('run_evaluation'),
            'refresh_models': action_container.get('buttons', {}).get('refresh_models'),
            'save': action_container.get('buttons', {}).get('save'),
            'reset': action_container.get('buttons', {}).get('reset')
        }
        
    except Exception as e:
        logger = get_module_logger("smartcash.ui.model.evaluation.components")
        logger.error(f"Failed to create evaluation UI: {e}")
        
        # Create minimal error container
        error_container = widgets.VBox([
            widgets.HTML(f"<h3 style='color: red;'>❌ UI Creation Failed</h3>"),
            widgets.HTML(f"<p>Error: {str(e)}</p>")
        ])
        
        return {
            'main_container': error_container,
            'error': str(e),
            'components_count': 0
        }

def _create_main_form_layout(config: Dict[str, Any]) -> widgets.Widget:
    """Create comprehensive 2-column layout: Left (scenarios + metrics), Right (available models)."""
    
    # Left column: Scenario and Metrics checkboxes
    left_column_items = []
    
    # Scenario selection section
    left_column_items.append(widgets.HTML("<h3>📐 Evaluation Scenarios</h3>"))
    
    # Scenario checkboxes
    scenario_checkboxes = []
    from smartcash.ui.model.evaluation.constants import RESEARCH_SCENARIOS
    for scenario_info in RESEARCH_SCENARIOS.values():
        scenario_checkbox = widgets.Checkbox(
            value=scenario_info.get('enabled', True),
            description=f"{scenario_info['icon']} {scenario_info['name']}",
            tooltip=scenario_info['description'],
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='auto', margin='3px 0')
        )
        scenario_checkboxes.append(scenario_checkbox)
    
    left_column_items.extend(scenario_checkboxes)
    
    # Add separator
    left_column_items.append(widgets.HTML('<div style="margin: 15px 0; border-top: 1px solid #dee2e6;"></div>'))
    
    # Metrics selection section
    left_column_items.append(widgets.HTML("<h3>📊 Evaluation Metrics</h3>"))
    
    # Metrics checkboxes
    metrics_checkboxes = []
    from smartcash.ui.model.evaluation.constants import EVALUATION_METRICS
    for metric_info in EVALUATION_METRICS.values():
        metric_checkbox = widgets.Checkbox(
            value=True,
            description=f"{metric_info['icon']} {metric_info['name']}",
            tooltip=metric_info['description'],
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='auto', margin='3px 0')
        )
        metrics_checkboxes.append(metric_checkbox)
    
    left_column_items.extend(metrics_checkboxes)
    
    # Right column: Available best models selection
    right_column_items = []
    right_column_items.append(widgets.HTML("<h3>🤖 Available Best Models</h3>"))
    
    # Refresh button
    refresh_button = widgets.Button(
        description="🔄 Refresh Models",
        tooltip="Refresh available models list",
        button_style='info',
        layout=widgets.Layout(width='150px', height='32px', margin='0 0 15px 0')
    )
    right_column_items.append(refresh_button)
    
    # Model selection checkboxes (using proper checkpoint format)
    model_checkboxes = _create_model_selection_checkboxes(config)
    right_column_items.extend(model_checkboxes)
    
    # Create the two-column layout with proper flex handling
    left_column = widgets.VBox(left_column_items, layout=widgets.Layout(
        width='48%', 
        margin='0 1% 0 0',
        overflow='hidden'
    ))
    right_column = widgets.VBox(right_column_items, layout=widgets.Layout(
        width='48%', 
        margin='0 0 0 1%',
        overflow='hidden'
    ))
    
    main_layout = widgets.HBox([left_column, right_column], layout=widgets.Layout(
        margin='15px 0',
        overflow='hidden',
        width='100%'
    ))
    
    # Store widget references for form value extraction
    main_layout._scenario_checkboxes = {scenario: cb for scenario, cb in zip(RESEARCH_SCENARIOS.keys(), scenario_checkboxes)}
    main_layout._metrics_checkboxes = {metric: cb for metric, cb in zip(EVALUATION_METRICS.keys(), metrics_checkboxes)}
    main_layout._refresh_button = refresh_button
    
    return main_layout

def _create_model_selection_checkboxes(_config: Dict[str, Any]) -> List[widgets.Widget]:
    """
    Create single model selection based on available checkpoints with proper naming format.
    Implements single selection (radio buttons) instead of multiple checkboxes.
    Shows empty placeholder when no models available (fail-fast principle).
    """
    
    model_widgets = []
    
    # TODO: Replace with actual checkpoint discovery from training results
    # This will be populated by the post-init hook in the evaluation module
    available_models = []  # Start with empty list for fail-fast demonstration
    
    if available_models:
        # Create radio button group for single model selection
        model_options = []
        for model in available_models:
            display_name = f"📦 {model['backbone']} ({model['date']}) - mAP: {model['map_score']:.3f}"
            model_options.append((display_name, model['checkpoint']))
        
        # Single model selection using radio buttons
        model_radio = widgets.RadioButtons(
            options=model_options,
            value=model_options[0][1] if model_options else None,  # Select first by default
            description='Select Model:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='auto', margin='10px 0')
        )
        
        model_widgets.append(model_radio)
        
        # Add detailed info for selected model
        model_info_html = widgets.HTML(
            value="""
            <div id='selected-model-info' style='background: #e8f5e9; border: 1px solid #c8e6c9; border-radius: 6px; padding: 12px; margin: 10px 0; font-size: 0.9em;'>
                <div style='font-weight: bold; color: #2e7d32; margin-bottom: 6px;'>✅ Selected Model Details</div>
                <div style='color: #4caf50;'>Select a model above to see details</div>
            </div>
            """,
            layout=widgets.Layout(margin='10px 0')
        )
        model_widgets.append(model_info_html)
        
        # Add evaluation requirements info
        requirements_info = widgets.HTML(
            value="""
            <div style='background: #fff3e0; border: 1px solid #ffcc02; border-radius: 6px; padding: 10px; margin: 10px 0; font-size: 0.85em;'>
                <div style='font-weight: bold; color: #f57c00; margin-bottom: 6px;'>ℹ️ Evaluation Requirements</div>
                <div style='color: #ef6c00;'>
                    • Only 1 best model can be selected at a time<br>
                    • Model must be validated and checkpoint available<br>
                    • Evaluation will fail fast if no valid models found
                </div>
            </div>
            """,
            layout=widgets.Layout(margin='10px 0')
        )
        model_widgets.append(requirements_info)
        
    else:
        # Show empty placeholder with dashed box (fail-fast principle)
        empty_placeholder = widgets.HTML(
            value="""
            <div style='text-align: center; color: #6c757d; padding: 40px 20px; border: 3px dashed #dee2e6; border-radius: 12px; background: #fafafa;'>
                <div style='font-size: 4em; opacity: 0.2; margin-bottom: 20px;'>🤖</div>
                <h4 style='margin: 10px 0; color: #dc3545; font-weight: 600;'>No Best Models Available</h4>
                <p style='margin: 8px 0; font-size: 1em; color: #6c757d;'>
                    Complete the <strong>Training Workflow</strong> first to generate best model checkpoints
                </p>
                <div style='background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 12px; margin: 15px 0; font-size: 0.85em;'>
                    <div style='font-weight: bold; color: #495057; margin-bottom: 6px;'>Expected checkpoint format:</div>
                    <code style='background: #f8f9fa; padding: 2px 6px; border-radius: 3px; color: #d63384;'>
                        best_{model_name}_{backbone}_{date:%Y%m%d}.pt
                    </code>
                </div>
                <div style='margin-top: 20px; padding: 10px; background: #ffebee; border: 1px solid #f8bbd9; border-radius: 6px;'>
                    <div style='font-weight: bold; color: #d32f2f; font-size: 0.9em;'>⚠️ Fail-Fast Principle</div>
                    <div style='color: #c62828; font-size: 0.85em; margin-top: 4px;'>
                        Evaluation is disabled until valid best models are detected
                    </div>
                </div>
            </div>
            """,
            layout=widgets.Layout(margin='15px 0')
        )
        model_widgets.append(empty_placeholder)
    
    return model_widgets

def _create_model_display_widget(_config: Dict[str, Any]) -> widgets.Widget:
    """Create model display widget showing available best models (display only, not selection)."""
    
    # Container for model display
    model_display_items = []
    model_display_items.append(widgets.HTML("<h3>📋 Available Best Models</h3>"))
    
    # Create refresh button with icon
    refresh_button = widgets.Button(
        description="🔄 Refresh",
        tooltip="Refresh available models list",
        button_style='info',
        layout=widgets.Layout(width='100px', height='28px', margin='0 0 10px 0')
    )
    
    # Placeholder for refresh functionality (will be connected in the module)
    refresh_button.on_click(lambda b: _refresh_model_list())
    model_display_items.append(refresh_button)
    
    # Mock model data following {scenario}_{backbone}_{layer} format
    # TODO: Replace with actual model discovery from training results
    available_models = [
        {'name': 'position_yolov5_efficientnet-b4', 'map': 0.847, 'scenario': 'position', 'backbone': 'yolov5_efficientnet-b4'},
        {'name': 'position_yolov5_cspdarknet', 'map': 0.782, 'scenario': 'position', 'backbone': 'yolov5_cspdarknet'},
        {'name': 'lighting_yolov5_efficientnet-b4', 'map': 0.823, 'scenario': 'lighting', 'backbone': 'yolov5_efficientnet-b4'},
        {'name': 'lighting_yolov5_cspdarknet', 'map': 0.798, 'scenario': 'lighting', 'backbone': 'yolov5_cspdarknet'}
    ]
    
    if available_models:
        # Display models in a compact list format (display only, no selection checkboxes)
        for model in available_models:
            model_info = widgets.HTML(
                value=f"""
                <div style='background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 8px; margin: 3px 0;'>
                    <div style='font-weight: bold; color: #495057;'>{model['name']}</div>
                    <div style='font-size: 0.85em; color: #6c757d;'>mAP: {model['map']:.3f} | Scenario: {model['scenario']}</div>
                </div>
                """,
                layout=widgets.Layout(margin='2px 0')
            )
            model_display_items.append(model_info)
        
        # Add info text
        info_text = widgets.HTML(
            value="""
            <div style='font-size: 0.8em; color: #6c757d; margin-top: 10px; padding: 5px;'>
                ℹ️ Models are selected automatically based on scenario, backbone, and layer settings above.
            </div>
            """,
            layout=widgets.Layout(margin='5px 0')
        )
        model_display_items.append(info_text)
        
    else:
        # Show empty placeholder when no models available
        empty_placeholder = widgets.HTML(
            value="""
            <div style='text-align: center; color: #6c757d; padding: 20px 10px;'>
                <div style='font-size: 2em; opacity: 0.3; margin-bottom: 10px;'>🤖</div>
                <p style='margin: 5px 0; font-size: 0.9em;'>No trained models available</p>
                <p style='margin: 5px 0; font-size: 0.8em; color: #999;'>
                    Complete backbone → training workflow first
                </p>
            </div>
            """,
            layout=widgets.Layout(margin='10px 0')
        )
        model_display_items.append(empty_placeholder)
    
    return widgets.VBox(model_display_items, layout=widgets.Layout(margin='5px 0'))

def _refresh_model_list():
    """Placeholder function for refreshing model list. Will be implemented in the module."""
    # This will be connected to the actual refresh operation in the evaluation module
    pass

# Legacy function - replaced by _create_main_form_layout
def _create_execution_model_row(config: Dict[str, Any]) -> widgets.Widget:
    """Create compact 2-column row with execution options and model configurations."""
    
    # Left column: Execution Options
    execution_items = []
    execution_items.append(widgets.HTML("<h3>⚙️ Execution Options</h3>"))
    
    # Scenario selection (radio buttons) with correct test counts
    scenario_radio = widgets.RadioButtons(
        options=[
            ('All Scenarios (8 tests)', 'all_scenarios'),
            ('Position Only (4 tests)', 'position_only'), 
            ('Lighting Only (4 tests)', 'lighting_only')
        ],
        value=config.get('evaluation', {}).get('execution', {}).get('run_mode', 'all_scenarios'),
        description='Scenarios:',
        style={'description_width': '80px'},
        layout=widgets.Layout(margin='5px 0')
    )
    execution_items.append(scenario_radio)
    
    # Parallel execution
    parallel_execution = widgets.Checkbox(
        value=config.get('evaluation', {}).get('execution', {}).get('parallel_execution', False),
        description="⚡ Parallel execution",
        style={'description_width': 'initial'},
        layout=widgets.Layout(margin='5px 0')
    )
    execution_items.append(parallel_execution)
    
    # Save intermediate results
    save_intermediate = widgets.Checkbox(
        value=config.get('evaluation', {}).get('execution', {}).get('save_intermediate_results', True),
        description="💾 Save intermediate results", 
        style={'description_width': 'initial'},
        layout=widgets.Layout(margin='5px 0')
    )
    execution_items.append(save_intermediate)
    
    # Right column: Model Selection (actual selection logic)
    model_items = []
    model_items.append(widgets.HTML("<h3>🤖 Model Selection</h3>"))
    
    # Backbone selection (radio buttons) - this determines which models to evaluate
    from smartcash.ui.model.evaluation.constants import MODEL_COMBINATIONS
    backbone_options = list(set([model['backbone'] for model in MODEL_COMBINATIONS]))
    backbone_radio = widgets.RadioButtons(
        options=backbone_options,
        value=backbone_options[0],
        description='Backbone:',
        style={'description_width': '80px'},
        layout=widgets.Layout(margin='5px 0')
    )
    model_items.append(backbone_radio)
    
    # Layer mode selection (radio buttons) - this determines which layer mode to use
    layer_options = list(set([model['layer_mode'] for model in MODEL_COMBINATIONS]))
    layer_radio = widgets.RadioButtons(
        options=layer_options,
        value=layer_options[0],
        description='Layer Mode:',
        style={'description_width': '80px'},
        layout=widgets.Layout(margin='5px 0')
    )
    model_items.append(layer_radio)
    
    # Auto-select best trained models checkbox
    auto_select = widgets.Checkbox(
        value=config.get('evaluation', {}).get('models', {}).get('auto_select_best', True),
        description="🎯 Auto-select best trained models",
        tooltip="Automatically select best models with format: {scenario}_{backbone}_{layer}",
        style={'description_width': 'initial'},
        layout=widgets.Layout(margin='5px 0')
    )
    model_items.append(auto_select)
    
    # Create 2-column layout with proper overflow handling
    left_column = widgets.VBox(execution_items, layout=widgets.Layout(
        width='48%', 
        margin='0 1% 0 0',
        overflow='hidden'
    ))
    right_column = widgets.VBox(model_items, layout=widgets.Layout(
        width='48%', 
        margin='0 0 0 1%',
        overflow='hidden'
    ))
    
    return widgets.HBox([left_column, right_column], layout=widgets.Layout(
        margin='10px 0',
        overflow='hidden',
        width='100%'
    ))


# Legacy function - replaced by _create_main_form_layout
def _create_metrics_model_row(config: Dict[str, Any]) -> widgets.Widget:
    """Create 2-column row with metrics selection and available best models."""
    
    # Left column: Metrics selection (restored original)
    metrics_items = []
    metrics_items.append(widgets.HTML("<h3>📈 Evaluation Metrics</h3>"))
    
    # Create metrics checkboxes in a more compact horizontal layout
    metrics_row1 = []
    metrics_row2 = []
    
    from smartcash.ui.model.evaluation.constants import EVALUATION_METRICS
    metric_keys = list(EVALUATION_METRICS.keys())
    half_point = len(metric_keys) // 2
    
    # First row of metrics
    for metric_key in metric_keys[:half_point]:
        metric_info = EVALUATION_METRICS[metric_key]
        checkbox = widgets.Checkbox(
            value=True,
            description=f"{metric_info['icon']} {metric_info['name']}",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px', margin='2px 10px 2px 0')
        )
        metrics_row1.append(checkbox)
    
    # Second row of metrics  
    for metric_key in metric_keys[half_point:]:
        metric_info = EVALUATION_METRICS[metric_key]
        checkbox = widgets.Checkbox(
            value=True,
            description=f"{metric_info['icon']} {metric_info['name']}",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px', margin='2px 10px 2px 0')
        )
        metrics_row2.append(checkbox)
    
    # Add the metric rows
    if metrics_row1:
        metrics_items.append(widgets.HBox(metrics_row1))
    if metrics_row2:
        metrics_items.append(widgets.HBox(metrics_row2))
    
    # Right column: Available Best Models (display only)
    model_display_widget = _create_model_display_widget(config)
    
    # Create 2-column layout with proper overflow handling
    left_column = widgets.VBox(metrics_items, layout=widgets.Layout(
        width='48%', 
        margin='0 1% 0 0',
        overflow='hidden'
    ))
    right_column = widgets.VBox([model_display_widget], layout=widgets.Layout(
        width='48%', 
        margin='0 0 0 1%',
        overflow='hidden'
    ))
    
    return widgets.HBox([left_column, right_column], layout=widgets.Layout(
        margin='10px 0',
        overflow='hidden',
        width='100%'
    ))


