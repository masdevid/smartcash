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
            icon='📊'  # Chart emoji for evaluation
        )
        
        # Create comprehensive 2-column layout: 
        # Left: Scenario and Metrics checkboxes
        # Right: Available best models selection
        main_form_row = _create_main_form_layout(config)
        
        # Create action container with single run scenario button (no double icons)
        action_container = create_action_container(
            buttons=[
                {
                    'id': 'run_scenario',
                    'text': 'Run Scenario',
                    'tooltip': 'Execute evaluation based on selected configuration',
                    'style': 'success'
                }
            ],
            title="Evaluation Actions",
            show_save_reset=False
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
            log_entry_style='compact',
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
        
        return {
            'main_container': main_widget,
            'header_container': header_container,
            'main_form_row': main_form_row,
            'action_container': action_container,
            'operation_container': operation_container,  # Store full container object like backbone
            'summary_container': summary_container,
            'components_count': 5
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
    
    # Create the two-column layout
    left_column = widgets.VBox(left_column_items, layout=widgets.Layout(width='48%', margin='0 1% 0 0'))
    right_column = widgets.VBox(right_column_items, layout=widgets.Layout(width='48%', margin='0 0 0 1%'))
    
    main_layout = widgets.HBox([left_column, right_column], layout=widgets.Layout(margin='15px 0'))
    
    # Store widget references for form value extraction
    main_layout._scenario_checkboxes = {scenario: cb for scenario, cb in zip(RESEARCH_SCENARIOS.keys(), scenario_checkboxes)}
    main_layout._metrics_checkboxes = {metric: cb for metric, cb in zip(EVALUATION_METRICS.keys(), metrics_checkboxes)}
    main_layout._refresh_button = refresh_button
    
    return main_layout

def _create_model_selection_checkboxes(_config: Dict[str, Any]) -> List[widgets.Widget]:
    """Create model selection checkboxes based on available checkpoints with proper naming format."""
    
    model_widgets = []
    
    # Mock available models following the checkpoint format: best_{model_name}_{backbone}_{date:%Y%m%d}.pt
    # TODO: Replace with actual checkpoint discovery from training results
    available_models = [
        {
            'checkpoint': 'best_smartcash_efficientnet_b4_20250719.pt',
            'model_name': 'smartcash',
            'backbone': 'efficientnet_b4',
            'date': '20250719',
            'map_score': 0.847,
            'scenarios': ['position_variation', 'lighting_variation']
        },
        {
            'checkpoint': 'best_smartcash_cspdarknet_20250718.pt',
            'model_name': 'smartcash',
            'backbone': 'cspdarknet',
            'date': '20250718',
            'map_score': 0.782,
            'scenarios': ['position_variation', 'lighting_variation']
        },
        {
            'checkpoint': 'best_smartcash_efficientnet_b4_20250717.pt',
            'model_name': 'smartcash',
            'backbone': 'efficientnet_b4',
            'date': '20250717',
            'map_score': 0.823,
            'scenarios': ['position_variation']
        }
    ]
    
    if available_models:
        for model in available_models:
            # Create checkbox for each model
            model_checkbox = widgets.Checkbox(
                value=True,  # Selected by default
                description=f"📦 {model['backbone']} ({model['date']})",
                tooltip=f"Checkpoint: {model['checkpoint']} | mAP: {model['map_score']:.3f}",
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='auto', margin='3px 0')
            )
            
            # Add detailed model info as HTML
            model_info = widgets.HTML(
                value=f"""
                <div style='background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 8px; margin: 2px 0 8px 20px; font-size: 0.85em;'>
                    <div style='font-weight: bold; color: #495057;'>{model['checkpoint']}</div>
                    <div style='color: #6c757d;'>mAP: {model['map_score']:.3f} | Scenarios: {", ".join(model['scenarios'])}</div>
                </div>
                """,
                layout=widgets.Layout(margin='0 0 5px 0')
            )
            
            model_widgets.extend([model_checkbox, model_info])
        
        # Add selection controls
        selection_controls = widgets.HBox([
            widgets.Button(
                description="✅ Select All",
                button_style='success',
                layout=widgets.Layout(width='100px', height='28px', margin='5px 5px 5px 0')
            ),
            widgets.Button(
                description="❌ Clear All",
                button_style='warning',
                layout=widgets.Layout(width='100px', height='28px', margin='5px 0')
            )
        ], layout=widgets.Layout(margin='10px 0'))
        
        model_widgets.append(selection_controls)
        
    else:
        # Show empty placeholder
        empty_placeholder = widgets.HTML(
            value="""
            <div style='text-align: center; color: #6c757d; padding: 30px 10px; border: 2px dashed #dee2e6; border-radius: 8px;'>
                <div style='font-size: 3em; opacity: 0.3; margin-bottom: 15px;'>🤖</div>
                <p style='margin: 5px 0; font-size: 1.1em; font-weight: 600;'>No Trained Models Available</p>
                <p style='margin: 5px 0; font-size: 0.9em; color: #999;'>
                    Complete the training workflow first to generate checkpoints
                </p>
                <p style='margin: 10px 0; font-size: 0.8em; color: #6c757d;'>
                    Expected format: <code>best_{model_name}_{backbone}_{date:%Y%m%d}.pt</code>
                </p>
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
    
    # Create 2-column layout
    left_column = widgets.VBox(execution_items, layout=widgets.Layout(width='48%', margin='0 1% 0 0'))
    right_column = widgets.VBox(model_items, layout=widgets.Layout(width='48%', margin='0 0 0 1%'))
    
    return widgets.HBox([left_column, right_column], layout=widgets.Layout(margin='10px 0'))


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
    
    # Create 2-column layout
    left_column = widgets.VBox(metrics_items, layout=widgets.Layout(width='48%', margin='0 1% 0 0'))
    right_column = widgets.VBox([model_display_widget], layout=widgets.Layout(width='48%', margin='0 0 0 1%'))
    
    return widgets.HBox([left_column, right_column], layout=widgets.Layout(margin='10px 0'))


