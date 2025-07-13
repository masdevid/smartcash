"""
Evaluation UI Components - Using Standard Container Components
Creates comprehensive UI for 2×4 evaluation matrix (2 scenarios × 4 models = 8 tests)
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display
from smartcash.ui.logger import get_module_logger
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.model.evaluation.constants import (
    UI_CONFIG,
    RESEARCH_SCENARIOS,
    MODEL_COMBINATIONS,
    EVALUATION_MATRIX,
    BUTTON_CONFIG,
    EVALUATION_METRICS
)

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
        
        # Create header container
        header_container = create_header_container(
            title=UI_CONFIG['title'],
            description=UI_CONFIG['description'],
            icon=UI_CONFIG['icon'],
            stats=[
                {'label': 'Scenarios', 'value': str(len(RESEARCH_SCENARIOS)), 'icon': '📊'},
                {'label': 'Model Types', 'value': str(len(MODEL_COMBINATIONS)), 'icon': '🤖'},
                {'label': 'Total Tests', 'value': str(len(EVALUATION_MATRIX)), 'icon': '🎯'},
                {'label': 'Metrics', 'value': str(len(EVALUATION_METRICS)), 'icon': '📈'}
            ]
        )
        
        # Create form sections for configuration
        scenario_section = _create_scenario_form_section(config)
        model_section = _create_model_form_section(config)
        metrics_section = _create_metrics_form_section(config)
        execution_section = _create_execution_form_section(config)
        
        # Create action container with evaluation buttons
        action_container = create_action_container(
            buttons=[
                {
                    'id': 'run_all_scenarios',
                    'text': BUTTON_CONFIG['run_all_scenarios']['text'],
                    'tooltip': BUTTON_CONFIG['run_all_scenarios']['tooltip'],
                    'style': 'success',
                    'icon': '🚀'
                },
                {
                    'id': 'run_position_scenario',
                    'text': BUTTON_CONFIG['run_position_scenario']['text'],
                    'tooltip': BUTTON_CONFIG['run_position_scenario']['tooltip'],
                    'style': 'info',
                    'icon': '📐'
                },
                {
                    'id': 'run_lighting_scenario', 
                    'text': BUTTON_CONFIG['run_lighting_scenario']['text'],
                    'tooltip': BUTTON_CONFIG['run_lighting_scenario']['tooltip'],
                    'style': 'info',
                    'icon': '💡'
                },
                {
                    'id': 'load_checkpoint',
                    'text': BUTTON_CONFIG['load_checkpoint']['text'],
                    'tooltip': BUTTON_CONFIG['load_checkpoint']['tooltip'],
                    'style': 'warning',
                    'icon': '📂'
                },
                {
                    'id': 'export_results',
                    'text': BUTTON_CONFIG['export_results']['text'],
                    'tooltip': BUTTON_CONFIG['export_results']['tooltip'],
                    'style': 'success',
                    'icon': '📊'
                }
            ],
            title="🎯 Evaluation Actions",
            show_save_reset=False
        )
        
        # Create operation container for progress and logging
        operation_container = create_operation_container(
            show_progress=True,
            show_dialog=True,
            show_logs=True,
            log_module_name="Evaluation",
            log_namespace_filter="evaluation"
        )
        
        # Create summary container for results
        summary_container = create_summary_container(
            theme="info",
            title="📊 Evaluation Results",
            icon="📊"
        )
        
        # Extract widget components from container objects
        header_widget = header_container.container if hasattr(header_container, 'container') else header_container
        action_widget = action_container.get('container') if isinstance(action_container, dict) else action_container
        operation_widget = operation_container.get('container') if isinstance(operation_container, dict) else operation_container
        summary_widget = summary_container.container if hasattr(summary_container, 'container') else summary_container
        
        # Create main container with all components
        main_container = create_main_container(
            components=[
                {'type': 'header', 'component': header_widget, 'order': 0},
                {'type': 'form', 'component': scenario_section, 'order': 1},
                {'type': 'form', 'component': model_section, 'order': 2}, 
                {'type': 'form', 'component': metrics_section, 'order': 3},
                {'type': 'form', 'component': execution_section, 'order': 4},
                {'type': 'action', 'component': action_widget, 'order': 5},
                {'type': 'operation', 'component': operation_widget, 'order': 6},
                {'type': 'custom', 'component': summary_widget, 'order': 7}
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
            'scenario_section': scenario_section,
            'model_section': model_section,
            'metrics_section': metrics_section,
            'execution_section': execution_section,
            'action_container': action_container,
            'operation_container': operation_container,
            'summary_container': summary_container,
            'components_count': 8
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

def _create_scenario_form_section(config: Dict[str, Any]) -> widgets.Widget:
    """Create scenario selection form section."""
    scenario_items = []
    
    # Title
    scenario_items.append(widgets.HTML("<h3>📊 Research Scenarios</h3>"))
    
    # Scenario checkboxes
    for scenario_key, scenario_info in RESEARCH_SCENARIOS.items():
        checkbox = widgets.Checkbox(
            value=scenario_info.get('enabled', True),
            description=f"{scenario_info['icon']} {scenario_info['name']}",
            style={'description_width': 'initial'},
            layout=widgets.Layout(margin='5px 0')
        )
        scenario_items.append(checkbox)
        
        # Add description
        desc_html = widgets.HTML(
            f"<p style='margin: 0 0 10px 25px; color: #666; font-size: 0.9em;'>"
            f"{scenario_info['description']}</p>"
        )
        scenario_items.append(desc_html)
    
    # Summary info
    summary_html = widgets.HTML(
        f"<div style='background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;'>"
        f"<strong>🎯 Evaluation Matrix:</strong> {len(RESEARCH_SCENARIOS)} scenarios × "
        f"{len(MODEL_COMBINATIONS)} models = {len(EVALUATION_MATRIX)} total tests"
        f"</div>"
    )
    scenario_items.append(summary_html)
    
    return widgets.VBox(scenario_items, layout=widgets.Layout(margin='10px 0'))

def _create_model_form_section(config: Dict[str, Any]) -> widgets.Widget:
    """Create model selection form section."""
    model_items = []
    
    # Title
    model_items.append(widgets.HTML("<h3>🤖 Model Configurations</h3>"))
    
    # Model checkboxes
    for model in MODEL_COMBINATIONS:
        checkbox = widgets.Checkbox(
            value=True,  # Default enabled
            description=f"🔸 {model['name']}",
            style={'description_width': 'initial'},
            layout=widgets.Layout(margin='5px 0')
        )
        model_items.append(checkbox)
        
        # Add model details
        details_html = widgets.HTML(
            f"<p style='margin: 0 0 10px 25px; color: #666; font-size: 0.9em;'>"
            f"Backbone: {model['backbone']} | Layer Mode: {model['layer_mode']}</p>"
        )
        model_items.append(details_html)
    
    # Auto-select best option
    auto_select = widgets.Checkbox(
        value=config.get('evaluation', {}).get('models', {}).get('auto_select_best', True),
        description="🎯 Auto-select best performing models",
        style={'description_width': 'initial'},
        layout=widgets.Layout(margin='10px 0')
    )
    model_items.append(auto_select)
    
    return widgets.VBox(model_items, layout=widgets.Layout(margin='10px 0'))

def _create_metrics_form_section(config: Dict[str, Any]) -> widgets.Widget:
    """Create metrics configuration form section."""
    metrics_items = []
    
    # Title
    metrics_items.append(widgets.HTML("<h3>📈 Evaluation Metrics</h3>"))
    
    # Metrics checkboxes (including accuracy)
    for metric_key, metric_info in EVALUATION_METRICS.items():
        checkbox = widgets.Checkbox(
            value=True,  # Default enabled
            description=f"{metric_info['icon']} {metric_info['name']}",
            style={'description_width': 'initial'},
            layout=widgets.Layout(margin='5px 0')
        )
        metrics_items.append(checkbox)
        
        # Add metric description
        desc_html = widgets.HTML(
            f"<p style='margin: 0 0 10px 25px; color: #666; font-size: 0.9em;'>"
            f"{metric_info['description']}</p>"
        )
        metrics_items.append(desc_html)
    
    return widgets.VBox(metrics_items, layout=widgets.Layout(margin='10px 0'))

def _create_execution_form_section(config: Dict[str, Any]) -> widgets.Widget:
    """Create execution options form section."""
    execution_items = []
    
    # Title
    execution_items.append(widgets.HTML("<h3>⚙️ Execution Options</h3>"))
    
    # Execution mode selection
    run_mode = widgets.RadioButtons(
        options=[
            ('All Scenarios (8 tests)', 'all_scenarios'),
            ('Position Variation Only (4 tests)', 'position_only'),
            ('Lighting Variation Only (4 tests)', 'lighting_only')
        ],
        value=config.get('evaluation', {}).get('execution', {}).get('run_mode', 'all_scenarios'),
        description='Run Mode:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(margin='10px 0')
    )
    execution_items.append(run_mode)
    
    # Additional options
    parallel_execution = widgets.Checkbox(
        value=config.get('evaluation', {}).get('execution', {}).get('parallel_execution', False),
        description="⚡ Enable parallel execution",
        style={'description_width': 'initial'},
        layout=widgets.Layout(margin='5px 0')
    )
    execution_items.append(parallel_execution)
    
    save_intermediate = widgets.Checkbox(
        value=config.get('evaluation', {}).get('execution', {}).get('save_intermediate_results', True),
        description="💾 Save intermediate results",
        style={'description_width': 'initial'},
        layout=widgets.Layout(margin='5px 0')
    )
    execution_items.append(save_intermediate)
    
    return widgets.VBox(execution_items, layout=widgets.Layout(margin='10px 0'))

def display_evaluation_ui(ui_components: Dict[str, Any]) -> None:
    """
    Display the evaluation UI with comprehensive error handling and logging.
    
    Args:
        ui_components: Dictionary containing all UI components with 'main_container' as the root
    """
    # Get logger first to ensure we can log any issues
    try:
        logger = get_module_logger("smartcash.ui.model.evaluation.components")
    except Exception as e:
        print(f"[WARNING] Failed to get logger: {e}")
        logger = None
    
    def log_info(msg):
        if logger:
            logger.info(msg)
        print(f"[INFO] {msg}")
        
    def log_error(msg, exc_info=False):
        if logger:
            logger.error(msg, exc_info=exc_info)
        print(f"[ERROR] {msg}")
    
    log_info("🖥️ ===== Starting Evaluation UI Display =====")
    
    try:
        # Check if we have components to display
        if not ui_components:
            error_msg = "No UI components provided to display"
            log_error(error_msg)
            display(HTML(f"<h3 style='color: red;'>❌ {error_msg}</h3>"))
            return
        
        # Log available component keys for debugging
        component_keys = list(ui_components.keys())
        log_info(f"📋 Available UI components: {component_keys}")
        
        # Get the main container
        main_container = ui_components.get('main_container')
        if main_container is None:
            error_msg = "Main container not found in UI components"
            log_error(error_msg)
            display(HTML(f"<h3 style='color: red;'>❌ {error_msg}</h3>"))
            return
        
        log_info("🎯 Found main container, checking type...")
        log_info(f"Main container type: {type(main_container).__name__}")
        log_info(f"Main container attributes: {dir(main_container) if hasattr(main_container, '__dir__') else 'N/A'}")
        
        # Clear any existing output first
        log_info("🧹 Clearing previous output...")
        clear_output(wait=True)
        
        # Display a test widget first to verify display is working
        try:
            log_info("🧪 Testing basic display functionality...")
            test_button = widgets.Button(description="Test Button")
            display(HTML("<h3>🔄 Display Test</h3>"))
            display(test_button)
            log_info("✅ Basic display test passed")
        except Exception as test_error:
            log_error(f"❌ Basic display test failed: {test_error}", exc_info=True)
        
        # Try different display methods in order of preference
        log_info("🔄 Attempting to display main UI...")
        
        try:
            # Method 1: If main_container is a widget with _ipython_display_
            if hasattr(main_container, '_ipython_display_'):
                log_info("🔹 Attempting to display using _ipython_display_ method")
                display(HTML("<h3>🔹 Using _ipython_display_ method</h3>"))
                display(main_container)
                log_info("✅ UI displayed successfully using _ipython_display_")
                return
            
            # Method 2: If main_container has a container attribute
            if hasattr(main_container, 'container'):
                log_info("🔹 Attempting to display using container attribute")
                display(HTML("<h3>🔹 Using container attribute</h3>"))
                display(main_container.container)
                log_info("✅ UI displayed successfully using container attribute")
                return
            
            # Method 3: If main_container has a show method
            if hasattr(main_container, 'show'):
                log_info("🔹 Attempting to display using show() method")
                display(HTML("<h3>🔹 Using show() method</h3>"))
                display(main_container.show())
                log_info("✅ UI displayed successfully using show() method")
                return
            
            # Method 4: Direct display as last resort
            log_info("🔹 Attempting direct display")
            display(HTML("<h3>🔹 Using direct display</h3>"))
            display(main_container)
            log_info("✅ UI displayed successfully using direct display")
            
        except Exception as display_error:
            log_error(f"❌ Failed to display UI: {str(display_error)}", exc_info=True)
            
            # Fallback: Show detailed error with component structure
            error_html = f"""
            <div style='border: 2px solid #ff4444; padding: 15px; border-radius: 5px; background: #ffeeee; margin: 10px 0;'>
                <h3 style='color: #cc0000; margin-top: 0;'>❌ Failed to Display Evaluation UI</h3>
                <p><strong>Error:</strong> {error}</p>
                <p><strong>Main Container Type:</strong> {container_type}</p>
                <p><strong>Available Components:</strong> {components}</p>
                <p>Please check the logs for more details.</p>
            </div>
            """.format(
                error=str(display_error),
                container_type=type(main_container).__name__,
                components=", ".join(component_keys) if component_keys else "None"
            )
            
            try:
                display(HTML(error_html))
                
                # Also try to display the main container directly as a fallback
                display(HTML("<h4>Attempting fallback display of main container:</h4>"))
                display(main_container)
                
            except Exception as fallback_error:
                log_error(f"❌ Fallback display also failed: {fallback_error}", exc_info=True)
                display(HTML(f"<div style='color:red;'>Fallback display failed: {fallback_error}</div>"))
    
    except Exception as e:
        log_error(f"❌ Fatal error in display_evaluation_ui: {str(e)}", exc_info=True)
        
        # Try to display at least some error message
        try:
            error_html = f"""
            <div style='border: 2px solid #ff0000; padding: 15px; border-radius: 5px; background: #ffdddd; margin: 10px 0;'>
                <h3 style='color: #cc0000; margin-top: 0;'>❌ Fatal Error Displaying UI</h3>
                <p><strong>Error:</strong> {error}</p>
                <p>Please check the logs for more details and report this issue.</p>
            </div>
            """.format(error=str(e))
            display(HTML(error_html))
        except:
            print(f"[FATAL] Could not display error message: {e}")
    
    finally:
        log_info("🏁 ===== Evaluation UI Display Completed =====")