"""
File: smartcash/ui/model/evaluate/components/evaluation_ui.py
Description: UI components for evaluation module with scenario testing interface
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from IPython.display import display

from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.summary_container import create_summary_container

from ..constants import UI_CONFIG, SCENARIO_CONFIGS, MODEL_CONFIGS, METRIC_CONFIGS, AVAILABLE_METRICS


def create_evaluation_ui() -> Dict[str, Any]:
    """Create evaluation UI with scenario testing interface.
    
    Returns:
        Dictionary containing all UI components
    """
    # Create main containers
    ui_components = {}
    
    # 1. Header Container
    header_container = create_header_container(
        title=UI_CONFIG["title"],
        subtitle=UI_CONFIG["subtitle"],
        icon=UI_CONFIG["icon"]
    )
    ui_components["header_container"] = header_container
    
    # 2. Summary Container - Evaluation Overview
    ui_components["evaluation_summary"] = widgets.HTML(
        value=_create_initial_summary(),
        layout=widgets.Layout(width='100%', margin='10px 0')
    )
    # Create summary container with the initial content
    summary_container = create_summary_container(
        title="Evaluation Overview"
    )
    # Set the content using the correct method
    summary_container.content = ui_components["evaluation_summary"]
    ui_components["summary_container"] = summary_container
    
    # 3. Form Container - Scenario and Model Selection
    form_widgets = _create_evaluation_form()
    ui_components.update(form_widgets)
    ui_components.update(create_form_container(
        form_widgets=form_widgets,
        title="Evaluation Configuration"
    ))
    
    # 4. Action Container - Control Buttons
    action_widgets = _create_action_buttons()
    ui_components.update(action_widgets)
    
    # Convert widget dictionary to list of button configs
    button_configs = [
        {
            'button_id': btn_id,
            'text': btn.description,
            'style': btn.button_style if hasattr(btn, 'button_style') else '',
            'tooltip': btn.tooltip if hasattr(btn, 'tooltip') else ''
        }
        for btn_id, btn in action_widgets.items()
    ]
    
    # Create action container with proper alignment
    action_container = create_action_container(
        buttons=button_configs,
        alignment="left"  # Align buttons to the left
    )
    ui_components.update(action_container)
    
    # 5. Operation Container - Progress and Logs
    operation_components = create_operation_container(
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        log_module_name="EVALUATION"
    )
    ui_components.update(operation_components)
    
    # 6. Tabs Container - Scenario Details and Checkpoint Management
    tabs_components = _create_tabs_interface()
    ui_components.update(tabs_components)
    
    # 7. Footer Container - Save/Reset
    footer_widgets = _create_footer_buttons()
    ui_components.update(footer_widgets)
    ui_components.update(create_footer_container(
        footer_widgets=footer_widgets
    ))
    
    # 8. Main Container - Combine all components
    ui_components["main_container"] = _create_main_container(ui_components)
    
    return ui_components


def _create_initial_summary() -> str:
    """Create initial evaluation summary HTML."""
    return f"""
    <div style="background: linear-gradient(135deg, {UI_CONFIG['theme']['primary_color']}, {UI_CONFIG['theme']['secondary_color']}); padding: 15px; border-radius: 10px; color: white; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <div style="font-size: 24px; margin-right: 10px;">{UI_CONFIG['icon']}</div>
            <div>
                <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 2px;">{UI_CONFIG['title']}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">{UI_CONFIG['subtitle']}</div>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 15px;">
            <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Scenarios</div>
                <div style="font-size: 1.1rem; font-weight: 600;">2</div>
                <div style="font-size: 0.8rem; opacity: 0.7;">Position & Lighting</div>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Models</div>
                <div style="font-size: 1.1rem; font-weight: 600;">2</div>
                <div style="font-size: 0.8rem; opacity: 0.7;">CSPDarknet & EfficientNet</div>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Total Tests</div>
                <div style="font-size: 1.1rem; font-weight: 600;">4</div>
                <div style="font-size: 0.8rem; opacity: 0.7;">Est. 12-20 min</div>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Metrics</div>
                <div style="font-size: 1.1rem; font-weight: 600;">5</div>
                <div style="font-size: 0.8rem; opacity: 0.7;">mAP, ACC, PREC +2</div>
            </div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Status</div>
                <div style="font-size: 1.1rem; font-weight: 600;">⏸️ Ready</div>
                <div style="font-size: 0.8rem; opacity: 0.7;">Waiting to start</div>
            </div>
        </div>
    </div>
    """


def _create_evaluation_form() -> Dict[str, Any]:
    """Create evaluation configuration form widgets."""
    form_widgets = {}
    
    # Scenario Selection Section
    scenario_title = widgets.HTML(
        value="<h4 style='margin: 15px 0 10px 0; color: #495057;'>📐 Scenario Selection</h4>"
    )
    form_widgets["scenario_title"] = scenario_title
    
    # Position Variation Checkbox
    pos_config = SCENARIO_CONFIGS["position_variation"]
    form_widgets["position_variation_checkbox"] = widgets.Checkbox(
        value=True,
        description=f"{pos_config['icon']} {pos_config['name']}",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px', margin='5px 10px')
    )
    
    # Lighting Variation Checkbox
    light_config = SCENARIO_CONFIGS["lighting_variation"]
    form_widgets["lighting_variation_checkbox"] = widgets.Checkbox(
        value=True,
        description=f"{light_config['icon']} {light_config['name']}",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px', margin='5px 10px')
    )
    
    # Model Selection Section
    model_title = widgets.HTML(
        value="<h4 style='margin: 15px 0 10px 0; color: #495057;'>🤖 Model Selection</h4>"
    )
    form_widgets["model_title"] = model_title
    
    # CSPDarknet Checkbox
    csp_config = MODEL_CONFIGS["cspdarknet"]
    form_widgets["cspdarknet_checkbox"] = widgets.Checkbox(
        value=True,
        description=f"{csp_config['icon']} {csp_config['name']}",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px', margin='5px 10px')
    )
    
    # EfficientNet Checkbox
    eff_config = MODEL_CONFIGS["efficientnet_b4"]
    form_widgets["efficientnet_checkbox"] = widgets.Checkbox(
        value=True,
        description=f"{eff_config['icon']} {eff_config['name']}",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px', margin='5px 10px')
    )
    
    # Evaluation Settings Section
    settings_title = widgets.HTML(
        value="<h4 style='margin: 15px 0 10px 0; color: #495057;'>⚙️ Evaluation Settings</h4>"
    )
    form_widgets["settings_title"] = settings_title
    
    # Confidence Threshold Slider
    form_widgets["confidence_threshold_slider"] = widgets.FloatSlider(
        value=0.25,
        min=0.0,
        max=1.0,
        step=0.05,
        description='Confidence:',
        readout_format='.2f',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='400px', margin='5px 10px')
    )
    
    # IoU Threshold Slider
    form_widgets["iou_threshold_slider"] = widgets.FloatSlider(
        value=0.45,
        min=0.0,
        max=1.0,
        step=0.05,
        description='IoU:',
        readout_format='.2f',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='400px', margin='5px 10px')
    )
    
    # Number of Variations Slider
    form_widgets["num_variations_slider"] = widgets.FloatSlider(
        value=5.0,
        min=1.0,
        max=10.0,
        step=1.0,
        description='Variations:',
        readout_format='.0f',
        style={'description_width': '100px'},
        layout=widgets.Layout(width='400px', margin='5px 10px')
    )
    
    # Metrics Selection Section
    metrics_title = widgets.HTML(
        value="<h4 style='margin: 15px 0 10px 0; color: #495057;'>📊 Metrics Selection</h4>"
    )
    form_widgets["metrics_title"] = metrics_title
    
    # Create metric checkboxes in two columns
    metrics_left = []
    metrics_right = []
    
    for i, metric in enumerate(AVAILABLE_METRICS):
        metric_config = METRIC_CONFIGS[metric]
        
        checkbox = widgets.Checkbox(
            value=metric_config.get('default_enabled', False),
            description=f"{metric_config['icon']} {metric_config['name']}",
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px', margin='5px 10px'),
            tooltip=metric_config['description']
        )
        
        form_widgets[f"{metric}_metric_checkbox"] = checkbox
        
        # Distribute metrics into two columns
        if i % 2 == 0:
            metrics_left.append(checkbox)
        else:
            metrics_right.append(checkbox)
    
    # Create metric selection columns
    metrics_left_column = widgets.VBox(
        metrics_left,
        layout=widgets.Layout(width='50%')
    )
    metrics_right_column = widgets.VBox(
        metrics_right,
        layout=widgets.Layout(width='50%')
    )
    
    form_widgets["metrics_selection_container"] = widgets.HBox(
        [metrics_left_column, metrics_right_column],
        layout=widgets.Layout(width='100%', margin='5px 0')
    )
    
    return form_widgets


def _create_action_buttons() -> Dict[str, Any]:
    """Create action control buttons."""
    action_widgets = {}
    
    # Primary Action Buttons
    action_widgets["run_scenario_btn"] = widgets.Button(
        description="🧪 Run Single Scenario",
        button_style='primary',
        tooltip="Run evaluation for first selected scenario and model",
        layout=widgets.Layout(width='200px', margin='5px')
    )
    
    action_widgets["run_comprehensive_btn"] = widgets.Button(
        description="🚀 Run All Scenarios",
        button_style='success',
        tooltip="Run comprehensive evaluation across all selected scenarios and models",
        layout=widgets.Layout(width='200px', margin='5px')
    )
    
    # Secondary Action Buttons
    action_widgets["load_checkpoint_btn"] = widgets.Button(
        description="📂 Load Checkpoint",
        button_style='info',
        tooltip="Load model checkpoint for evaluation",
        layout=widgets.Layout(width='150px', margin='5px')
    )
    
    action_widgets["list_checkpoints_btn"] = widgets.Button(
        description="📁 List Checkpoints",
        button_style='',
        tooltip="List available model checkpoints",
        layout=widgets.Layout(width='150px', margin='5px')
    )
    
    action_widgets["stop_evaluation_btn"] = widgets.Button(
        description="🛑 Stop",
        button_style='danger',
        tooltip="Stop current evaluation",
        layout=widgets.Layout(width='100px', margin='5px')
    )
    
    return action_widgets


def _create_tabs_interface() -> Dict[str, Any]:
    """Create tabs interface for scenario details and checkpoint management."""
    tabs_widgets = {}
    
    # Scenario Details Tab
    scenario_content = _create_scenario_details()
    tabs_widgets["scenario_details_tab"] = scenario_content
    
    # Checkpoint Management Tab  
    checkpoint_content = _create_checkpoint_management()
    tabs_widgets["checkpoint_management_tab"] = checkpoint_content
    
    # Create tabs widget
    tabs_widgets["evaluation_tabs"] = widgets.Tab(
        children=[scenario_content, checkpoint_content],
        layout=widgets.Layout(width='100%', margin='10px 0')
    )
    tabs_widgets["evaluation_tabs"].set_title(0, "📊 Scenario Details")
    tabs_widgets["evaluation_tabs"].set_title(1, "💾 Checkpoint Management")
    
    return tabs_widgets


def _create_scenario_details() -> widgets.Widget:
    """Create scenario details content."""
    pos_info = SCENARIO_CONFIGS["position_variation"]
    light_info = SCENARIO_CONFIGS["lighting_variation"]
    
    content_html = f"""
    <div style="padding: 20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <!-- Position Variation -->
            <div style="border: 2px solid {pos_info['color']}; border-radius: 10px; padding: 15px;">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 24px; margin-right: 10px;">{pos_info['icon']}</span>
                    <h3 style="margin: 0; color: {pos_info['color']};">{pos_info['name']}</h3>
                </div>
                <p style="color: #6c757d; margin-bottom: 15px;">{pos_info['description']}</p>
                
                <h5 style="color: #495057; margin: 10px 0 5px 0;">Augmentations:</h5>
                <ul style="color: #6c757d; margin: 0; padding-left: 20px;">
                    <li><strong>Rotation:</strong> -30° to +30°</li>
                    <li><strong>Translation:</strong> ±20% of image size</li>
                    <li><strong>Scale:</strong> 0.8x to 1.2x</li>
                </ul>
                
                <div style="margin-top: 15px; padding: 10px; background: rgba{tuple(int(pos_info['color'][i:i+2], 16) for i in (1, 3, 5)) + (0.1,)}; border-radius: 5px;">
                    <strong style="color: {pos_info['color']};">Purpose:</strong> Test robustness against banknote position changes
                </div>
            </div>
            
            <!-- Lighting Variation -->
            <div style="border: 2px solid {light_info['color']}; border-radius: 10px; padding: 15px;">
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <span style="font-size: 24px; margin-right: 10px;">{light_info['icon']}</span>
                    <h3 style="margin: 0; color: {light_info['color']};">{light_info['name']}</h3>
                </div>
                <p style="color: #6c757d; margin-bottom: 15px;">{light_info['description']}</p>
                
                <h5 style="color: #495057; margin: 10px 0 5px 0;">Augmentations:</h5>
                <ul style="color: #6c757d; margin: 0; padding-left: 20px;">
                    <li><strong>Brightness:</strong> ±30% variation</li>
                    <li><strong>Contrast:</strong> 0.7x to 1.3x</li>
                    <li><strong>Gamma:</strong> 0.7 to 1.3</li>
                </ul>
                
                <div style="margin-top: 15px; padding: 10px; background: rgba(255, 193, 7, 0.1); border-radius: 5px;">
                    <strong style="color: {light_info['color']};">Purpose:</strong> Test robustness against lighting condition changes
                </div>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 10px;">
            <h4 style="color: #495057; margin: 0 0 15px 0;">📈 Available Metrics</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                {"".join([
                    f'''<div style="display: flex; align-items: center; padding: 10px; background: white; border-radius: 5px; border-left: 3px solid {config['color']};">
                        <span style="font-size: 18px; margin-right: 8px;">{config['icon']}</span>
                        <div>
                            <div style="font-weight: 600; color: {config['color']};">{config['name']}</div>
                            <div style="font-size: 0.85em; color: #6c757d;">{config['description']}</div>
                        </div>
                    </div>'''
                    for metric, config in METRIC_CONFIGS.items()
                ])}
            </div>
            <div style="margin-top: 15px; padding: 10px; background: #e7f3ff; border-radius: 5px; border-left: 4px solid #007bff;">
                <strong style="color: #007bff;">💡 Tip:</strong> Select the metrics you want to calculate using the checkboxes in the evaluation form above.
            </div>
        </div>
    </div>
    """
    
    return widgets.HTML(value=content_html)


def _create_checkpoint_management() -> widgets.Widget:
    """Create checkpoint management content."""
    content_html = """
    <div style="padding: 20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h4 style="color: #495057; margin: 0 0 10px 0;">💾 Available Checkpoints</h4>
            <p style="color: #6c757d; margin: 0;">Checkpoint management interface will be displayed here after implementation.</p>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div style="border: 1px solid #dee2e6; border-radius: 10px; padding: 15px;">
                <h5 style="color: #495057; margin: 0 0 10px 0;">🌙 CSPDarknet Checkpoints</h5>
                <div style="background: #fff; border: 1px solid #dee2e6; border-radius: 5px; padding: 10px; margin: 5px 0;">
                    <div style="font-weight: 600;">best.pt</div>
                    <div style="font-size: 0.9em; color: #6c757d;">mAP: 0.847 | Size: 125.3 MB</div>
                </div>
                <div style="background: #fff; border: 1px solid #dee2e6; border-radius: 5px; padding: 10px; margin: 5px 0;">
                    <div style="font-weight: 600;">latest.pt</div>
                    <div style="font-size: 0.9em; color: #6c757d;">mAP: 0.852 | Size: 125.3 MB</div>
                </div>
            </div>
            
            <div style="border: 1px solid #dee2e6; border-radius: 10px; padding: 15px;">
                <h5 style="color: #495057; margin: 0 0 10px 0;">⚡ EfficientNet-B4 Checkpoints</h5>
                <div style="background: #fff; border: 1px solid #dee2e6; border-radius: 5px; padding: 10px; margin: 5px 0;">
                    <div style="font-weight: 600;">best.pt</div>
                    <div style="font-size: 0.9em; color: #6c757d;">mAP: 0.891 | Size: 87.6 MB</div>
                </div>
                <div style="background: #fff; border: 1px solid #dee2e6; border-radius: 5px; padding: 10px; margin: 5px 0;">
                    <div style="font-weight: 600;">latest.pt</div>
                    <div style="font-size: 0.9em; color: #6c757d;">mAP: 0.885 | Size: 87.6 MB</div>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #e7f3ff; border-radius: 10px; border-left: 4px solid #007bff;">
            <strong style="color: #007bff;">💡 Note:</strong> Checkpoints are automatically selected based on highest mAP score. 
            You can manually load specific checkpoints using the "Load Checkpoint" button above.
        </div>
    </div>
    """
    
    return widgets.HTML(value=content_html)


def _create_footer_buttons() -> Dict[str, Any]:
    """Create footer control buttons."""
    footer_widgets = {}
    
    footer_widgets["save_config_btn"] = widgets.Button(
        description="💾 Save Config",
        button_style='success',
        tooltip="Save current evaluation configuration",
        layout=widgets.Layout(width='120px', margin='5px')
    )
    
    footer_widgets["reset_config_btn"] = widgets.Button(
        description="🔄 Reset",
        button_style='warning',
        tooltip="Reset to default configuration",
        layout=widgets.Layout(width='120px', margin='5px')
    )
    
    return footer_widgets


def _create_main_container(ui_components: Dict[str, Any]) -> widgets.Widget:
    """Create main container combining all UI components."""
    children = []
    
    # Add containers in order
    container_order = [
        'header_container',
        'summary_container', 
        'form_container',
        'action_container',
        'container',  # operation container
        'evaluation_tabs',
        'footer_container'
    ]
    
    for container_name in container_order:
        if container_name in ui_components:
            children.append(ui_components[container_name])
    
    main_container = widgets.VBox(
        children=children,
        layout=widgets.Layout(
            width='100%',
            padding='10px',
            border='1px solid #e0e0e0',
            border_radius='10px'
        )
    )
    
    return main_container