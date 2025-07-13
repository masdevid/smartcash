"""
File: smartcash/ui/model/train/components/training_ui.py
Training UI components with dual live charts following UIModule pattern.
"""

from typing import Dict, Any, Optional, List
import ipywidgets as widgets
from smartcash.ui.logger import get_module_logger
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.summary_container import create_summary_container
from ..constants import (
    UI_CONFIG, BUTTON_CONFIG, CHART_CONFIG, 
    LayerMode, OptimizationType, generate_model_name
)
from ..configs.train_defaults import (
    get_layer_mode_configs, get_optimization_types, get_chart_configurations
)


def create_training_ui(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create training UI components with proper container structure.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Dictionary containing all UI components
    """
    logger = get_module_logger("smartcash.ui.model.train.components")
    logger.debug("Creating training UI components with proper container structure...")
    
    try:
        # Extract configurations
        training_config = config.get('training', {})
        ui_config = config.get('ui', {})
        
        # 1. Create header container with title and status panel
        header_container = create_header_container(
            title=UI_CONFIG['title'],
            subtitle=UI_CONFIG['subtitle'],
            icon="🚀",
            status_message="Ready for training",
            status_type="info",
            show_status_panel=True
        )
        
        # 2. Create form container for training configuration
        form_container = _create_training_form(training_config, ui_config)
        
        # 3. Create action container with training operation buttons
        action_buttons = [
            {
                'id': 'start_training',
                'text': 'Start Training',
                'style': 'success',
                'icon': 'play',
                'tooltip': 'Start model training',
                'disabled': False
            },
            {
                'id': 'stop_training', 
                'text': 'Stop Training',
                'style': 'danger',
                'icon': 'stop',
                'tooltip': 'Stop current training',
                'disabled': True
            },
            {
                'id': 'resume_training',
                'text': 'Resume Training',
                'style': 'warning',
                'icon': 'play',
                'tooltip': 'Resume training from checkpoint',
                'disabled': True
            },
            {
                'id': 'validate_model',
                'text': 'Validate Model',
                'style': 'info',
                'icon': 'check',
                'tooltip': 'Run model validation',
                'disabled': False
            }
        ]
        
        action_container_result = create_action_container(
            buttons=action_buttons,
            title="Training Operations",
            show_save_reset=True,
            container_margin="15px 0"
        )
        action_container = action_container_result['container']
        
        # 4. Create dual chart layout
        charts_data = _create_dual_charts_layout(training_config, ui_config)
        charts_container = widgets.HBox([
            charts_data['loss_chart'],
            charts_data['map_chart']
        ], layout=widgets.Layout(width='100%'))
        
        # 5. Create metrics results summary panel
        metrics_summary = _create_metrics_results_panel()
        
        # 6. Create operation container for progress tracking and logs
        operation_container_result = create_operation_container(
            show_progress=True,
            show_logs=True,
            log_module_name="Training",
            log_namespace_filter="smartcash.ui.model.train"
        )
        operation_container = operation_container_result.get('container', operation_container_result)
        
        # 7. Create configuration summary container
        summary_container = _create_configuration_summary(config)
        
        # 8. Include charts and metrics in form container
        enhanced_form = widgets.VBox([
            form_container,
            charts_container,
            metrics_summary
        ], layout=widgets.Layout(width='100%'))
        
        # 9. Create main container with proper structure
        main_container_result = create_main_container(
            header_container=header_container.container,
            form_container=enhanced_form,
            action_container=action_container,
            operation_container=operation_container,
            footer_container=summary_container
        )
        main_container = main_container_result.container
        
        # Prepare UI components dictionary
        ui_components = {
            'main_container': main_container,
            'header_container': header_container,
            'form_container': form_container,
            'action_container': action_container_result,  # Store full result for button access
            'loss_chart': charts_data['loss_chart'],
            'map_chart': charts_data['map_chart'],
            'charts_container': charts_container,
            'metrics_summary': metrics_summary,
            'operation_container': operation_container_result,  # Store the full result for operation access
            'summary_container': summary_container
        }
        
        logger.debug(f"✅ Created {len(ui_components)} training UI components")
        return ui_components
        
    except Exception as e:
        logger.error(f"Failed to create training UI: {e}")
        raise

def _create_training_form(training_config: Dict[str, Any], ui_config: Dict[str, Any]) -> widgets.Widget:
    """Create training configuration form."""
    logger = get_module_logger("smartcash.ui.model.train.components")
    
    try:
        layer_configs = get_layer_mode_configs()
        optimization_types = get_optimization_types()
        
        # Layer mode selection
        layer_mode_dropdown = widgets.Dropdown(
            description="Training Mode:",
            options=[(config['display_name'], key) for key, config in layer_configs.items()],
            value=training_config.get('layer_mode', 'single'),
            style={'description_width': '150px'},
            layout=widgets.Layout(width='100%')
        )
        
        # Basic training parameters
        epochs_input = widgets.IntSlider(
            description="Epochs:",
            value=training_config.get('epochs', 100),
            min=1,
            max=1000,
            step=10,
            style={'description_width': '150px'},
            layout=widgets.Layout(width='100%')
        )
        
        batch_size_input = widgets.IntSlider(
            description="Batch Size:",
            value=training_config.get('batch_size', 16),
            min=1,
            max=256,
            step=2,
            style={'description_width': '150px'},
            layout=widgets.Layout(width='100%')
        )
        
        learning_rate_input = widgets.FloatLogSlider(
            description="Learning Rate:",
            value=training_config.get('learning_rate', 0.001),
            base=10,
            min=-6,
            max=0,
            step=0.1,
            style={'description_width': '150px'},
            layout=widgets.Layout(width='100%')
        )
        
        # Optimization type
        optimization_dropdown = widgets.Dropdown(
            description="Optimization:",
            options=[(config['display_name'], key) for key, config in optimization_types.items()],
            value=training_config.get('optimization_type', 'default'),
            style={'description_width': '150px'},
            layout=widgets.Layout(width='100%')
        )
        
        # Advanced options (simplified)
        early_stopping_config = training_config.get('early_stopping', {})
        mixed_precision_checkbox = widgets.Checkbox(
            description="Mixed Precision",
            value=training_config.get('mixed_precision', True),
            style={'description_width': '150px'}
        )
        
        early_stopping_checkbox = widgets.Checkbox(
            description="Early Stopping",
            value=early_stopping_config.get('enabled', True),
            style={'description_width': '150px'}
        )
        
        # Create form container using VBox
        form_widgets = [
            widgets.HTML("<h4>🚀 Training Configuration</h4>"),
            layer_mode_dropdown,
            epochs_input,
            batch_size_input,
            learning_rate_input,
            optimization_dropdown,
            widgets.HTML("<h5>Advanced Options</h5>"),
            widgets.HBox([mixed_precision_checkbox, early_stopping_checkbox])
        ]
        
        form_container = widgets.VBox(
            form_widgets,
            layout=widgets.Layout(
                border='1px solid #ddd',
                padding='15px',
                margin='10px 0',
                border_radius='5px'
            )
        )
        
        # Store references for value retrieval
        form_container._layer_mode_dropdown = layer_mode_dropdown
        form_container._epochs_input = epochs_input
        form_container._batch_size_input = batch_size_input
        form_container._learning_rate_input = learning_rate_input
        form_container._optimization_dropdown = optimization_dropdown
        form_container._mixed_precision_checkbox = mixed_precision_checkbox
        form_container._early_stopping_checkbox = early_stopping_checkbox
        
        # Add method to get form values
        def get_form_values():
            values = {
                'layer_mode': layer_mode_dropdown.value,
                'epochs': int(epochs_input.value),
                'batch_size': int(batch_size_input.value),
                'learning_rate': float(learning_rate_input.value),
                'optimization_type': optimization_dropdown.value,
                'mixed_precision': mixed_precision_checkbox.value,
                'early_stopping_enabled': early_stopping_checkbox.value
            }
            return values
        
        form_container.get_form_values = get_form_values
        
        logger.debug("✅ Training form created successfully")
        return form_container
        
    except Exception as e:
        logger.error(f"Failed to create training form: {e}")
        raise


def _create_dual_charts_layout(chart_config: Dict[str, Any], ui_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create dual live charts for loss and mAP monitoring."""
    logger = get_module_logger("smartcash.ui.model.train.components")
    
    try:
        # Create placeholder charts using simple HTML widgets
        loss_chart = widgets.HTML(
            value="""
            <div style="border: 2px solid #ff6b6b; padding: 20px; margin: 10px; border-radius: 8px; text-align: center;">
                <h4 style="color: #ff6b6b; margin-top: 0;">📈 Training Loss Chart</h4>
                <p>Real-time training and validation loss will be displayed here</p>
                <div style="background: #fff5f5; padding: 10px; border-radius: 4px; margin-top: 10px;">
                    <small>Metrics: Train Loss, Validation Loss</small>
                </div>
            </div>
            """,
            layout=widgets.Layout(width='100%', height='200px')
        )
        
        map_chart = widgets.HTML(
            value="""
            <div style="border: 2px solid #4ecdc4; padding: 20px; margin: 10px; border-radius: 8px; text-align: center;">
                <h4 style="color: #4ecdc4; margin-top: 0;">📊 mAP Performance Chart</h4>
                <p>Real-time mAP performance metrics will be displayed here</p>
                <div style="background: #f0fdfc; padding: 10px; border-radius: 4px; margin-top: 10px;">
                    <small>Metrics: mAP@0.5, mAP@0.75</small>
                </div>
            </div>
            """,
            layout=widgets.Layout(width='100%', height='200px')
        )
        
        # Add update methods to charts
        def update_loss_data(data):
            # Placeholder for live update functionality
            pass
        
        def update_map_data(data):
            # Placeholder for live update functionality  
            pass
            
        loss_chart.add_data = update_loss_data
        map_chart.add_data = update_map_data
        
        charts = {
            'loss_chart': loss_chart,
            'map_chart': map_chart
        }
        
        logger.debug("✅ Dual live charts created successfully")
        return charts
        
    except Exception as e:
        logger.error(f"Failed to create dual charts: {e}")
        raise


def _create_metrics_results_panel() -> widgets.Widget:
    """Create metrics results summary panel for final training results."""
    logger = get_module_logger("smartcash.ui.model.train.components")
    
    try:
        # Create metrics table container
        metrics_html = widgets.HTML(value=_get_initial_metrics_html())
        
        # Create collapsible container for metrics
        metrics_container = widgets.Accordion(children=[metrics_html])
        metrics_container.set_title(0, "📊 Training Results Metrics")
        metrics_container.selected_index = None  # Initially collapsed
        
        # Add method to update metrics
        def update_metrics(metrics_data: Dict[str, float]):
            """Update the metrics display with new training results."""
            try:
                html_content = _generate_metrics_table_html(metrics_data)
                metrics_html.value = html_content
                logger.debug("✅ Metrics panel updated")
            except Exception as e:
                logger.error(f"Failed to update metrics: {e}")
        
        metrics_container.update_metrics = update_metrics
        
        logger.debug("✅ Metrics results panel created")
        return metrics_container
        
    except Exception as e:
        logger.error(f"Failed to create metrics panel: {e}")
        raise


def _get_initial_metrics_html() -> str:
    """Get initial HTML for metrics panel when no training has completed."""
    return """
    <div style="padding: 20px; text-align: center; color: #666;">
        <h4 style="margin-top: 0;">🎯 Training Metrics</h4>
        <p>Training metrics will appear here after completion</p>
        <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px;">
            <small>Expected metrics: mAP, Accuracy, Precision, Recall, F1-Score</small>
        </div>
    </div>
    """


def _generate_metrics_table_html(metrics: Dict[str, float]) -> str:
    """Generate HTML table for training metrics results."""
    return f"""
    <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
        <h4 style="margin-top: 0; color: #495057; text-align: center;">🏆 Final Training Results</h4>
        
        <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
            <thead>
                <tr style="background: #007bff; color: white;">
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Metric</th>
                    <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Value</th>
                    <th style="padding: 12px; text-align: center; border: 1px solid #ddd;">Quality</th>
                </tr>
            </thead>
            <tbody>
                <tr style="background: white;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">📊 mAP@0.5</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #007bff;">
                        {metrics.get('val_map50', 0.0):.3f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {_get_quality_indicator(metrics.get('val_map50', 0.0), 'map')}
                    </td>
                </tr>
                <tr style="background: #f8f9fa;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">📈 mAP@0.75</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #007bff;">
                        {metrics.get('val_map75', 0.0):.3f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {_get_quality_indicator(metrics.get('val_map75', 0.0), 'map')}
                    </td>
                </tr>
                <tr style="background: white;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">🎯 Accuracy</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #28a745;">
                        {metrics.get('accuracy', 0.0):.3f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {_get_quality_indicator(metrics.get('accuracy', 0.0), 'accuracy')}
                    </td>
                </tr>
                <tr style="background: #f8f9fa;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">🔍 Precision</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #17a2b8;">
                        {metrics.get('precision', 0.0):.3f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {_get_quality_indicator(metrics.get('precision', 0.0), 'precision')}
                    </td>
                </tr>
                <tr style="background: white;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">📋 Recall</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #fd7e14;">
                        {metrics.get('recall', 0.0):.3f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {_get_quality_indicator(metrics.get('recall', 0.0), 'recall')}
                    </td>
                </tr>
                <tr style="background: #f8f9fa;">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: 500;">⚖️ F1-Score</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center; font-weight: 600; color: #6f42c1;">
                        {metrics.get('f1_score', 0.0):.3f}
                    </td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: center;">
                        {_get_quality_indicator(metrics.get('f1_score', 0.0), 'f1')}
                    </td>
                </tr>
            </tbody>
        </table>
        
        <div style="margin-top: 15px; padding: 10px; background: #e3f2fd; border-radius: 4px;">
            <small style="color: #1976d2;">
                <strong>💡 Quality Indicators:</strong> 
                🟢 Excellent (>0.8) | 🟡 Good (0.6-0.8) | 🟠 Fair (0.4-0.6) | 🔴 Poor (<0.4)
            </small>
        </div>
    </div>
    """


def _get_quality_indicator(value: float, metric_type: str) -> str:
    """Get quality indicator emoji based on metric value and type."""
    if value >= 0.8:
        return "🟢 Excellent"
    elif value >= 0.6:
        return "🟡 Good"
    elif value >= 0.4:
        return "🟠 Fair"
    else:
        return "🔴 Poor"


def _create_configuration_summary(config: Dict[str, Any]) -> widgets.Widget:
    """Create configuration summary panel."""
    logger = get_module_logger("smartcash.ui.model.train.components")
    
    try:
        training_config = config.get('training', {})
        backbone_integration = config.get('backbone_integration', {})
        model_storage = config.get('model_storage', {})
        
        # Generate model name preview
        backbone_type = backbone_integration.get('backbone_type', 'efficientnet_b4')
        layer_mode = training_config.get('layer_mode', 'single')
        optimization_type = training_config.get('optimization_type', 'default')
        model_name = generate_model_name(backbone_type, layer_mode, optimization_type)
        
        # Create summary content
        summary_html = f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;">
            <h4 style="margin-top: 0; color: #495057;">🚀 Training Configuration Summary</h4>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                <div>
                    <h5 style="color: #6c757d; margin-bottom: 8px;">📋 Training Settings</h5>
                    <p><strong>Mode:</strong> {training_config.get('layer_mode', 'single').title()}</p>
                    <p><strong>Epochs:</strong> {training_config.get('epochs', 100)}</p>
                    <p><strong>Batch Size:</strong> {training_config.get('batch_size', 16)}</p>
                    <p><strong>Learning Rate:</strong> {training_config.get('learning_rate', 0.001)}</p>
                    <p><strong>Optimization:</strong> {training_config.get('optimization_type', 'default').title()}</p>
                </div>
                
                <div>
                    <h5 style="color: #6c757d; margin-bottom: 8px;">🧬 Model Information</h5>
                    <p><strong>Backbone:</strong> {backbone_type}</p>
                    <p><strong>Model Name:</strong> <code>{model_name}</code></p>
                    <p><strong>Mixed Precision:</strong> {'✅' if training_config.get('mixed_precision', True) else '❌'}</p>
                    <p><strong>Early Stopping:</strong> {'✅' if training_config.get('early_stopping', {}).get('enabled', True) else '❌'}</p>
                </div>
            </div>
            
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #dee2e6;">
                <h5 style="color: #6c757d; margin-bottom: 8px;">📊 Monitoring Features</h5>
                <div style="display: flex; gap: 20px;">
                    <span>📈 Live Loss Chart</span>
                    <span>📊 Live mAP Chart</span>
                    <span>🔄 Real-time Progress</span>
                    <span>💾 Auto-save Best Model</span>
                    <span>🏆 Final Metrics Table</span>
                </div>
            </div>
            
            <div style="margin-top: 15px; padding: 10px; background: #e3f2fd; border-radius: 4px;">
                <small style="color: #1976d2;">
                    <strong>📝 Note:</strong> Training will continue from backbone configuration. 
                    Best model will be saved as <code>{model_name}</code> and will overwrite any existing model with the same name.
                </small>
            </div>
        </div>
        """
        
        summary_container = widgets.HTML(
            value=summary_html,
            layout=widgets.Layout(width='100%', margin='10px 0')
        )
        
        logger.debug("✅ Configuration summary created successfully")
        return summary_container
        
    except Exception as e:
        logger.error(f"Failed to create configuration summary: {e}")
        # Return simple fallback
        return widgets.HTML("<p>Configuration summary unavailable</p>")


def update_training_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """
    Update training UI components from configuration.
    
    Args:
        ui_components: UI components dictionary
        config: Configuration dictionary
        
    Returns:
        True if update successful, False otherwise
    """
    logger = get_module_logger("smartcash.ui.model.train.components")
    
    try:
        if not ui_components or not config:
            logger.warning("Missing UI components or config for update")
            return False
        
        training_config = config.get('training', {})
        
        # Update form container if available
        form_container = ui_components.get('form_container')
        if form_container:
            # Update form values
            if hasattr(form_container, '_layer_mode_dropdown'):
                form_container._layer_mode_dropdown.value = training_config.get('layer_mode', 'single')
            
            if hasattr(form_container, '_epochs_input'):
                form_container._epochs_input.value = training_config.get('epochs', 100)
            
            if hasattr(form_container, '_batch_size_input'):
                form_container._batch_size_input.value = training_config.get('batch_size', 16)
            
            if hasattr(form_container, '_learning_rate_input'):
                form_container._learning_rate_input.value = training_config.get('learning_rate', 0.001)
            
            if hasattr(form_container, '_optimization_dropdown'):
                form_container._optimization_dropdown.value = training_config.get('optimization_type', 'default')
            
            if hasattr(form_container, '_mixed_precision_checkbox'):
                form_container._mixed_precision_checkbox.value = training_config.get('mixed_precision', True)
                
            if hasattr(form_container, '_early_stopping_checkbox'):
                early_stopping_config = training_config.get('early_stopping', {})
                form_container._early_stopping_checkbox.value = early_stopping_config.get('enabled', True)
        
        # Update summary container
        summary_container = ui_components.get('summary_container')
        if summary_container:
            updated_summary = _create_configuration_summary(config)
            # Replace summary content
            if hasattr(summary_container, 'children') and len(summary_container.children) > 0:
                summary_container.children = updated_summary.children
        
        logger.debug("✅ Training UI updated from configuration")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update training UI: {e}")
        return False


def get_training_form_values(ui_components: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get current form values from training UI.
    
    Args:
        ui_components: UI components dictionary
        
    Returns:
        Form values dictionary or None if failed
    """
    logger = get_module_logger("smartcash.ui.model.train.components")
    
    try:
        form_container = ui_components.get('form_container')
        if form_container and hasattr(form_container, 'get_form_values'):
            return form_container.get_form_values()
        
        logger.warning("Form container not available for value retrieval")
        return None
        
    except Exception as e:
        logger.error(f"Failed to get training form values: {e}")
        return None


def update_metrics_display(ui_components: Dict[str, Any], metrics: Dict[str, float]) -> bool:
    """
    Update the metrics results panel with new training results.
    
    Args:
        ui_components: UI components dictionary
        metrics: Training metrics dictionary
        
    Returns:
        True if update successful, False otherwise
    """
    logger = get_module_logger("smartcash.ui.model.train.components")
    
    try:
        metrics_summary = ui_components.get('metrics_summary')
        if metrics_summary and hasattr(metrics_summary, 'update_metrics'):
            metrics_summary.update_metrics(metrics)
            # Expand the accordion to show results
            metrics_summary.selected_index = 0
            logger.info("✅ Training metrics display updated")
            return True
        
        logger.warning("Metrics summary container not available")
        return False
        
    except Exception as e:
        logger.error(f"Failed to update metrics display: {e}")
        return False