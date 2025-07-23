"""
File: smartcash/ui/model/training/components/training_ui.py
Training UI components with dual live charts following BaseUIModule pattern.
"""

# Standard library imports
from typing import Dict, Any, Optional

# Third-party imports
import ipywidgets as widgets

# SmartCash UI imports
from smartcash.ui.logger import get_module_logger
from smartcash.ui.components import (
    create_action_container,
    create_header_container,
    create_main_container,
    create_operation_container,
    create_summary_container
)

# Local application imports
from .training_charts import create_dual_charts_layout
from .training_config_summary import create_config_summary
from .training_form import create_training_form
from .training_metrics import (
    generate_metrics_table_html,
    get_initial_metrics_html,
    get_quality_indicator
)


# UI Configuration for training module
UI_CONFIG = {
    'title': 'SmartCash Model Training',
    'subtitle': 'Train object detection models with live monitoring and automatic backbone integration',
    'module_name': 'training',
    'container_width': 'auto',
    'form_width': '100%',
    'chart_height': '300px',
    'operation_height': '200px'
}


def create_training_ui(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create training UI components with proper container structure.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Dictionary containing all UI components
    """
    logger = get_module_logger("smartcash.ui.model.training.components")
    
    try:
        # Extract configurations
        training_config = config.get('training', {})
        ui_config = config.get('ui', {})
        
        # 1. Create header with title and subtitle
        header_container = create_header_container(
            title=UI_CONFIG['title'],
            subtitle=UI_CONFIG['subtitle'],
            icon='🚀',  # Rocket emoji for training
            show_environment=True,
            environment='local',
            config_path='training_config.yaml'
        )
        
        # 2. Create form container for training configuration
        form_container = create_training_form(training_config, ui_config)
        
        # 3. Create action container with training operation buttons
        action_buttons = [
            {
                'id': 'start_training',
                'text': '▶️ Start Training',
                'style': 'success',
                'icon': None,
                'tooltip': 'Start model training with backbone integration',
                'disabled': False
            },
            {
                'id': 'stop_training',
                'text': '⏹️ Stop Training',
                'style': 'danger',
                'icon': None,
                'tooltip': 'Stop current training process safely',
                'disabled': True
            },
            {
                'id': 'resume_training',
                'text': '⏯️ Resume Training',
                'style': 'warning',
                'icon': None,
                'tooltip': 'Resume training from checkpoint',
                'disabled': True
            },
            # validate_model button removed - overlaps with backbone module
            {
                'id': 'refresh_backbone_config',
                'text': '🔄 Refresh Config',
                'style': 'info',
                'icon': None,
                'tooltip': 'Refresh configuration from backbone module',
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
        buttons = action_container_result['buttons']
        
        # 4. Create dual chart layout for live monitoring
        charts_data = create_dual_charts_layout(training_config, ui_config)
        charts_container = widgets.HBox([
            charts_data['loss_chart'],
            charts_data['map_chart']
        ], layout=widgets.Layout(
            width='100%', 
            padding='8px 0',
            justify_content='space-around'
        ))
        
        # 5. Create metrics results summary panel
        metrics_summary = create_metrics_results_panel()
        
        # 6. Create operation container for progress tracking and logs with triple progress bars
        operation_container_result = create_operation_container(
            show_progress=True,
            show_dialog=True,
            show_logs=True,
            progress_levels='triple',  # Enable triple progress bars for granular tracking
            log_module_name=UI_CONFIG['module_name'],
            log_height="150px",
            log_entry_style='compact',
            collapsible=True,
            collapsed=False
        )
        operation_container = operation_container_result.get('container', operation_container_result)
        
        # 7. Create configuration summary container
        summary_container = create_config_summary(config)
        
        # 8. Create enhanced form with charts and metrics
        enhanced_form = widgets.VBox([
            form_container,
            widgets.HTML('<div style="margin: 10px 0; border-top: 1px solid #ddd;"></div>'),
            widgets.HTML('<h4>📊 Live Training Monitoring</h4>'),
            charts_container,
            widgets.HTML('<div style="margin: 10px 0; border-top: 1px solid #ddd;"></div>'),
            metrics_summary
        ], layout=widgets.Layout(width='100%'))
        
        # 9. Create main container with proper BaseUIModule structure
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
            'main_container': main_container,  # Use the actual widget
            'header_container': header_container,
            'form_container': form_container,
            'action_container': action_container_result,  # Store full result for button access
            # Action buttons (all buttons from the action container)
            'save': action_container_result.get('buttons', {}).get('save'),
            'reset': action_container_result.get('buttons', {}).get('reset'),
            'start_training': action_container_result.get('buttons', {}).get('start_training'),
            'stop_training': action_container_result.get('buttons', {}).get('stop_training'),
            'resume_training': action_container_result.get('buttons', {}).get('resume_training'),
            'refresh_backbone_config': action_container_result.get('buttons', {}).get('refresh_backbone_config'),
            'charts': {
                'loss_chart': charts_data['loss_chart'],
                'map_chart': charts_data['map_chart']
            },
            'loss_chart': charts_data['loss_chart'],  # Direct access for compatibility
            'map_chart': charts_data['map_chart'],    # Direct access for compatibility
            'charts_container': charts_container,
            'metrics_summary': metrics_summary,
            'operation_container': operation_container_result,  # Store the full result for operation access
            'summary_container': summary_container
        }
        
        logger.debug(f"✅ Created {len(ui_components)} training UI components with live charts")
        return ui_components
        
    except Exception as e:
        logger.error(f"Failed to create training UI: {e}")
        raise


def create_metrics_results_panel() -> widgets.Widget:
    """Create metrics results summary panel for training results."""
    logger = get_module_logger("smartcash.ui.model.training.components")
    
    try:
        # Create metrics table container
        metrics_html = widgets.HTML(value=get_initial_metrics_html())
        
        # Create collapsible container for metrics
        metrics_container = widgets.Accordion(children=[metrics_html])
        metrics_container.set_title(0, "📊 Training Results & Performance Metrics")
        metrics_container.selected_index = None  # Initially collapsed
        
        # Add method to update metrics
        def update_metrics(metrics_data: Dict[str, float]):
            """Update the metrics display with new training results."""
            try:
                html_content = generate_metrics_table_html(metrics_data)
                metrics_html.value = html_content
                # Expand accordion to show updated results
                metrics_container.selected_index = 0
                logger.debug("✅ Training metrics panel updated")
            except Exception as e:
                logger.error(f"Failed to update metrics: {e}")
        
        # Add method to show validation results
        def show_validation_results(validation_data: Dict[str, Any]):
            """Show validation results in the metrics panel."""
            try:
                performance_summary = validation_data.get('performance_summary', {})
                validation_metrics = validation_data.get('validation_metrics', {})
                
                # Create enhanced metrics display for validation
                enhanced_metrics = {
                    **validation_metrics,
                    'overall_grade': performance_summary.get('overall_grade', 'N/A'),
                    'validation_samples': validation_metrics.get('validation_samples', 0)
                }
                
                update_metrics(enhanced_metrics)
                logger.info("✅ Validation results displayed")
            except Exception as e:
                logger.error(f"Failed to show validation results: {e}")
        
        # Attach methods to container
        metrics_container.update_metrics = update_metrics
        metrics_container.show_validation_results = show_validation_results
        
        logger.debug("✅ Metrics results panel created with update methods")
        return metrics_container
        
    except Exception as e:
        logger.error(f"Failed to create metrics panel: {e}")
        raise


def update_training_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """
    Update training UI components from configuration.
    
    Args:
        ui_components: UI components dictionary
        config: Configuration dictionary
        
    Returns:
        True if update successful, False otherwise
    """
    logger = get_module_logger("smartcash.ui.model.training.components")
    
    try:
        if not ui_components or not config:
            logger.warning("Missing UI components or config for update")
            return False
        
        training_config = config.get('training', {})
        # model_selection = config.get('model_selection', {})  # Not needed with enhanced button management
        
        # Update form container if available
        form_container = ui_components.get('form_container')
        if form_container and hasattr(form_container, 'update_from_config'):
            form_container.update_from_config(config)
        
        # Update summary container with new configuration
        summary_container = ui_components.get('summary_container')
        if summary_container:
            try:
                updated_summary = create_config_summary(config)
                if hasattr(summary_container, 'children') and len(summary_container.children) > 0:
                    summary_container.children = updated_summary.children
            except Exception as e:
                logger.warning(f"Failed to update summary container: {e}")
        
        # Note: Button state management now handled by enhanced button mixin
        # TrainingUIModule._update_training_button_states() handles all button states
        # based on training state, model availability, and checkpoint status.
        # This provides consistent dependency-based button management across all modules.
        
        logger.debug("✅ Training UI updated from configuration (enhanced button management)")
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
    logger = get_module_logger("smartcash.ui.model.training.components")
    
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
    logger = get_module_logger("smartcash.ui.model.training.components")
    
    try:
        metrics_summary = ui_components.get('metrics_summary')
        if metrics_summary and hasattr(metrics_summary, 'update_metrics'):
            metrics_summary.update_metrics(metrics)
            logger.info("✅ Training metrics display updated")
            return True
        
        logger.warning("Metrics summary container not available")
        return False
        
    except Exception as e:
        logger.error(f"Failed to update metrics display: {e}")
        return False


def update_chart_data(ui_components: Dict[str, Any], chart_type: str, data: Dict[str, Any]) -> bool:
    """
    Update chart data for live monitoring.
    
    Args:
        ui_components: UI components dictionary
        chart_type: Type of chart ('loss_chart' or 'map_chart')
        data: Chart data to update
        
    Returns:
        True if update successful, False otherwise
    """
    logger = get_module_logger("smartcash.ui.model.training.components")
    
    try:
        chart_widget = ui_components.get(chart_type)
        if chart_widget and hasattr(chart_widget, 'add_data'):
            chart_widget.add_data(data)
            logger.debug(f"✅ {chart_type} updated with new data")
            return True
        
        logger.warning(f"Chart widget {chart_type} not available or doesn't support add_data")
        return False
        
    except Exception as e:
        logger.error(f"Failed to update chart {chart_type}: {e}")
        return False


def show_validation_results(ui_components: Dict[str, Any], validation_data: Dict[str, Any]) -> bool:
    """
    Show validation results in the UI.
    
    Args:
        ui_components: UI components dictionary
        validation_data: Validation results data
        
    Returns:
        True if successful, False otherwise
    """
    logger = get_module_logger("smartcash.ui.model.training.components")
    
    try:
        metrics_summary = ui_components.get('metrics_summary')
        if metrics_summary and hasattr(metrics_summary, 'show_validation_results'):
            metrics_summary.show_validation_results(validation_data)
            logger.info("✅ Validation results displayed in UI")
            return True
        
        logger.warning("Metrics summary not available for validation results")
        return False
        
    except Exception as e:
        logger.error(f"Failed to show validation results: {e}")
        return False