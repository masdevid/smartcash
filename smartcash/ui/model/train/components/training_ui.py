"""
File: smartcash/ui/model/train/components/training_ui.py
Training UI components with dual live charts following UIModule pattern.
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
from ..constants import UI_CONFIG
from .training_charts import create_dual_charts_layout as _create_dual_charts_layout
from .training_config_summary import create_config_summary as _create_configuration_summary
from .training_form import create_training_form as _create_training_form
from .training_metrics import (
    generate_metrics_table_html as _generate_metrics_table_html,
    get_initial_metrics_html as _get_initial_metrics_html,
    get_quality_indicator as _get_quality_indicator
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
        
        # 1. Create header container with title
        header_container = create_header_container(
            title=UI_CONFIG['title'].replace('🚀 ', ''),  # Remove emoji from title
            subtitle=UI_CONFIG['subtitle'],
            icon="🚀"  # Set icon separately
        )
        
        # 2. Create form container for training configuration
        form_container = _create_training_form(training_config, ui_config)
        
        # 3. Create action container with training operation buttons
        # Define action buttons configuration
        action_buttons = [
            # fmt: off
            ('start_training', 'Start Training', 'success', 'play', 'Start model training', False),
            ('stop_training', 'Stop Training', 'danger', 'stop', 'Stop current training', True),
            ('resume_training', 'Resume Training', 'warning', 'play', 'Resume training from checkpoint', True),
            ('validate_model', 'Validate Model', 'info', 'check', 'Run model validation', False),
            ('refresh_backbone_config', 'Refresh Config', 'secondary', 'sync', 
             'Refresh backbone configuration from backbone module', False),
            # fmt: on
        ]
        
        # Convert to list of dicts for compatibility with create_action_container
        action_buttons = [
            {
                'id': btn[0],
                'text': btn[1],
                'style': btn[2],
                'icon': btn[3],
                'tooltip': btn[4],
                'disabled': btn[5]
            }
            for btn in action_buttons
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
            log_namespace_filter="smartcash.ui.model.train",
            log_height="200px",
            log_entry_style='compact'  # Ensure consistent hover behavior
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