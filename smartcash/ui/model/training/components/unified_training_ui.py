"""
File: smartcash/ui/model/training/components/unified_training_ui.py
Description: Simplified training UI that uses the unified training pipeline with proper container components.
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from smartcash.ui.logger import get_module_logger

# Standard container imports
from smartcash.ui.components import (
    create_header_container, 
    create_form_container, 
    create_action_container,
    create_operation_container, 
    create_summary_container, 
    create_main_container
)
from smartcash.ui.components.form_container import LayoutType
from smartcash.ui.core.decorators import handle_ui_errors
from .unified_training_form import create_unified_training_form
from .training_charts import create_dual_charts_layout
from .training_metrics import generate_metrics_table_html, get_initial_metrics_html


def create_charts_and_metrics_section(config: Dict[str, Any]) -> widgets.Widget:
    """Create collapsible charts and metrics section.
    
    Args:
        config: Configuration for charts and metrics
        
    Returns:
        Accordion widget containing charts and metrics
    """
    logger = get_module_logger("smartcash.ui.model.training.components")
    
    try:
        # Create chart configuration
        chart_config = config.get('charts', {
            'charts': {
                'loss_chart': {'update_frequency': 'epoch'},
                'map_chart': {'update_frequency': 'epoch'}
            },
            'monitoring': {'primary_metric': 'mAP@0.5'}
        })
        
        # Create dual charts
        charts = create_dual_charts_layout(chart_config, config.get('ui', {}))
        
        # Create charts container
        charts_container = widgets.HBox([
            charts['loss_chart'],
            charts['map_chart']
        ], layout=widgets.Layout(width='100%', padding='10px'))
        
        # Create metrics display
        metrics_display = widgets.HTML(
            value=get_initial_metrics_html(),
            layout=widgets.Layout(width='100%', padding='10px')
        )
        
        # Create accordion for charts and metrics
        accordion = widgets.Accordion(children=[
            charts_container,
            metrics_display
        ])
        
        accordion.set_title(0, "📈 Live Training Charts")
        accordion.set_title(1, "📊 Performance Metrics")
        accordion.selected_index = None  # Start collapsed
        
        # Add update methods to the accordion for external access
        accordion._charts = charts
        accordion._metrics_display = metrics_display
        
        def update_charts_data(data: Dict[str, Any]):
            """Update charts with training data."""
            try:
                charts['update_loss'](data)
                charts['update_map'](data)
            except Exception as e:
                logger.warning(f"Failed to update charts: {e}")
        
        def update_metrics_display_func(metrics: Dict[str, Any]):
            """Update metrics display with new data."""
            try:
                metrics_html = generate_metrics_table_html(metrics)
                metrics_display.value = metrics_html
            except Exception as e:
                logger.warning(f"Failed to update metrics display: {e}")
        
        def reset_charts_and_metrics():
            """Reset both charts and metrics to initial state."""
            try:
                charts['reset_charts']()
                metrics_display.value = get_initial_metrics_html()
            except Exception as e:
                logger.warning(f"Failed to reset charts and metrics: {e}")
        
        # Attach methods to accordion
        accordion.update_charts = update_charts_data
        accordion.update_metrics = update_metrics_display_func
        accordion.reset_all = reset_charts_and_metrics
        
        logger.debug("✅ Charts and metrics section created successfully")
        return accordion
        
    except Exception as e:
        logger.error(f"Failed to create charts and metrics section: {e}")
        # Return a simple placeholder on error
        return widgets.HTML(
            value="<div style='padding: 20px; text-align: center; color: #6c757d;'>Charts and metrics will be available during training</div>",
            layout=widgets.Layout(width='100%', padding='10px')
        )


@handle_ui_errors(error_component_title="Unified Training UI Creation Error")
def create_unified_training_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Create unified training UI using standard container components.
    
    Args:
        config: Training configuration
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing UI components
    """
    if config is None:
        config = {}
    
    # Handle additional kwargs if needed (currently unused but reserved for future extension)
    _ = kwargs  # Suppress unused warning
    
    logger = get_module_logger("smartcash.ui.model.training.components")
    
    # 1. Create Header Container
    header_container = create_header_container(
        title="🚀 Unified Training Pipeline",
        subtitle="Simplified training using the unified training pipeline",
        icon='🚀',
        theme='gradient',
        gradient_colors=['#667eea', '#764ba2']
    )
    
    # 2. Create Form Container with accordion
    form_widgets = create_unified_training_form(config)
    
    form_container = create_form_container(
        layout_type=LayoutType.COLUMN,
        container_padding="10px",
        gap="12px"
    )
    
    # Add the form accordion to the form container
    form_container['add_item'](form_widgets, "training_form")
    
    # 3. Create Action Container (only Start Training button)
    actions = [
        {
            'id': 'start_training',
            'text': '🚀 Start Training',
            'style': 'success',
            'tooltip': 'Start unified training pipeline',
            'order': 1
        }
    ]
    
    action_container = create_action_container(
        title="🎯 Training Actions",
        buttons=actions,
        show_save_reset=True  # Use standard save/reset buttons
    )
    
    # 4. Create Operation Container with triple progress tracker
    operation_container = create_operation_container(
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        progress_levels='triple',
        log_module_name="Training",
        log_height="200px",
        collapsible=True,
        collapsed=False
    )
    
    # 5. Create Charts and Metrics Section
    charts_and_metrics = create_charts_and_metrics_section(config)
    
    # 6. Create Summary Container
    summary_container = create_summary_container(
        theme='default',
        title='📈 Training Summary',
        icon='📊'
    )
    
    # Set initial content for summary container
    summary_container.set_content('''
        <div style="padding: 15px; background: #f8f9fa; border-radius: 8px;">
            <h4 style="color: #495057; margin: 0 0 10px 0;">📊 Training Summary</h4>
            <p style="margin: 5px 0; color: #6c757d;">
                Training results and final metrics will appear here after completion.
            </p>
            <p style="margin: 5px 0; color: #6c757d;">
                Charts and visualizations will be saved to <code>data/visualization/</code>
            </p>
        </div>
        ''')
    
    # 7. Create Main Container
    main_container = create_main_container(
        components=[
            {'component': header_container.container, 'type': 'header'},
            {'component': form_container['container'], 'type': 'form'},
            {'component': action_container['container'], 'type': 'action'},
            {'component': operation_container['container'], 'type': 'operation'},
            {'component': charts_and_metrics, 'type': 'charts'},
            {'component': summary_container.container, 'type': 'summary'}
        ]
    )
    
    # 8. Create UI components dictionary
    ui_components = {
        'main_container': main_container.container,
        'header_container': header_container,
        'form_container': form_container,
        'action_container': action_container,
        'operation_container': operation_container,
        'charts_and_metrics': charts_and_metrics,
        'summary_container': summary_container,
        
        # Individual action buttons for easy access
        'start_training': action_container['buttons'].get('start_training'),
        'save': action_container['buttons'].get('save'),
        'reset': action_container['buttons'].get('reset'),
        
        # Operation container methods
        'update_progress': operation_container.get('update_progress'),
        'log': operation_container.get('log'),
        'log_info': operation_container.get('info'),
        'log_success': operation_container.get('success'),
        'log_warning': operation_container.get('warning'),
        'log_error': operation_container.get('error'),
        'clear_logs': operation_container.get('clear_logs'),
        
        # Form access methods
        'form_widgets': getattr(form_widgets, '_widgets', {}),
        'get_form_values': getattr(form_widgets, 'get_form_values', lambda: {}),
        'update_form_from_config': getattr(form_widgets, 'update_from_config', lambda x: None),
        
        # Charts and metrics methods
        'update_charts': getattr(charts_and_metrics, 'update_charts', lambda x: None),
        'update_metrics': getattr(charts_and_metrics, 'update_metrics', lambda x: None),
        'reset_charts_and_metrics': getattr(charts_and_metrics, 'reset_all', lambda: None)
    }
    
    logger.debug("✅ Unified training UI created successfully with standard containers")
    return ui_components


def update_training_buttons_state(ui_components: Dict[str, Any], 
                                is_training: bool = False,
                                has_model: bool = True):
    """Update training button states based on current conditions.
    
    Args:
        ui_components: UI components dictionary
        is_training: Whether training is currently active
        has_model: Whether a model is available for training
    """
    try:
        start = ui_components.get('start_training')
        
        if start and hasattr(start, 'disabled'):
            start.disabled = is_training or not has_model
            if not has_model:
                start.tooltip = "No model available - configure backbone first"
            elif is_training:
                start.tooltip = "Training in progress"
            else:
                start.tooltip = "Start unified training pipeline"
                
    except Exception as e:
        logger = get_module_logger("smartcash.ui.model.training.components")
        logger.warning(f"Failed to update training button states: {e}")


def update_summary_display(ui_components: Dict[str, Any], result: Dict[str, Any]):
    """Update summary display with training results and final metrics.
    
    Args:
        ui_components: UI components dictionary
        result: Training result dictionary
    """
    try:
        summary_container = ui_components.get('summary_container')
        if not summary_container or not hasattr(summary_container, 'set_content'):
            return
        
        if result.get('success'):
            # Get training results
            training_result = result.get('final_training_result', {})
            best_metrics = training_result.get('best_metrics', {})
            
            # Update charts and metrics with final results
            update_charts = ui_components.get('update_charts')
            update_metrics = ui_components.get('update_metrics')
            
            if update_charts and best_metrics:
                # Update charts with final training data
                final_chart_data = {
                    'epoch': best_metrics.get('epoch', 0),
                    'train_loss': best_metrics.get('train_loss', 0.0),
                    'val_loss': best_metrics.get('val_loss', 0.0),
                    'mAP@0.5': best_metrics.get('val_map50', 0.0),
                    'mAP@0.75': best_metrics.get('val_map75', 0.0)
                }
                update_charts(final_chart_data)
            
            if update_metrics and best_metrics:
                # Update metrics display with final results
                update_metrics(best_metrics)
            
            # Get visualization results
            viz_result = result.get('visualization_result', {})
            session_id = viz_result.get('session_id', 'N/A')
            charts_count = viz_result.get('charts_count', 0)
            
            # Create success summary
            summary_html = f'''
            <div style="padding: 15px; background: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px;">
                <h4 style="color: #155724; margin: 0 0 15px 0;">🎉 Training Completed Successfully!</h4>
                
                <div style="margin-bottom: 15px;">
                    <h5 style="color: #155724; margin: 0 0 8px 0;">📊 Performance Metrics:</h5>
                    <p style="margin: 3px 0; color: #155724;"><strong>Best mAP@0.5:</strong> {best_metrics.get('val_map50', 0):.4f}</p>
                    <p style="margin: 3px 0; color: #155724;"><strong>Final Train Loss:</strong> {best_metrics.get('train_loss', 0):.4f}</p>
                    <p style="margin: 3px 0; color: #155724;"><strong>Final Val Loss:</strong> {best_metrics.get('val_loss', 0):.4f}</p>
                </div>
                
                <div style="margin-bottom: 15px;">
                    <h5 style="color: #155724; margin: 0 0 8px 0;">🎯 Layer Performance:</h5>
            '''
            
            # Add layer metrics
            for layer in ['layer_1', 'layer_2', 'layer_3']:
                acc = best_metrics.get(f'{layer}_accuracy', 0)
                f1 = best_metrics.get(f'{layer}_f1', 0)
                if acc > 0 or f1 > 0:
                    layer_name = layer.replace('_', ' ').title()
                    summary_html += f'<p style="margin: 3px 0; color: #155724;"><strong>{layer_name}:</strong> Acc={acc:.4f} F1={f1:.4f}</p>'
            
            # Add visualization info
            summary_html += f'''
                </div>
                
                <div>
                    <h5 style="color: #155724; margin: 0 0 8px 0;">📈 Visualizations:</h5>
                    <p style="margin: 3px 0; color: #155724;"><strong>Charts Generated:</strong> {charts_count}</p>
                    <p style="margin: 3px 0; color: #155724;"><strong>Session ID:</strong> {session_id}</p>
                    <p style="margin: 3px 0; color: #155724;"><strong>Location:</strong> <code>data/visualization/{session_id}/</code></p>
                </div>
            </div>
            '''
        else:
            # Create error summary
            error_msg = result.get('message', 'Unknown error')
            summary_html = f'''
            <div style="padding: 15px; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px;">
                <h4 style="color: #721c24; margin: 0 0 10px 0;">❌ Training Failed</h4>
                <p style="margin: 0; color: #721c24;"><strong>Error:</strong> {error_msg}</p>
            </div>
            '''
        
        # Update the summary container content
        summary_container.set_content(summary_html)
            
    except Exception as e:
        logger = get_module_logger("smartcash.ui.model.training.components")
        logger.warning(f"Failed to update summary display: {e}")


def update_training_progress(ui_components: Dict[str, Any], 
                           epoch_data: Dict[str, Any] = None,
                           metrics_data: Dict[str, Any] = None):
    """Update charts and metrics during training progress.
    
    Args:
        ui_components: UI components dictionary
        epoch_data: Data for updating charts (loss, mAP per epoch)
        metrics_data: Current metrics for display
    """
    try:
        logger = get_module_logger("smartcash.ui.model.training.components")
        
        # Update charts if epoch data is provided
        if epoch_data and ui_components.get('update_charts'):
            ui_components['update_charts'](epoch_data)
            logger.debug(f"📈 Updated charts for epoch {epoch_data.get('epoch', 'N/A')}")
        
        # Update metrics display if metrics data is provided
        if metrics_data and ui_components.get('update_metrics'):
            ui_components['update_metrics'](metrics_data)
            logger.debug(f"📊 Updated metrics display")
            
    except Exception as e:
        logger = get_module_logger("smartcash.ui.model.training.components")
        logger.warning(f"Failed to update training progress: {e}")


def reset_training_displays(ui_components: Dict[str, Any]):
    """Reset charts and metrics displays to initial state.
    
    Args:
        ui_components: UI components dictionary
    """
    try:
        logger = get_module_logger("smartcash.ui.model.training.components")
        
        # Reset charts and metrics
        reset_func = ui_components.get('reset_charts_and_metrics')
        if reset_func:
            reset_func()
            logger.debug("🔄 Reset charts and metrics displays")
            
    except Exception as e:
        logger = get_module_logger("smartcash.ui.model.training.components")
        logger.warning(f"Failed to reset training displays: {e}")