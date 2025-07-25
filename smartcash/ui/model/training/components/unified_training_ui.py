"""
File: smartcash/ui/model/training/components/unified_training_ui.py
Description: Simplified training UI that uses the unified training pipeline.
"""

import ipywidgets as widgets
from typing import Dict, Any
from smartcash.ui.logger import get_module_logger
from smartcash.ui.components.operation_container import OperationContainer
from .unified_training_form import create_unified_training_form


def create_unified_training_ui(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create simplified training UI that uses the unified training pipeline.
    
    Args:
        config: Training configuration
        
    Returns:
        Dictionary containing UI components
    """
    logger = get_module_logger("smartcash.ui.model.training.components")
    
    try:
        # Header section
        header_html = widgets.HTML(
            value='''
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; border-radius: 10px; margin-bottom: 20px;">
                <h2 style="margin: 0; font-weight: bold;">🚀 Unified Training Pipeline</h2>
                <p style="margin: 10px 0 0 0; opacity: 0.9;">Simplified training using the unified training pipeline</p>
            </div>
            ''',
            layout=widgets.Layout(width='100%')
        )
        
        # Form section
        form_container = create_unified_training_form(config)
        form_wrapper = widgets.VBox([
            widgets.HTML("<h3 style='color: #007bff; margin: 20px 0 15px 0;'>⚙️ Training Configuration</h3>"),
            form_container
        ], layout=widgets.Layout(width='100%', padding='10px'))
        
        # Action buttons section
        start_button = widgets.Button(
            description="🚀 Start Training",
            button_style='success',
            layout=widgets.Layout(width='180px', height='40px'),
            tooltip="Start unified training pipeline"
        )
        
        stop_button = widgets.Button(
            description="⏹️ Stop Training",
            button_style='danger',
            layout=widgets.Layout(width='180px', height='40px'),
            tooltip="Stop training (if supported)",
            disabled=True
        )
        
        save_button = widgets.Button(
            description="💾 Save Config",
            button_style='info',
            layout=widgets.Layout(width='180px', height='40px'),
            tooltip="Save current configuration"
        )
        
        reset_button = widgets.Button(
            description="🔄 Reset Config",
            button_style='warning',
            layout=widgets.Layout(width='180px', height='40px'),
            tooltip="Reset to default configuration"
        )
        
        action_buttons = widgets.HBox([
            start_button,
            stop_button,
            save_button,
            reset_button
        ], layout=widgets.Layout(justify_content='center', padding='20px'))
        
        action_container = widgets.VBox([
            widgets.HTML("<h3 style='color: #28a745; margin: 20px 0 15px 0;'>🎯 Training Actions</h3>"),
            action_buttons
        ], layout=widgets.Layout(width='100%'))
        
        # Progress and logging section
        operation_container = OperationContainer(
            title="Training Progress & Logs",
            show_progress=True,
            show_logs=True,
            max_log_entries=200,
            enable_log_filtering=True
        )
        
        # Summary section (simplified)
        summary_html = widgets.HTML(
            value='''
            <div style="padding: 15px; background: #f8f9fa; border-radius: 8px; margin-top: 20px;">
                <h4 style="color: #495057; margin: 0 0 10px 0;">📊 Training Summary</h4>
                <p style="margin: 5px 0; color: #6c757d;">
                    Training results and metrics will appear here after completion.
                </p>
                <p style="margin: 5px 0; color: #6c757d;">
                    Charts and visualizations will be saved to <code>data/visualization/</code>
                </p>
            </div>
            ''',
            layout=widgets.Layout(width='100%')
        )
        
        summary_container = widgets.VBox([
            widgets.HTML("<h3 style='color: #fd7e14; margin: 20px 0 15px 0;'>📈 Training Summary</h3>"),
            summary_html
        ], layout=widgets.Layout(width='100%'))
        
        # Main layout
        main_layout = widgets.VBox([
            header_html,
            form_wrapper,
            action_container,
            operation_container.container,
            summary_container
        ], layout=widgets.Layout(width='100%', padding='10px'))
        
        # Create UI components dictionary
        ui_components = {
            'main_container': main_layout,
            'header_container': header_html,
            'form_container': form_container,
            'action_container': action_container,
            'operation_container': operation_container,
            'summary_container': summary_container,
            
            # Individual widgets for easy access
            'start_button': start_button,
            'stop_button': stop_button,
            'save_button': save_button,
            'reset_button': reset_button,
            
            # Progress and logging
            'progress_tracker': operation_container.progress_tracker,
            'log_accordion': operation_container.log_accordion,
            
            # Form access
            'form_widgets': getattr(form_container, '_widgets', {}),
            'get_form_values': getattr(form_container, 'get_form_values', lambda: {}),
            'update_form_from_config': getattr(form_container, 'update_from_config', lambda x: None)
        }
        
        # Add helper methods to operation container for easy access
        def update_progress(progress: int, message: str = ""):
            """Update progress bar and message."""
            operation_container.update_progress(progress, message)
        
        def log_info(message: str):
            """Log info message."""
            operation_container.log(message, level='info')
        
        def log_success(message: str):
            """Log success message."""
            operation_container.log(message, level='success')
        
        def log_warning(message: str):
            """Log warning message."""
            operation_container.log(message, level='warning')
        
        def log_error(message: str):
            """Log error message."""
            operation_container.log(message, level='error')
        
        def clear_logs():
            """Clear all logs."""
            operation_container.clear_logs()
        
        # Attach helper methods to main container
        main_layout.update_progress = update_progress
        main_layout.log_info = log_info
        main_layout.log_success = log_success
        main_layout.log_warning = log_warning
        main_layout.log_error = log_error
        main_layout.clear_logs = clear_logs
        
        # Set initial button states
        start_button.disabled = False
        stop_button.disabled = True
        
        logger.debug("✅ Unified training UI created successfully")
        return ui_components
        
    except Exception as e:
        logger.error(f"Failed to create unified training UI: {e}")
        raise


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
        start_button = ui_components.get('start_button')
        stop_button = ui_components.get('stop_button')
        
        if start_button:
            start_button.disabled = is_training or not has_model
            if not has_model:
                start_button.tooltip = "No model available - configure backbone first"
            elif is_training:
                start_button.tooltip = "Training in progress"
            else:
                start_button.tooltip = "Start unified training pipeline"
        
        if stop_button:
            stop_button.disabled = not is_training
            if is_training:
                stop_button.tooltip = "Stop current training"
            else:
                stop_button.tooltip = "No active training to stop"
                
    except Exception as e:
        logger = get_module_logger("smartcash.ui.model.training.components")
        logger.warning(f"Failed to update training button states: {e}")


def update_summary_display(ui_components: Dict[str, Any], result: Dict[str, Any]):
    """Update summary display with training results.
    
    Args:
        ui_components: UI components dictionary
        result: Training result dictionary
    """
    try:
        summary_container = ui_components.get('summary_container')
        if not summary_container or not isinstance(summary_container, widgets.VBox):
            return
        
        if result.get('success'):
            # Get training results
            training_result = result.get('final_training_result', {})
            best_metrics = training_result.get('best_metrics', {})
            
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
        
        # Update the summary HTML widget
        if len(summary_container.children) > 1:
            summary_container.children = (
                summary_container.children[0],  # Keep the title
                widgets.HTML(summary_html, layout=widgets.Layout(width='100%'))
            )
            
    except Exception as e:
        logger = get_module_logger("smartcash.ui.model.training.components")
        logger.warning(f"Failed to update summary display: {e}")