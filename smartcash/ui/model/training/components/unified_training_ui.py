"""
File: smartcash/ui/model/training/components/unified_training_ui.py
Description: Simplified training UI that uses the unified training pipeline with proper container components.
"""

from typing import Dict, Any, Optional
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
    action_buttons = [
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
        buttons=action_buttons,
        show_save_reset=True  # Use standard save/reset buttons
    )
    
    # 4. Create Operation Container
    operation_container = create_operation_container(
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        progress_levels='dual',
        log_module_name="Training",
        log_height="200px",
        collapsible=True,
        collapsed=False
    )
    
    # 5. Create Summary Container
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
                Training results and metrics will appear here after completion.
            </p>
            <p style="margin: 5px 0; color: #6c757d;">
                Charts and visualizations will be saved to <code>data/visualization/</code>
            </p>
        </div>
        ''')
    
    # 6. Create Main Container
    main_container = create_main_container(
        components=[
            {'component': header_container.container, 'type': 'header'},
            {'component': form_container['container'], 'type': 'form'},
            {'component': action_container['container'], 'type': 'action'},
            {'component': operation_container['container'], 'type': 'operation'},
            {'component': summary_container.container, 'type': 'summary'}
        ]
    )
    
    # 7. Create UI components dictionary
    ui_components = {
        'main_container': main_container.container,
        'header_container': header_container,
        'form_container': form_container,
        'action_container': action_container,
        'operation_container': operation_container,
        'summary_container': summary_container,
        
        # Individual action buttons for easy access
        'start_training_button': action_container['buttons'].get('start_training'),
        'save_button': action_container['buttons'].get('save'),
        'reset_button': action_container['buttons'].get('reset'),
        
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
        'update_form_from_config': getattr(form_widgets, 'update_from_config', lambda x: None)
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
        start_button = ui_components.get('start_training_button')
        
        if start_button and hasattr(start_button, 'disabled'):
            start_button.disabled = is_training or not has_model
            if not has_model:
                start_button.tooltip = "No model available - configure backbone first"
            elif is_training:
                start_button.tooltip = "Training in progress"
            else:
                start_button.tooltip = "Start unified training pipeline"
                
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
        if not summary_container or not hasattr(summary_container, 'set_content'):
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
        
        # Update the summary container content
        summary_container.set_content(summary_html)
            
    except Exception as e:
        logger = get_module_logger("smartcash.ui.model.training.components")
        logger.warning(f"Failed to update summary display: {e}")