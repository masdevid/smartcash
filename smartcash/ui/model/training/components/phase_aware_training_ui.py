"""
File: smartcash/ui/model/training/components/phase_aware_training_ui.py
Description: Phase-aware training UI with two-column metrics layout and enhanced callbacks.
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from smartcash.ui.logger import get_ui_logger, LogLevel

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
from .phase_aware_metrics import (
    generate_phase_aware_metrics_html,
    get_initial_phase_aware_metrics_html,
    create_live_metrics_update
)
from .enhanced_callbacks import create_enhanced_training_callbacks

def create_phase_aware_charts_and_metrics(config: Dict[str, Any]) -> widgets.Widget:
    """Create phase-aware charts and metrics section with two-column layout.
    
    Args:
        config: Configuration for charts and metrics
        
    Returns:
        Accordion widget containing phase-aware charts and metrics
    """
    # Create logger with suppression for initialization
    logger = get_ui_logger(
        __name__,
        ui_components=None,
        level=LogLevel.INFO
    )
    logger.suppress()  # Suppress logging until UI is ready
    
    try:
        # Get training mode from config
        training_mode = config.get('training', {}).get('training_mode', 'two_phase')
        
        # Create chart configuration
        chart_config = config.get('charts', {
            'charts': {
                'loss_chart': {'update_frequency': 'epoch'},
                'metrics_chart': {'update_frequency': 'epoch'}  # Renamed from map_chart
            },
            'monitoring': {'primary_metric': 'phase_metrics'}
        })
        
        # Create dual charts with phase awareness
        charts = create_dual_charts_layout(chart_config, config.get('ui', {}))
        
        # Create phase-aware charts container
        charts_container = widgets.VBox([
            widgets.HTML(
                value="<h4 style='text-align: center; color: #495057; margin: 10px 0;'>📈 Phase-Aware Training Charts</h4>"
            ),
            widgets.HBox([
                charts['loss_chart'],
                charts.get('metrics_chart', charts.get('map_chart'))  # Fallback compatibility
            ], layout=widgets.Layout(width='100%'))
        ], layout=widgets.Layout(width='100%', padding='10px'))
        
        # Create phase-aware metrics display
        metrics_display = widgets.HTML(
            value=get_initial_phase_aware_metrics_html(training_mode),
            layout=widgets.Layout(width='100%', padding='10px')
        )
        
        # Create live progress display
        progress_display = widgets.VBox([
            widgets.HTML(
                value="<h5 style='color: #495057; margin-bottom: 10px;'>🔄 Training Progress</h5>"
            ),
            widgets.HTML(
                value="<div style='background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; color: #6c757d;'>Training progress will appear here</div>",
                layout=widgets.Layout(width='100%')
            )
        ], layout=widgets.Layout(width='100%', padding='10px'))
        
        # Create log output display
        log_display = widgets.Textarea(
            value="Training logs will appear here...\n",
            placeholder="Training logs...",
            layout=widgets.Layout(width='100%', height='200px'),
            disabled=True
        )
        
        # Create accordion for all sections
        accordion = widgets.Accordion(children=[
            charts_container,
            metrics_display,
            progress_display,
            log_display
        ])
        
        accordion.set_title(0, "📈 Live Training Charts")
        accordion.set_title(1, "📊 Phase-Aware Metrics")
        accordion.set_title(2, "🔄 Training Progress")
        accordion.set_title(3, "📝 Training Logs")
        accordion.selected_index = 1  # Start with metrics visible
        
        # Store references for updates
        accordion._charts = charts
        accordion._metrics_display = metrics_display
        accordion._progress_display = progress_display.children[1]
        accordion._log_display = log_display
        accordion._training_mode = training_mode
        accordion._current_phase = 1
        accordion._current_epoch = 0
        
        # Enhanced update methods
        def update_phase_aware_metrics(update_data: Dict[str, Any]):
            """Update phase-aware metrics display."""
            try:
                accordion._current_phase = update_data.get('phase', 1)
                accordion._current_epoch = update_data.get('epoch', 0)
                
                # Update metrics HTML
                metrics_display.value = update_data.get('html', metrics_display.value)
                
                logger.debug(f"Updated phase-aware metrics: Phase {accordion._current_phase}, Epoch {accordion._current_epoch}")
            except Exception as e:
                logger.warning(f"Failed to update phase-aware metrics: {e}")
        
        def update_charts_data(data: Dict[str, Any]):
            """Update charts with phase-aware training data."""
            try:
                # Update loss chart
                charts['update_loss'](data)
                
                # Update metrics chart with phase-aware data
                if 'update_metrics' in charts:
                    charts['update_metrics'](data)
                elif 'update_map' in charts:  # Fallback compatibility
                    charts['update_map'](data)
                    
                logger.debug(f"Updated charts with data: {list(data.keys())}")
            except Exception as e:
                logger.warning(f"Failed to update charts: {e}")
        
        def update_progress_display(progress_data: Dict[str, Any]):
            """Update progress display."""
            try:
                progress_type = progress_data.get('type', 'unknown')
                current = progress_data.get('current', 0)
                total = progress_data.get('total', 1)
                message = progress_data.get('message', '')
                percentage = progress_data.get('percentage', 0)
                
                # Create progress HTML
                progress_html = f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-weight: 600; color: #495057;">{progress_type.replace('_', ' ').title()}</span>
                        <span style="font-weight: 600; color: #007bff;">{percentage:.1f}%</span>
                    </div>
                    <div style="background: #e9ecef; border-radius: 10px; height: 8px; margin-bottom: 8px;">
                        <div style="background: #007bff; border-radius: 10px; height: 8px; width: {percentage}%; transition: width 0.3s ease;"></div>
                    </div>
                    <div style="font-size: 12px; color: #6c757d; text-align: center;">{message}</div>
                    <div style="font-size: 11px; color: #6c757d; text-align: center; margin-top: 4px;">({current}/{total})</div>
                </div>
                """
                
                accordion._progress_display.value = progress_html
                
            except Exception as e:
                logger.warning(f"Failed to update progress display: {e}")
        
        def update_log_output(log_message: str):
            """Update log output display."""
            try:
                current_logs = accordion._log_display.value
                accordion._log_display.value = current_logs + log_message + "\n"
                
                # Auto-scroll to bottom (simulate)
                lines = accordion._log_display.value.split('\n')
                if len(lines) > 100:  # Keep last 100 lines
                    accordion._log_display.value = '\n'.join(lines[-100:])
                    
            except Exception as e:
                logger.warning(f"Failed to update log output: {e}")
        
        def reset_all_displays():
            """Reset all displays to initial state."""
            try:
                # Reset charts
                if 'reset_charts' in charts:
                    charts['reset_charts']()
                
                # Reset metrics
                metrics_display.value = get_initial_phase_aware_metrics_html(accordion._training_mode)
                
                # Reset progress
                accordion._progress_display.value = "<div style='background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; color: #6c757d;'>Training progress will appear here</div>"
                
                # Reset logs
                accordion._log_display.value = "Training logs will appear here...\n"
                
                # Reset state
                accordion._current_phase = 1
                accordion._current_epoch = 0
                
                logger.info("Reset all displays to initial state")
                
            except Exception as e:
                logger.warning(f"Failed to reset displays: {e}")
        
        # Attach methods to accordion
        accordion.update_phase_aware_metrics = update_phase_aware_metrics
        accordion.update_charts_data = update_charts_data
        accordion.update_progress_display = update_progress_display
        accordion.update_log_output = update_log_output
        accordion.reset_all_displays = reset_all_displays
        
        return accordion
    
    except Exception as e:
        logger.error(f"Failed to create phase-aware charts and metrics: {e}")
        # Return fallback widget
        return widgets.HTML(
            value=f"<div style='background: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px;'>❌ Failed to create phase-aware display: {e}</div>"
        )


@handle_ui_errors("create_phase_aware_training_ui")
def create_phase_aware_training_ui(config: Dict[str, Any]) -> widgets.Widget:
    """Create complete phase-aware training UI with enhanced callbacks.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Complete training UI widget
    """
    # Create logger with suppression for initialization
    logger = get_ui_logger(
        __name__,
        ui_components=None,
        level=LogLevel.INFO
    )
    logger.suppress()  # Suppress logging until UI is ready
    
    try:
        # Create header
        header = create_header_container(
            title="🚀 SmartCash Model Training",
            subtitle="Phase-aware training with real-time metrics and progress tracking",
            config=config.get('ui', {})
        )
        
        # Create unified training form
        form_components = create_unified_training_form(config)
        form_container = create_form_container(
            "Training Configuration",
            form_components,
            layout_type=LayoutType.VERTICAL,
            config=config.get('ui', {})
        )
        
        # Create action container (will be populated by operation handler)
        action_container = create_action_container(
            "Training Actions",
            [],  # Will be populated by operation
            config=config.get('ui', {})
        )
        
        # Create phase-aware charts and metrics
        charts_and_metrics = create_phase_aware_charts_and_metrics(config)
        
        # Create operation container for training execution
        operation_container = create_operation_container(
            "Training Execution",
            config=config.get('ui', {})
        )
        
        # Create summary container
        summary_container = create_summary_container(
            "Training Summary",
            "Training results and performance metrics will appear here",
            config=config.get('ui', {})
        )
        
        # Create main container with all components
        main_ui = create_main_container(
            [
                header,
                form_container,
                action_container,
                charts_and_metrics,
                operation_container,
                summary_container
            ],
            config=config.get('ui', {})
        )
        
        # Store references for external access
        main_ui._header = header
        main_ui._form_container = form_container
        main_ui._action_container = action_container
        main_ui._charts_and_metrics = charts_and_metrics
        main_ui._operation_container = operation_container
        main_ui._summary_container = summary_container
        
        # Enhanced callback integration methods
        def setup_enhanced_callbacks():
            """Setup enhanced callbacks for training integration."""
            try:
                # Create enhanced callbacks with UI integration
                callbacks = create_enhanced_training_callbacks(
                    ui_module=main_ui,
                    verbose=config.get('training', {}).get('verbose', True)
                )
                
                # Store callbacks for use by operations
                main_ui._enhanced_callbacks = callbacks
                
                logger.info("Enhanced callbacks setup completed")
                return callbacks
                
            except Exception as e:
                logger.error(f"Failed to setup enhanced callbacks: {e}")
                return None
        
        def _update_metrics_display(update_data: Dict[str, Any]):
            """Update metrics display with phase-aware data."""
            try:
                charts_and_metrics.update_phase_aware_metrics(update_data)
            except Exception as e:
                logger.warning(f"Failed to update metrics display: {e}")
        
        def _update_progress_display(progress_data: Dict[str, Any]):
            """Update progress display."""
            try:
                charts_and_metrics.update_progress_display(progress_data)
            except Exception as e:
                logger.warning(f"Failed to update progress display: {e}")
        
        def _update_log_output(log_message: str):
            """Update log output."""
            try:
                charts_and_metrics.update_log_output(log_message)
            except Exception as e:
                logger.warning(f"Failed to update log output: {e}")
        
        def _update_live_charts(chart_data: Dict[str, Any]):
            """Update live charts."""
            try:
                charts_and_metrics.update_charts_data(chart_data)
            except Exception as e:
                logger.warning(f"Failed to update live charts: {e}")
        
        def _handle_colored_metrics(phase: str, epoch: int, metrics: Dict[str, Any], colored_metrics: Dict[str, Dict]):
            """Handle colored metrics from UI callback."""
            try:
                # This could be used for additional UI enhancements
                logger.debug(f"Received colored metrics for {phase} epoch {epoch}")
            except Exception as e:
                logger.warning(f"Failed to handle colored metrics: {e}")
        
        # Attach methods to main UI
        main_ui.setup_enhanced_callbacks = setup_enhanced_callbacks
        main_ui._update_metrics_display = _update_metrics_display
        main_ui._update_progress_display = _update_progress_display
        main_ui._update_log_output = _update_log_output
        main_ui._update_live_charts = _update_live_charts
        main_ui._handle_colored_metrics = _handle_colored_metrics
        
        # Set up UI components for logger
        logger.ui_components = {
            'operation_container': operation_container,
            'log_output': log_display
        }
        logger.unsuppress()  # Enable logging now that UI is ready
        
        logger.info("Phase-aware training UI created successfully")
        return main_ui
    
    except Exception as e:
        logger.error(f"Failed to create phase-aware training UI: {e}")
        raise


# Backward compatibility
def create_unified_training_ui(config: Dict[str, Any]) -> widgets.Widget:
    """Backward compatibility wrapper."""
    return create_phase_aware_training_ui(config)
