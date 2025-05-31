"""
File: smartcash/ui/training/handlers/start_training_handler.py
Deskripsi: Updated start training handler dengan full progress tracking integration
"""

from typing import Dict, Any, Callable
from smartcash.ui.training.handlers.training_button_handlers import get_state, set_state
from smartcash.ui.training.utils.training_status_utils import update_training_status
from smartcash.ui.training.utils.training_progress_utils import (
    update_training_progress, update_checkpoint_progress, update_model_loading_progress,
    show_training_progress, complete_all_progress, error_all_progress
)
from smartcash.ui.training.utils.training_logging_utils import (
    log_training_start, log_training_complete, log_epoch_metrics, log_checkpoint_save
)


def handle_start_training(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Handle start training dengan full progress integration"""
    if get_state()['active']:
        return
    
    logger = ui_components.get('logger')
    training_service = ui_components.get('training_service')
    
    # Validate training service
    if not training_service:
        update_training_status(ui_components, "❌ Training service tidak tersedia", 'error')
        error_all_progress(ui_components, "Training service tidak tersedia")
        return
    
    # Update state dan UI
    set_state(active=True, stop_requested=False)
    _update_ui_for_training_start(ui_components)
    
    # Show progress container untuk training
    show_training_progress(ui_components)
    update_training_status(ui_components, "🚀 Memulai training dengan EfficientNet-B4...", 'info')
    
    # Log training start
    log_training_start(ui_components, config)
    
    # Create comprehensive callbacks
    progress_callback = _create_progress_callback(ui_components, training_service)
    metrics_callback = _create_metrics_callback(ui_components)
    checkpoint_callback = _create_checkpoint_callback(ui_components)
    
    # Set callbacks ke training service
    training_service.set_progress_callbacks(progress_callback, metrics_callback, checkpoint_callback)
    
    # Jalankan training secara langsung (tidak perlu threading di colab)
    _execute_training_process(ui_components, training_service, config)
    
    logger and logger.info("🚀 Training started dengan full progress integration")


def _update_ui_for_training_start(ui_components: Dict[str, Any]):
    """Update UI state untuk training start"""
    from smartcash.ui.training.handlers.training_button_handlers import enable_stopping_mode
    enable_stopping_mode(ui_components)


def _create_progress_callback(ui_components: Dict[str, Any], training_service) -> Callable:
    """Create comprehensive progress callback"""
    def progress_callback(current, total, data):
        if get_state()['stop_requested']:
            training_service.stop_training()
            return
        
        # Handle different types of progress data
        if isinstance(data, dict):
            if 'message' in data:
                # General progress message
                update_model_loading_progress(ui_components, current, total, data['message'])
            else:
                # Training metrics progress
                update_training_progress(ui_components, current, total, data)
                
                # Update chart setiap 5 epochs
                if (current + 1) % 5 == 0:
                    from smartcash.ui.training.utils.training_chart_utils import update_training_chart
                    update_training_chart(ui_components, data, current + 1, total)
        else:
            # Simple message
            update_model_loading_progress(ui_components, current, total, str(data))
    
    return progress_callback


def _create_metrics_callback(ui_components: Dict[str, Any]) -> Callable:
    """Create metrics logging callback"""
    def metrics_callback(epoch, metrics):
        # Log detailed metrics
        log_epoch_metrics(ui_components, epoch, metrics)
        
        # Update real-time chart every epoch
        from smartcash.ui.training.utils.training_chart_utils import update_training_chart
        training_config = ui_components.get('config', {})
        total_epochs = training_config.get('epochs', 100)
        update_training_chart(ui_components, metrics, epoch + 1, total_epochs)
    
    return metrics_callback


def _create_checkpoint_callback(ui_components: Dict[str, Any]) -> Callable:
    """Create checkpoint progress callback"""
    def checkpoint_callback(current, total, message):
        # Update checkpoint progress
        update_checkpoint_progress(ui_components, current, total, message)
        
        # Log checkpoint operations
        if 'saved' in message.lower():
            log_checkpoint_save(ui_components, message.split(':')[-1].strip() if ':' in message else 'checkpoint.pt', current)
    
    return checkpoint_callback


def _execute_training_process(ui_components: Dict[str, Any], training_service, config: Dict[str, Any]):
    """Execute training process dengan comprehensive error handling"""
    try:
        # Start training dengan full callback integration
        success = training_service.start_training()
        
        # Handle completion
        _handle_training_completion(ui_components, success, training_service)
        
    except Exception as e:
        _handle_training_error(ui_components, e)
    
    finally:
        _reset_training_state(ui_components)


def _handle_training_completion(ui_components: Dict[str, Any], success: bool, training_service):
    """Handle training completion dengan progress updates"""
    if success:
        # Get final metrics from training service atau chart utils
        try:
            from smartcash.ui.training.utils.training_chart_utils import get_training_summary
            summary = get_training_summary()
            final_metrics = summary.get('final_metrics', {})
        except Exception:
            final_metrics = {}
        
        final_message = "✅ Training selesai dengan sukses!"
        update_training_status(ui_components, final_message, 'success')
        complete_all_progress(ui_components, final_message)
        log_training_complete(ui_components, True, final_metrics)
    else:
        final_message = "❌ Training dihentikan atau gagal"
        update_training_status(ui_components, final_message, 'error')
        error_all_progress(ui_components, final_message)
        log_training_complete(ui_components, False)


def _handle_training_error(ui_components: Dict[str, Any], error: Exception):
    """Handle training error dengan comprehensive logging"""
    error_message = f"❌ Training error: {str(error)}"
    update_training_status(ui_components, error_message, 'error')
    error_all_progress(ui_components, error_message)
    
    logger = ui_components.get('logger')
    logger and logger.error(error_message)
    
    # Log error details
    from smartcash.ui.training.utils.training_logging_utils import log_error
    log_error(ui_components, f"Training execution failed: {str(error)}")


def _reset_training_state(ui_components: Dict[str, Any]):
    """Reset training state dan UI dengan cleanup"""
    from smartcash.ui.training.handlers.training_button_handlers import enable_training_mode
    
    set_state(active=False, stop_requested=False)
    enable_training_mode(ui_components)
    
    # Reset training service state jika ada
    training_service = ui_components.get('training_service')
    if training_service and hasattr(training_service, 'reset_training_state'):
        training_service.reset_training_state()


def validate_training_readiness(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Validate training readiness sebelum start"""
    from smartcash.ui.training.handlers.validation_handler import validate_model_before_training
    return validate_model_before_training(ui_components, config)


# One-liner utilities untuk training flow
can_start_training = lambda ui: not get_state()['active'] and ui.get('training_service') and ui.get('model_manager')
is_training_ready = lambda ui, config: validate_training_readiness(ui, config)
start_training_safe = lambda ui, config: handle_start_training(ui, config) if can_start_training(ui) else None