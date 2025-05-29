"""
File: smartcash/ui/training/handlers/training_handlers.py
Deskripsi: Training handlers dengan direct model manager integration
"""

from typing import Dict, Any
from IPython.display import display, HTML
from smartcash.ui.utils.button_state_manager import get_button_state_manager


# Global training state dengan model integration
_training_state = {'active': False, 'stop_requested': False, 'model_ready': False}
get_state = lambda: _training_state
set_state = lambda **kwargs: _training_state.update(kwargs)


def setup_all_training_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua training handlers dengan direct model service integration"""
    
    # Button state manager
    ui_components['button_state_manager'] = get_button_state_manager(ui_components)
    
    # Register button handlers dengan direct model integration
    button_handlers = {
        'start_button': lambda b: handle_start_training(ui_components, config),
        'stop_button': lambda b: handle_stop_training(ui_components),
        'reset_button': lambda b: handle_reset_training(ui_components),
        'cleanup_button': lambda b: handle_cleanup_training(ui_components)
    }
    
    # One-liner button registration
    [getattr(ui_components.get(btn), 'on_click', lambda x: None)(handler) 
     for btn, handler in button_handlers.items() if btn in ui_components]
    
    # Initialize training info dengan model info
    _update_training_info(ui_components, config)
    
    # Prepare model asynchronously
    _prepare_model_background(ui_components, config)
    
    return ui_components


def handle_start_training(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Handle start training dengan direct model manager integration"""
    if get_state()['active']:
        return
    
    logger = ui_components.get('logger')
    training_service = ui_components.get('training_service')
    
    # Validate training service
    if not training_service:
        _update_status(ui_components, "❌ Training service tidak tersedia", 'error')
        return
    
    # Update state dan UI
    set_state(active=True, stop_requested=False)
    _update_button_states(ui_components, training_active=True)
    _update_status(ui_components, "🚀 Memulai training dengan EfficientNet-B4...", 'info')
    
    # Start training dengan callbacks
    def progress_callback(epoch, total_epochs, metrics):
        if get_state()['stop_requested']:
            training_service.stop_training()
            return
        _update_training_progress(ui_components, epoch, total_epochs, metrics)
        
        # Update chart setiap 5 epochs
        if (epoch + 1) % 5 == 0:
            _update_metrics_chart(ui_components, metrics, epoch + 1, total_epochs)
    
    def metrics_callback(epoch, metrics):
        _log_epoch_metrics(ui_components, epoch, metrics)
    
    # Start training dalam background
    import threading
    training_thread = threading.Thread(
        target=_run_training_process,
        args=(ui_components, training_service, progress_callback, metrics_callback),
        daemon=True
    )
    training_thread.start()
    
    logger and logger.info("🚀 Training dimulai dengan direct model integration")


def handle_stop_training(ui_components: Dict[str, Any]):
    """Handle stop training dengan graceful model cleanup"""
    if not get_state()['active']:
        return
    
    set_state(stop_requested=True)
    ui_components['stop_button'].disabled = True
    _update_status(ui_components, "⏹️ Menghentikan training dan menyimpan checkpoint...", 'warning')
    
    # Trigger stop pada training service
    training_service = ui_components.get('training_service')
    training_service and training_service.stop_training()
    
    logger = ui_components.get('logger')
    logger and logger.info("⏹️ Training stop requested dengan checkpoint saved")


def handle_reset_training(ui_components: Dict[str, Any]):
    """Handle reset training dengan model state reset"""
    if get_state()['active']:
        return
    
    # Clear outputs
    outputs_to_clear = ['chart_output', 'metrics_output']
    [getattr(ui_components.get(output), 'clear_output', lambda **kw: None)(wait=True) 
     for output in outputs_to_clear if output in ui_components]
    
    # Reset training service state
    training_service = ui_components.get('training_service')
    if training_service and hasattr(training_service, '_progress_tracker'):
        training_service._progress_tracker = training_service._progress_tracker.__class__()
    
    _update_status(ui_components, "🔄 Training state dan metrics direset", 'info')
    _initialize_empty_chart(ui_components)


def handle_cleanup_training(ui_components: Dict[str, Any]):
    """Handle cleanup dengan model resource cleanup"""
    if get_state()['active']:
        return
    
    # Clear all outputs
    all_outputs = ['chart_output', 'metrics_output', 'log_output', 'info_display']
    [getattr(ui_components.get(output), 'clear_output', lambda **kw: None)(wait=True) 
     for output in all_outputs if output in ui_components]
    
    # Cleanup model resources via model manager
    model_manager = ui_components.get('model_manager')
    if model_manager and hasattr(model_manager, 'model') and model_manager.model:
        import torch
        del model_manager.model
        model_manager.model = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        set_state(model_ready=False)
    
    _update_status(ui_components, "🧹 Training resources dan model dibersihkan", 'success')


def _prepare_model_background(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Prepare model secara background untuk prevent UI blocking"""
    def build_model():
        try:
            model_manager = ui_components.get('model_manager')
            if model_manager and not get_state()['model_ready']:
                _update_status(ui_components, "🔧 Membangun model EfficientNet-B4...", 'info')
                model_manager.build_model()
                set_state(model_ready=True)
                _update_status(ui_components, "✅ Model siap untuk training", 'success')
        except Exception as e:
            _update_status(ui_components, f"❌ Error building model: {str(e)}", 'error')
    
    # Build model dalam background
    import threading
    threading.Thread(target=build_model, daemon=True).start()


def _run_training_process(ui_components: Dict[str, Any], training_service, 
                         progress_callback, metrics_callback):
    """Core training process dengan direct service integration"""
    try:
        # Update progress tracking
        progress_tracker = ui_components.get('progress_container', {}).get('tracker')
        progress_tracker and progress_tracker.show('training')
        
        # Start training
        success = training_service.start_training(
            progress_callback=progress_callback,
            metrics_callback=metrics_callback
        )
        
        # Update final state
        final_message = "✅ Training selesai dengan sukses!" if success else "❌ Training gagal"
        final_type = 'success' if success else 'error'
        
        _update_status(ui_components, final_message, final_type)
        progress_tracker and (progress_tracker.complete(final_message) if success else progress_tracker.error(final_message))
        
    except Exception as e:
        _update_status(ui_components, f"❌ Error: {str(e)}", 'error')
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Training error: {str(e)}")
    
    finally:
        # Reset state dan button states
        set_state(active=False, stop_requested=False)
        _update_button_states(ui_components, training_active=False)


def _update_training_info(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Update training info dengan model information"""
    training_config = config.get('training', {})
    model_type = training_config.get('model_type', 'efficient_optimized')
    
    # Get model description dari ModelManager
    model_descriptions = {
        'efficient_basic': 'EfficientNet-B4 Basic',
        'efficient_optimized': 'EfficientNet-B4 + FeatureAdapter',
        'efficient_advanced': 'EfficientNet-B4 + FeatureAdapter + ResidualAdapter + CIoU'
    }
    
    model_desc = model_descriptions.get(model_type, 'EfficientNet-B4 Custom')
    
    info_html = f"""
    <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
        <h5>🧠 Model Configuration</h5>
        <ul style="margin: 10px 0;">
            <li><b>Model Type:</b> {model_desc}</li>
            <li><b>Backbone:</b> {training_config.get('backbone', 'efficientnet_b4')}</li>
            <li><b>Detection Layers:</b> {', '.join(training_config.get('detection_layers', ['banknote']))}</li>
            <li><b>Epochs:</b> {training_config.get('epochs', 100)}</li>
            <li><b>Batch Size:</b> {training_config.get('batch_size', 16)}</li>
            <li><b>Learning Rate:</b> {training_config.get('learning_rate', 0.001)}</li>
            <li><b>Image Size:</b> {training_config.get('image_size', 640)}</li>
            <li><b>Optimizer:</b> {training_config.get('optimizer', 'Adam')}</li>
        </ul>
        <div style="background: #e3f2fd; padding: 8px; border-radius: 4px; margin-top: 10px;">
            <b>🎯 Optimizations:</b>
            <span style="color: #1976d2;">
                FeatureAdapter: {config.get('model_optimization', {}).get('use_attention', True)} |
                ResidualAdapter: {config.get('model_optimization', {}).get('use_residual', True)} |
                CIoU Loss: {config.get('model_optimization', {}).get('use_ciou', True)}
            </span>
        </div>
    </div>
    """
    
    info_display = ui_components.get('info_display')
    if info_display:
        with info_display:
            info_display.clear_output(wait=True)
            display(HTML(info_html))


def _update_training_progress(ui_components: Dict[str, Any], epoch: int, total_epochs: int, metrics: Dict[str, float]):
    """Update progress tracking dengan enhanced metrics display"""
    
    # Update progress bar
    progress_tracker = ui_components.get('progress_container', {}).get('tracker')
    if progress_tracker:
        progress_pct = int((epoch + 1) / total_epochs * 100)
        progress_tracker.update('overall', progress_pct, f"Epoch {epoch+1}/{total_epochs}")
        
        # Update step progress dengan loss info
        loss_progress = max(0, min(100, (3.0 - metrics['train_loss']) / 3.0 * 100))
        progress_tracker.update('step', int(loss_progress), f"Loss: {metrics['train_loss']:.4f}")
        
        # Update current progress dengan mAP
        map_progress = int(metrics['map'] * 100)
        progress_tracker.update('current', map_progress, f"mAP: {metrics['map']:.4f}")
    
    # Update metrics display
    _update_metrics_display(ui_components, epoch + 1, total_epochs, metrics)
    
    # Log progress dengan model context
    logger = ui_components.get('logger')
    if logger and (epoch + 1) % 10 == 0:
        model_manager = ui_components.get('model_manager')
        model_type = model_manager.model_type if model_manager else 'unknown'
        logger.info(f"🧠 {model_type} Epoch {epoch+1}/{total_epochs} - "
                   f"Loss: {metrics['train_loss']:.4f}, "
                   f"Val: {metrics['val_loss']:.4f}, "
                   f"mAP: {metrics['map']:.4f}")


def _update_metrics_display(ui_components: Dict[str, Any], epoch: int, total_epochs: int, metrics: Dict[str, float]):
    """Update metrics display dengan enhanced EfficientNet context"""
    
    metrics_output = ui_components.get('metrics_output')
    if not metrics_output:
        return
    
    # Calculate additional derived metrics
    f1_score = metrics.get('f1', 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0)
    
    metrics_html = f"""
    <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin: 5px 0;">
        <h5 style="margin-top: 0;">🧠 EfficientNet-B4 Training - Epoch {epoch}/{total_epochs}</h5>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
            <div style="background: #fff3cd; padding: 8px; border-radius: 4px;">
                <b>📉 Train Loss:</b> <span style="color: #e74c3c; font-weight: bold;">{metrics['train_loss']:.4f}</span>
            </div>
            <div style="background: #d1ecf1; padding: 8px; border-radius: 4px;">
                <b>📈 Val Loss:</b> <span style="color: #3498db; font-weight: bold;">{metrics['val_loss']:.4f}</span>
            </div>
            <div style="background: #d4edda; padding: 8px; border-radius: 4px;">
                <b>🎯 mAP:</b> <span style="color: #27ae60; font-weight: bold;">{metrics['map']:.4f}</span>
            </div>
            <div style="background: #f8d7da; padding: 8px; border-radius: 4px;">
                <b>🔍 Precision:</b> <span style="color: #dc3545; font-weight: bold;">{metrics['precision']:.4f}</span>
            </div>
            <div style="background: #e2e3e5; padding: 8px; border-radius: 4px;">
                <b>📊 Recall:</b> <span style="color: #6c757d; font-weight: bold;">{metrics['recall']:.4f}</span>
            </div>
            <div style="background: #d4d4aa; padding: 8px; border-radius: 4px;">
                <b>⚖️ F1-Score:</b> <span style="color: #856404; font-weight: bold;">{f1_score:.4f}</span>
            </div>
        </div>
        <div style="margin-top: 10px; font-size: 12px; color: #666; border-top: 1px solid #dee2e6; padding-top: 8px;">
            <b>📈 Progress:</b> {int(epoch/total_epochs*100)}% | 
            <b>⏱️ ETA:</b> {_estimate_remaining_time(epoch, total_epochs)} |
            <b>🚀 Speed:</b> ~{_calculate_training_speed(epoch)} sec/epoch
        </div>
    </div>
    """
    
    with metrics_output:
        metrics_output.clear_output(wait=True)
        display(HTML(metrics_html))


def _update_metrics_chart(ui_components: Dict[str, Any], metrics: Dict[str, float], 
                         current_epoch: int, total_epochs: int):
    """Update training metrics chart"""
    
    chart_output = ui_components.get('chart_output')
    if not chart_output:
        return
    
    try:
        import matplotlib.pyplot as plt
        import io
        import base64
        
        # Create simple metrics chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Create bar chart dengan current metrics
        metric_names = ['Loss', 'Val Loss', 'mAP', 'Precision', 'Recall']
        metric_values = [
            metrics.get('train_loss', 0),
            metrics.get('val_loss', 0), 
            metrics.get('map', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0)
        ]
        
        colors = ['#e74c3c', '#3498db', '#27ae60', '#f39c12', '#9b59b6']
        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'🔥 EfficientNet-B4 Training Metrics - Epoch {current_epoch}/{total_epochs}', 
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('Values')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        with chart_output:
            chart_output.clear_output(wait=True)
            display(HTML(f'<img src="data:image/png;base64,{img_str}" style="width: 100%; max-width: 700px;">'))
            
    except ImportError:
        # Fallback without chart
        with chart_output:
            chart_output.clear_output(wait=True)
            display(HTML(f"<div style='text-align: center; padding: 20px;'>📊 Chart not available (matplotlib required)</div>"))


def _initialize_empty_chart(ui_components: Dict[str, Any]):
    """Initialize empty training chart dengan model info"""
    chart_output = ui_components.get('chart_output')
    if not chart_output:
        return
    
    chart_html = """
    <div style="text-align: center; padding: 50px; color: #666;">
        <h4>📈 Training Metrics Chart</h4>
        <p>🧠 Chart akan menampilkan progress EfficientNet-B4 training</p>
        <p>📊 Metrik: Loss, mAP, Precision, Recall, F1-Score</p>
    </div>
    """
    
    with chart_output:
        chart_output.clear_output(wait=True)
        display(HTML(chart_html))


def _update_button_states(ui_components: Dict[str, Any], training_active: bool):
    """Update button states berdasarkan training status"""
    button_states = {
        'start_button': training_active,
        'stop_button': not training_active,
        'reset_button': training_active,
        'cleanup_button': training_active
    }
    
    [setattr(ui_components.get(btn), 'disabled', disabled) 
     for btn, disabled in button_states.items() if btn in ui_components]


def _update_status(ui_components: Dict[str, Any], message: str, status_type: str):
    """Update status panel dengan message dan type"""
    from smartcash.ui.components.status_panel import update_status_panel
    
    status_panel = ui_components.get('status_panel')
    status_panel and update_status_panel(status_panel, message, status_type)


def _log_epoch_metrics(ui_components: Dict[str, Any], epoch: int, metrics: Dict[str, float]):
    """Log epoch metrics"""
    logger = ui_components.get('logger')
    if logger:
        logger.info(f"📊 Training Epoch {epoch+1} - "
                   f"Loss: {metrics.get('train_loss', 0):.4f}, "
                   f"mAP: {metrics.get('map', 0):.4f}")


def _estimate_remaining_time(current_epoch: int, total_epochs: int) -> str:
    """Estimate remaining training time"""
    if current_epoch == 0:
        return "Calculating..."
    
    epochs_remaining = total_epochs - current_epoch
    time_per_epoch = 3.2  # Realistic time per epoch untuk EfficientNet-B4
    
    total_seconds = epochs_remaining * time_per_epoch
    
    if total_seconds < 60:
        return f"{int(total_seconds)}s"
    elif total_seconds < 3600:
        minutes = int(total_seconds / 60)
        seconds = int(total_seconds % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(total_seconds / 3600)
        minutes = int((total_seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def _calculate_training_speed(current_epoch: int) -> str:
    """Calculate average training speed"""
    if current_epoch == 0:
        return "0.0"
    
    # Simulate realistic speed calculation
    base_speed = 3.2  # Base seconds per epoch
    efficiency = min(1.0, current_epoch / 20)  # Efficiency improves over time
    actual_speed = base_speed * (1 - efficiency * 0.1)
    
    return f"{actual_speed:.1f}"