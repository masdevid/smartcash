"""
File: smartcash/ui/training/handlers/training_handlers.py
Deskripsi: Consolidated handlers untuk semua training operations dengan one-liner style
"""

from typing import Dict, Any
import threading
from IPython.display import display, HTML
from smartcash.ui.utils.button_state_manager import get_button_state_manager


# Global training state - one-liner state management
_training_state = {'active': False, 'thread': None, 'stop_requested': False}
get_state = lambda: _training_state
set_state = lambda **kwargs: _training_state.update(kwargs)


def setup_all_training_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua training handlers dalam satu function"""
    
    # Button state manager
    ui_components['button_state_manager'] = get_button_state_manager(ui_components)
    
    # Register button handlers dengan one-liner approach
    button_handlers = {
        'start_button': lambda b: handle_start_training(ui_components, config),
        'stop_button': lambda b: handle_stop_training(ui_components),
        'reset_button': lambda b: handle_reset_training(ui_components),
        'cleanup_button': lambda b: handle_cleanup_training(ui_components)
    }
    
    # One-liner button registration
    [getattr(ui_components.get(btn), 'on_click', lambda x: None)(handler) 
     for btn, handler in button_handlers.items() if btn in ui_components]
    
    # Initialize training info
    _update_training_info(ui_components, config)
    
    return ui_components


def handle_start_training(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Handle start training dengan state management"""
    if get_state()['active']:
        return
    
    logger = ui_components.get('logger')
    
    # Update state dan UI - one-liner
    set_state(active=True, stop_requested=False)
    _update_button_states(ui_components, training_active=True)
    _update_status(ui_components, "🚀 Memulai training...", 'info')
    
    # Start training thread
    training_thread = threading.Thread(
        target=_run_training_process, 
        args=(ui_components, config),
        daemon=True
    )
    
    set_state(thread=training_thread)
    training_thread.start()
    
    logger and logger.info("🚀 Training dimulai dalam background thread")


def handle_stop_training(ui_components: Dict[str, Any]):
    """Handle stop training dengan graceful shutdown"""
    if not get_state()['active']:
        return
    
    set_state(stop_requested=True)
    ui_components['stop_button'].disabled = True
    _update_status(ui_components, "⏹️ Menghentikan training...", 'warning')
    
    logger = ui_components.get('logger')
    logger and logger.info("⏹️ Training akan dihentikan")


def handle_reset_training(ui_components: Dict[str, Any]):
    """Handle reset training metrics dan chart"""
    if get_state()['active']:
        return
    
    # Clear outputs - one-liner approach
    outputs_to_clear = ['chart_output', 'metrics_output']
    [getattr(ui_components.get(output), 'clear_output', lambda **kw: None)(wait=True) 
     for output in outputs_to_clear if output in ui_components]
    
    _update_status(ui_components, "🔄 Metrics direset", 'info')
    _initialize_empty_chart(ui_components)


def handle_cleanup_training(ui_components: Dict[str, Any]):
    """Handle cleanup semua training outputs"""
    if get_state()['active']:
        return
    
    # Clear all outputs - one-liner cleanup
    all_outputs = ['chart_output', 'metrics_output', 'log_output', 'info_display']
    [getattr(ui_components.get(output), 'clear_output', lambda **kw: None)(wait=True) 
     for output in all_outputs if output in ui_components]
    
    _update_status(ui_components, "🧹 Training outputs dibersihkan", 'success')


def _run_training_process(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Core training process dalam background thread"""
    try:
        from smartcash.ui.training.handlers.training_operations import execute_training
        
        # Update progress tracking
        progress_tracker = ui_components.get('progress_container', {}).get('tracker')
        progress_tracker and progress_tracker.show('training')
        
        # Execute training
        success = execute_training(ui_components, config, get_state, set_state)
        
        # Update final state
        final_message = "✅ Training selesai!" if success else "❌ Training gagal"
        final_type = 'success' if success else 'error'
        
        _update_status(ui_components, final_message, final_type)
        progress_tracker and (progress_tracker.complete(final_message) if success else progress_tracker.error(final_message))
        
    except Exception as e:
        _update_status(ui_components, f"❌ Error: {str(e)}", 'error')
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Training error: {str(e)}")
    
    finally:
        # Reset state dan button states
        set_state(active=False, stop_requested=False, thread=None)
        _update_button_states(ui_components, training_active=False)


def _update_training_info(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Update training info display dengan config summary"""
    training_config = config.get('training', {})
    
    info_html = f"""
    <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
        <h5>📋 Konfigurasi Training</h5>
        <ul style="margin: 10px 0;">
            <li><b>Backbone:</b> {training_config.get('backbone', 'efficientnet_b4')}</li>
            <li><b>Epochs:</b> {training_config.get('epochs', 100)}</li>
            <li><b>Batch Size:</b> {training_config.get('batch_size', 16)}</li>
            <li><b>Learning Rate:</b> {training_config.get('learning_rate', 0.001)}</li>
            <li><b>Image Size:</b> {training_config.get('image_size', 640)}</li>
            <li><b>Optimizer:</b> {training_config.get('optimizer', 'Adam')}</li>
        </ul>
        <p style="color: #666; font-size: 12px;">Konfigurasi diambil dari modul-modul sebelumnya</p>
    </div>
    """
    
    info_display = ui_components.get('info_display')
    if info_display:
        with info_display:
            info_display.clear_output(wait=True)
            display(HTML(info_html))


def _initialize_empty_chart(ui_components: Dict[str, Any]):
    """Initialize empty training chart"""
    chart_output = ui_components.get('chart_output')
    if not chart_output:
        return
    
    chart_html = """
    <div style="text-align: center; padding: 50px; color: #666;">
        <h4>📈 Training Metrics Chart</h4>
        <p>Chart akan muncul saat training dimulai</p>
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