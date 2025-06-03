"""
File: smartcash/ui/training/handlers/reset_training_handler.py
Deskripsi: Handler untuk reset training metrics dan state
"""

from typing import Dict, Any
from smartcash.ui.training.handlers.training_button_handlers import get_state
from smartcash.ui.training.utils.training_status_utils import update_training_status
from smartcash.ui.training.handlers.cleanup_handler import cleanup_training_outputs


def handle_reset_training(ui_components: Dict[str, Any]):
    """Handle reset training dengan model state reset"""
    if get_state()['active']:
        return
    
    # Clear training outputs
    cleanup_training_outputs(ui_components)
    
    # Gunakan training manager jika tersedia
    training_manager = ui_components.get('training_manager')
    if training_manager:
        training_manager.reset_training_state()
    else:
        # Fallback ke training service langsung
        training_service = ui_components.get('training_service')
        if training_service:
            if hasattr(training_service, 'reset_training_state'):
                training_service.reset_training_state()
            elif hasattr(training_service, '_progress_tracker'):
                # Legacy reset
                training_service._progress_tracker = training_service._progress_tracker.__class__()
    
    # Initialize empty chart
    _initialize_empty_chart(ui_components)
    
    update_training_status(ui_components, "🔄 Training metrics direset", 'info')


def _initialize_empty_chart(ui_components: Dict[str, Any]):
    """Initialize empty training chart"""
    # Gunakan fungsi yang sudah ada untuk inisialisasi chart
    from smartcash.ui.training.utils.training_chart_utils import initialize_empty_training_chart
    initialize_empty_training_chart(ui_components)
    
    # Reset metrics history jika ada
    if 'metrics_history' in ui_components:
        ui_components['metrics_history'] = {'train_loss': [], 'val_loss': [], 'epochs': []}
            
    # Reset progress bars jika ada
    for progress_key in ['epoch_progress', 'batch_progress', 'validation_progress']:
        progress_bar = ui_components.get(progress_key)
        if progress_bar and hasattr(progress_bar, 'value'):
            progress_bar.value = 0