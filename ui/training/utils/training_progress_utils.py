"""
File: smartcash/ui/training/utils/training_progress_utils.py
Deskripsi: Progress utilities dengan 3-bar progress tracking system
"""

from typing import Dict, Any


def update_training_progress(ui_components: Dict[str, Any], epoch: int, total_epochs: int, metrics: Dict[str, float]):
    """Update training progress dengan 3-bar system"""
    progress_tracker = ui_components.get('progress_container', {}).get('tracker')
    if not progress_tracker:
        return
    
    # Overall progress (epochs)
    overall_progress = int((epoch / total_epochs) * 100)
    progress_tracker.update('overall', overall_progress, f"Epoch {epoch+1}/{total_epochs}")
    
    # Step progress (based on loss improvement)
    step_progress = _calculate_step_progress(metrics)
    progress_tracker.update('step', step_progress, f"Loss: {metrics.get('train_loss', 0):.4f}")
    
    # Current operation progress (based on metrics quality)
    current_progress = _calculate_current_progress(metrics)
    progress_tracker.update('current', current_progress, f"mAP: {metrics.get('map', 0):.4f}")


def update_checkpoint_progress(ui_components: Dict[str, Any], current: int, total: int, message: str):
    """Update checkpoint operation progress"""
    progress_tracker = ui_components.get('progress_container', {}).get('tracker')
    if not progress_tracker:
        return
    
    # Use current bar untuk checkpoint operations
    checkpoint_progress = int((current / total) * 100) if total > 0 else 0
    progress_tracker.update('current', checkpoint_progress, message)


def update_model_loading_progress(ui_components: Dict[str, Any], current: int, total: int, message: str):
    """Update model loading progress"""
    progress_tracker = ui_components.get('progress_container', {}).get('tracker')
    if not progress_tracker:
        return
    
    # Use step bar untuk model operations
    loading_progress = int((current / total) * 100) if total > 0 else 0
    progress_tracker.update('step', loading_progress, message)


def update_validation_progress(ui_components: Dict[str, Any], current: int, total: int, message: str):
    """Update validation progress"""
    progress_tracker = ui_components.get('progress_container', {}).get('tracker')
    if not progress_tracker:
        return
    
    # Use overall bar untuk validation operations
    validation_progress = int((current / total) * 100) if total > 0 else 0
    progress_tracker.update('overall', validation_progress, message)


def reset_all_progress(ui_components: Dict[str, Any]):
    """Reset semua progress bars"""
    progress_tracker = ui_components.get('progress_container', {}).get('tracker')
    if not progress_tracker:
        return
    
    progress_tracker.update('overall', 0, "Siap memulai...")
    progress_tracker.update('step', 0, "Menunggu...")
    progress_tracker.update('current', 0, "Standby...")


def complete_all_progress(ui_components: Dict[str, Any], message: str = "Selesai!"):
    """Complete semua progress bars"""
    progress_tracker = ui_components.get('progress_container', {}).get('tracker')
    if not progress_tracker:
        return
    
    progress_tracker.complete(message)


def error_all_progress(ui_components: Dict[str, Any], message: str = "Error occurred"):
    """Set error state untuk semua progress bars"""
    progress_tracker = ui_components.get('progress_container', {}).get('tracker')
    if not progress_tracker:
        return
    
    progress_tracker.error(message)


def _calculate_step_progress(metrics: Dict[str, float]) -> int:
    """Calculate step progress based on loss improvement"""
    train_loss = metrics.get('train_loss', 1.0)
    
    # Loss progression: dari ~2.0 ke ~0.05
    if train_loss > 1.5:
        return 10
    elif train_loss > 1.0:
        return 25
    elif train_loss > 0.5:
        return 50
    elif train_loss > 0.2:
        return 75
    elif train_loss > 0.1:
        return 90
    else:
        return 100


def _calculate_current_progress(metrics: Dict[str, float]) -> int:
    """Calculate current progress based on mAP quality"""
    map_score = metrics.get('map', 0.0)
    
    # mAP progression: 0 to ~0.9
    return min(100, int(map_score * 100))


# One-liner utilities untuk specific operations
update_epoch_progress = lambda ui, epoch, total, msg="": update_training_progress(ui, epoch, total, {'train_loss': 0.5, 'map': epoch/total})
update_batch_progress = lambda ui, batch, total, msg="": update_checkpoint_progress(ui, batch, total, msg or f"Batch {batch}/{total}")
update_save_progress = lambda ui, step, total, msg="": update_checkpoint_progress(ui, step, total, msg or f"Saving {step}/{total}")
update_load_progress = lambda ui, step, total, msg="": update_model_loading_progress(ui, step, total, msg or f"Loading {step}/{total}")
show_training_progress = lambda ui: ui.get('progress_container', {}).get('tracker', {}).get('show', lambda x: None)('training')
hide_training_progress = lambda ui: ui.get('progress_container', {}).get('tracker', {}).get('hide', lambda: None)()