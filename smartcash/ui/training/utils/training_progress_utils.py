"""
File: smartcash/ui/training/utils/training_progress_utils.py
Deskripsi: Enhanced progress utilities dengan detailed step-by-step progress tracking
"""

from typing import Dict, Any


def update_training_progress(ui_components: Dict[str, Any], epoch: int, total_epochs: int, metrics: Dict[str, float]):
    """Update training progress dengan detailed 3-bar system"""
    progress_tracker = ui_components.get('progress_container', {}).get('tracker')
    if not progress_tracker:
        return
    
    # Overall progress (epochs) dengan enhanced messaging
    overall_progress = int((epoch / total_epochs) * 100)
    epoch_msg = f"Epoch {epoch+1}/{total_epochs} ({overall_progress}%)"
    progress_tracker.update('overall', overall_progress, epoch_msg, color='success' if overall_progress > 80 else 'info')
    
    # Step progress (loss improvement) dengan adaptive coloring
    step_progress = _calculate_step_progress(metrics)
    loss_msg = f"Loss: {metrics.get('train_loss', 0):.4f} | Val: {metrics.get('val_loss', 0):.4f}"
    step_color = 'success' if step_progress > 75 else 'warning' if step_progress > 50 else 'info'
    progress_tracker.update('step', step_progress, loss_msg, color=step_color)
    
    # Current operation (metrics quality) dengan performance indicators
    current_progress = _calculate_current_progress(metrics)
    metrics_msg = f"mAP: {metrics.get('map', 0):.4f} | F1: {metrics.get('f1', 0):.4f}"
    current_color = 'success' if current_progress > 70 else 'warning' if current_progress > 40 else 'info'
    progress_tracker.update('current', current_progress, metrics_msg, color=current_color)


def update_checkpoint_progress(ui_components: Dict[str, Any], current: int, total: int, message: str):
    """Update checkpoint operation dengan detailed steps"""
    progress_tracker = ui_components.get('progress_container', {}).get('tracker')
    if not progress_tracker:
        return
    
    checkpoint_progress = int((current / total) * 100) if total > 0 else 0
    
    # Enhanced checkpoint messaging
    step_messages = {
        0: "🔄 Preparing checkpoint...",
        1: "📦 Collecting model state...", 
        2: "💾 Writing to disk...",
        3: "✅ Checkpoint saved!"
    }
    
    enhanced_message = step_messages.get(current, message)
    color = 'success' if current == total else 'info'
    
    progress_tracker.update('current', checkpoint_progress, enhanced_message, color=color)


def update_model_loading_progress(ui_components: Dict[str, Any], current: int, total: int, message: str):
    """Update model loading dengan detailed initialization steps"""
    progress_tracker = ui_components.get('progress_container', {}).get('tracker')
    if not progress_tracker:
        return
    
    loading_progress = int((current / total) * 100) if total > 0 else 0
    
    # Enhanced model loading steps
    step_details = {
        1: "📋 Parsing configuration and validating parameters...",
        2: "🧠 Building EfficientNet-B4 architecture...",
        3: "💾 Setting up checkpoint manager...",
        4: "🚀 Initializing training services..."
    }
    
    detailed_message = step_details.get(current, message)
    color = 'success' if current == total else 'info'
    
    progress_tracker.update('step', loading_progress, detailed_message, color=color)


def update_validation_progress(ui_components: Dict[str, Any], current: int, total: int, message: str):
    """Update validation progress dengan comprehensive checks"""
    progress_tracker = ui_components.get('progress_container', {}).get('tracker')
    if not progress_tracker:
        return
    
    validation_progress = int((current / total) * 100) if total > 0 else 0
    
    # Validation step details
    validation_steps = {
        1: "🔍 Checking model manager status...",
        2: "📦 Validating pre-trained weights...",
        3: "⚙️ Verifying training configuration...", 
        4: "🖥️ Checking GPU availability...",
        5: "📊 Validating detection layers..."
    }
    
    step_message = validation_steps.get(current, message)
    color = 'success' if current == total else 'warning' if 'error' in message.lower() else 'info'
    
    progress_tracker.update('overall', validation_progress, step_message, color=color)


def show_operation_progress(ui_components: Dict[str, Any], operation: str):
    """Show progress container untuk specific operation"""
    # Get progress container dan tracker
    progress_container = ui_components.get('progress_container')
    progress_tracker = ui_components.get('progress_container', {}).get('tracker')
    
    # Return jika tidak ada tracker
    if not progress_tracker:
        return
    
    # Make progress container visible
    if progress_container and hasattr(progress_container, 'layout'):
        progress_container.layout.display = 'block'  # Tampilkan container
    
    # Operation-specific messaging
    operation_messages = {
        'training': " Memulai training model...",
        'validation': " Memvalidasi model...",
        'checkpoint': " Menyimpan checkpoint...",
        'loading': " Memuat komponen model...",
        'evaluation': " Mengevaluasi performa model..."
    }
    
    message = operation_messages.get(operation, f"Operation: {operation}")
    progress_tracker.show(message)


def complete_operation_progress(ui_components: Dict[str, Any], message: str = "Operation completed!", show_summary: bool = True):
    """Complete operation dengan summary"""
    progress_tracker = ui_components.get('progress_container', {}).get('tracker')
    if not progress_tracker:
        return
    
    if show_summary:
        # Get final metrics dari tracker jika ada
        final_metrics = _extract_final_metrics(progress_tracker)
        summary_msg = f"{message} {final_metrics}" if final_metrics else message
        progress_tracker.complete(summary_msg)
    else:
        progress_tracker.complete(message)


def error_operation_progress(ui_components: Dict[str, Any], error_message: str, show_details: bool = True):
    """Set error state dengan detailed error info"""
    progress_tracker = ui_components.get('progress_container', {}).get('tracker')
    if not progress_tracker:
        return
    
    if show_details:
        detailed_error = f"❌ {error_message} - Check logs untuk detail"
        progress_tracker.error(detailed_error)
    else:
        progress_tracker.error(error_message)


def reset_all_progress(ui_components: Dict[str, Any], initial_message: str = "Ready to start..."):
    """Reset progress dengan enhanced initial state"""
    progress_tracker = ui_components.get('progress_container', {}).get('tracker')
    if not progress_tracker:
        return
    
    # Reset dengan informative messages
    progress_tracker.update('overall', 0, f"🎯 {initial_message}")
    progress_tracker.update('step', 0, "⏳ Awaiting operation...")
    progress_tracker.update('current', 0, "💤 Standby mode")


def _calculate_step_progress(metrics: Dict[str, float]) -> int:
    """Enhanced step progress calculation based on loss convergence"""
    train_loss = metrics.get('train_loss', 1.0)
    val_loss = metrics.get('val_loss', 1.0)
    
    # Combined loss assessment
    avg_loss = (train_loss + val_loss) / 2
    
    # Progressive improvement curve
    if avg_loss > 2.0:
        return 5
    elif avg_loss > 1.5:
        return 15
    elif avg_loss > 1.0:
        return 30
    elif avg_loss > 0.5:
        return 50
    elif avg_loss > 0.2:
        return 75
    elif avg_loss > 0.1:
        return 90
    else:
        return 100


def _calculate_current_progress(metrics: Dict[str, float]) -> int:
    """Enhanced current progress based on multiple metrics"""
    map_score = metrics.get('map', 0.0)
    f1_score = metrics.get('f1', 0.0)
    precision = metrics.get('precision', 0.0)
    recall = metrics.get('recall', 0.0)
    
    # Weighted average of performance metrics
    weights = {'map': 0.4, 'f1': 0.3, 'precision': 0.15, 'recall': 0.15}
    weighted_score = (
        map_score * weights['map'] + 
        f1_score * weights['f1'] + 
        precision * weights['precision'] + 
        recall * weights['recall']
    )
    
    return min(100, int(weighted_score * 100))


def _extract_final_metrics(progress_tracker) -> str:
    """Extract final metrics summary dari progress tracker"""
    try:
        # Attempt to get last displayed metrics
        if hasattr(progress_tracker, '_last_metrics'):
            metrics = progress_tracker._last_metrics
            return f"(mAP: {metrics.get('map', 0):.3f}, Loss: {metrics.get('train_loss', 0):.3f})"
    except Exception:
        pass
    return ""


# One-liner utilities untuk specific operations
update_epoch_progress = lambda ui, epoch, total, metrics: update_training_progress(ui, epoch, total, metrics)
update_batch_progress = lambda ui, batch, total: update_checkpoint_progress(ui, batch, total, f"Processing batch {batch}/{total}")
update_save_progress = lambda ui, step, total: update_checkpoint_progress(ui, step, total, f"Saving checkpoint {step}/{total}")
update_load_progress = lambda ui, step, total: update_model_loading_progress(ui, step, total, f"Loading component {step}/{total}")
show_training_progress = lambda ui: show_operation_progress(ui, 'training')
show_validation_progress = lambda ui: show_operation_progress(ui, 'validation')
hide_progress = lambda ui: ui.get('progress_container', {}).get('tracker', {}).get('hide', lambda: None)()
complete_training = lambda ui, msg="Training completed!": complete_operation_progress(ui, msg, show_summary=True)
error_training = lambda ui, msg="Training failed": error_operation_progress(ui, msg, show_details=True)

# Complete all progress bars dengan satu panggilan
complete_all_progress = lambda ui, msg="All operations completed!": (
    progress_tracker := ui.get('progress_container', {}).get('tracker'),
    progress_tracker.update('overall', 100, "✅ Complete", color='success'),
    progress_tracker.update('step', 100, "✅ Complete", color='success'),
    progress_tracker.update('current', 100, msg, color='success'),
    progress_tracker.complete(msg)
)[-1] if ui.get('progress_container', {}).get('tracker') else None

# Set error state untuk semua progress bars
error_all_progress = lambda ui, msg="Operation failed": (
    progress_tracker := ui.get('progress_container', {}).get('tracker'),
    progress_tracker.update('overall', 0, "❌ Failed", color='danger'),
    progress_tracker.update('step', 0, "❌ Failed", color='danger'),
    progress_tracker.update('current', 0, msg, color='danger'),
    progress_tracker.error(msg)
)[-1] if ui.get('progress_container', {}).get('tracker') else None