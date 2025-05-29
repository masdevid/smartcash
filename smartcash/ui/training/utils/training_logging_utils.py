"""
File: smartcash/ui/training/utils/training_logging_utils.py
Deskripsi: Logging utilities untuk training dengan colored metrics dan context-aware messages
"""

from typing import Dict, Any
from IPython.display import display, HTML


def log_epoch_metrics(ui_components: Dict[str, Any], epoch: int, metrics: Dict[str, float]):
    """Log epoch metrics dengan colored formatting"""
    log_output = ui_components.get('log_output')
    metrics_output = ui_components.get('metrics_output')
    
    if not log_output:
        return
    
    # Format metrics dengan colors
    colored_metrics = _format_colored_metrics(metrics)
    epoch_msg = f"🚀 <b>Epoch {epoch + 1}</b> | {colored_metrics}"
    
    # Log ke training log
    with log_output:
        display(HTML(f'<div style="margin: 2px 0; font-family: monospace;">{epoch_msg}</div>'))
    
    # Update detailed metrics display
    if metrics_output:
        _update_metrics_display(metrics_output, epoch, metrics)


def log_training_start(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Log training start dengan configuration summary"""
    log_output = ui_components.get('log_output')
    if not log_output:
        return
    
    training_config = config.get('training', {})
    model_type = training_config.get('model_type', 'efficient_optimized')
    
    start_msg = f"""
    <div style="padding: 10px; background: #e8f5e8; border-radius: 6px; margin: 5px 0;">
        <h4 style="margin: 0 0 10px 0; color: #2e7d32;">🚀 Training Started: {model_type.upper()}</h4>
        <ul style="margin: 0; padding-left: 20px;">
            <li><b>Epochs:</b> <span style="color: #1976d2;">{training_config.get('epochs', 100)}</span></li>
            <li><b>Batch Size:</b> <span style="color: #1976d2;">{training_config.get('batch_size', 16)}</span></li>
            <li><b>Learning Rate:</b> <span style="color: #1976d2;">{training_config.get('learning_rate', 0.001)}</span></li>
            <li><b>Optimizer:</b> <span style="color: #1976d2;">{training_config.get('optimizer', 'Adam')}</span></li>
        </ul>
    </div>
    """
    
    with log_output:
        display(HTML(start_msg))


def log_training_complete(ui_components: Dict[str, Any], success: bool, final_metrics: Dict[str, float] = None):
    """Log training completion dengan final summary"""
    log_output = ui_components.get('log_output')
    if not log_output:
        return
    
    if success and final_metrics:
        colored_final = _format_colored_metrics(final_metrics)
        complete_msg = f"""
        <div style="padding: 10px; background: #e8f5e8; border-radius: 6px; margin: 5px 0;">
            <h4 style="margin: 0 0 10px 0; color: #2e7d32;">✅ Training Completed Successfully!</h4>
            <p style="margin: 5px 0;"><b>Final Metrics:</b> {colored_final}</p>
        </div>
        """
    else:
        complete_msg = f"""
        <div style="padding: 10px; background: #ffeaea; border-radius: 6px; margin: 5px 0;">
            <h4 style="margin: 0; color: #d32f2f;">❌ Training {'Failed' if not success else 'Stopped'}</h4>
        </div>
        """
    
    with log_output:
        display(HTML(complete_msg))


def log_checkpoint_save(ui_components: Dict[str, Any], checkpoint_path: str, epoch: int):
    """Log checkpoint save operation"""
    log_output = ui_components.get('log_output')
    if not log_output:
        return
    
    checkpoint_msg = f'💾 <b style="color: #7b1fa2;">Checkpoint saved:</b> <code>{checkpoint_path}</code> (Epoch {epoch + 1})'
    
    with log_output:
        display(HTML(f'<div style="margin: 2px 0; font-family: monospace;">{checkpoint_msg}</div>'))


def log_model_validation(ui_components: Dict[str, Any], validation_results: Dict[str, Any]):
    """Log model validation results"""
    log_output = ui_components.get('log_output')
    if not log_output:
        return
    
    model_ready = validation_results.get('model_status', {}).get('model_ready', False)
    status_icon = "✅" if model_ready else "❌"
    status_color = "#2e7d32" if model_ready else "#d32f2f"
    
    validation_msg = f"""
    <div style="padding: 8px; background: #f5f5f5; border-radius: 4px; margin: 5px 0;">
        <span style="color: {status_color}; font-weight: bold;">{status_icon} Model Validation</span>
        - Ready: {model_ready}
    </div>
    """
    
    with log_output:
        display(HTML(validation_msg))


def log_gpu_cleanup(ui_components: Dict[str, Any], cleanup_results: Dict[str, Any]):
    """Log GPU cleanup results"""
    log_output = ui_components.get('log_output')
    if not log_output:
        return
    
    models_cleaned = len(cleanup_results.get('model_cleanup', {}).get('models_cleaned', []))
    cache_cleared = cleanup_results.get('cache_cleanup', {}).get('cache_cleared', False)
    
    cleanup_msg = f"""
    <div style="padding: 8px; background: #fff3e0; border-radius: 4px; margin: 5px 0;">
        🧹 <b style="color: #ef6c00;">GPU Cleanup:</b> 
        {models_cleaned} models freed, Cache: {'✅' if cache_cleared else '❌'}
    </div>
    """
    
    with log_output:
        display(HTML(cleanup_msg))


def _format_colored_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dengan color coding"""
    formatted_parts = []
    
    # Train Loss (red if high, green if low)
    train_loss = metrics.get('train_loss', 0)
    loss_color = "#d32f2f" if train_loss > 0.5 else "#ff9800" if train_loss > 0.2 else "#2e7d32"
    formatted_parts.append(f'<span style="color: {loss_color};">Loss: {train_loss:.4f}</span>')
    
    # Validation Loss
    val_loss = metrics.get('val_loss', 0)
    val_color = "#d32f2f" if val_loss > 0.5 else "#ff9800" if val_loss > 0.2 else "#2e7d32"
    formatted_parts.append(f'<span style="color: {val_color};">Val: {val_loss:.4f}</span>')
    
    # mAP (green if high, red if low)
    map_score = metrics.get('map', 0)
    map_color = "#2e7d32" if map_score > 0.7 else "#ff9800" if map_score > 0.4 else "#d32f2f"
    formatted_parts.append(f'<span style="color: {map_color};">mAP: {map_score:.4f}</span>')
    
    # F1 Score
    f1_score = metrics.get('f1', 0)
    f1_color = "#2e7d32" if f1_score > 0.7 else "#ff9800" if f1_score > 0.4 else "#d32f2f"
    formatted_parts.append(f'<span style="color: {f1_color};">F1: {f1_score:.4f}</span>')
    
    # Learning Rate (if available)
    if 'lr' in metrics:
        lr = metrics['lr']
        formatted_parts.append(f'<span style="color: #7b1fa2;">LR: {lr:.6f}</span>')
    
    return " | ".join(formatted_parts)


def _update_metrics_display(metrics_output, epoch: int, metrics: Dict[str, float]):
    """Update detailed metrics display"""
    with metrics_output:
        # Create detailed metrics table
        metrics_html = f"""
        <div style="padding: 10px; border-radius: 6px; background: #f8f9fa;">
            <h4 style="margin: 0 0 10px 0; color: #495057;">📊 Training Metrics - Epoch {epoch + 1}</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background: #e9ecef;">
                    <th style="padding: 8px; text-align: left; border: 1px solid #dee2e6;">Metric</th>
                    <th style="padding: 8px; text-align: right; border: 1px solid #dee2e6;">Value</th>
                    <th style="padding: 8px; text-align: center; border: 1px solid #dee2e6;">Status</th>
                </tr>
        """
        
        # Add metrics rows
        metric_configs = [
            ('Train Loss', 'train_loss', lambda x: '🟢' if x < 0.2 else '🟡' if x < 0.5 else '🔴'),
            ('Val Loss', 'val_loss', lambda x: '🟢' if x < 0.2 else '🟡' if x < 0.5 else '🔴'),
            ('mAP Score', 'map', lambda x: '🟢' if x > 0.7 else '🟡' if x > 0.4 else '🔴'),
            ('F1 Score', 'f1', lambda x: '🟢' if x > 0.7 else '🟡' if x > 0.4 else '🔴'),
            ('Precision', 'precision', lambda x: '🟢' if x > 0.8 else '🟡' if x > 0.6 else '🔴'),
            ('Recall', 'recall', lambda x: '🟢' if x > 0.8 else '🟡' if x > 0.6 else '🔴')
        ]
        
        for metric_name, metric_key, status_fn in metric_configs:
            if metric_key in metrics:
                value = metrics[metric_key]
                status = status_fn(value)
                metrics_html += f"""
                <tr>
                    <td style="padding: 6px; border: 1px solid #dee2e6;">{metric_name}</td>
                    <td style="padding: 6px; text-align: right; border: 1px solid #dee2e6; font-family: monospace;">{value:.4f}</td>
                    <td style="padding: 6px; text-align: center; border: 1px solid #dee2e6;">{status}</td>
                </tr>
                """
        
        metrics_html += """
            </table>
        </div>
        """
        
        display(HTML(metrics_html))


# One-liner utilities untuk quick logging
log_info = lambda ui, msg: ui.get('log_output') and display(HTML(f'<div style="color: #007bff; margin: 2px 0;">ℹ️ {msg}</div>')) if ui.get('log_output') else None
log_success = lambda ui, msg: ui.get('log_output') and display(HTML(f'<div style="color: #28a745; margin: 2px 0;">✅ {msg}</div>')) if ui.get('log_output') else None
log_warning = lambda ui, msg: ui.get('log_output') and display(HTML(f'<div style="color: #ffc107; margin: 2px 0;">⚠️ {msg}</div>')) if ui.get('log_output') else None
log_error = lambda ui, msg: ui.get('log_output') and display(HTML(f'<div style="color: #dc3545; margin: 2px 0;">❌ {msg}</div>')) if ui.get('log_output') else None
clear_logs = lambda ui: ui.get('log_output') and ui['log_output'].clear_output(wait=True)