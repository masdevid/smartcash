# File: smartcash/ui/hyperparameters/utils/form_validation.py
# Deskripsi: Validation utilities untuk hyperparameters form

from typing import Dict, Any, List, Tuple
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def validate_hyperparameters_form(ui_components: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validasi form hyperparameters dan return errors jika ada 🔍"""
    
    errors = []
    
    try:
        # Training parameters validation
        epochs = ui_components['epochs'].value
        if epochs < 10 or epochs > 1000:
            errors.append("⚠️ Epochs harus antara 10-1000")
            
        batch_size = ui_components['batch_size'].value
        if batch_size not in [8, 16, 32, 64]:
            errors.append("⚠️ Batch size harus salah satu dari: 8, 16, 32, 64")
            
        learning_rate = ui_components['learning_rate'].value
        if learning_rate <= 0 or learning_rate > 1:
            errors.append("⚠️ Learning rate harus antara 0-1")
            
        # Optimizer validation
        weight_decay = ui_components['weight_decay'].value
        if weight_decay < 0 or weight_decay > 0.1:
            errors.append("⚠️ Weight decay harus antara 0-0.1")
            
        momentum = ui_components['momentum'].value
        if momentum < 0 or momentum >= 1:
            errors.append("⚠️ Momentum harus antara 0-1")
            
        # Scheduler validation
        warmup_epochs = ui_components['warmup_epochs'].value
        if warmup_epochs >= epochs:
            errors.append("⚠️ Warmup epochs tidak boleh >= total epochs")
            
        min_lr = ui_components['min_lr'].value
        if min_lr >= learning_rate:
            errors.append("⚠️ Min learning rate harus < learning rate awal")
            
        # Loss weights validation
        loss_gains = [
            ui_components['box_loss_gain'].value,
            ui_components['cls_loss_gain'].value,
            ui_components['obj_loss_gain'].value
        ]
        
        if any(gain <= 0 for gain in loss_gains):
            errors.append("⚠️ Semua loss gain harus > 0")
            
        # Early stopping validation
        if ui_components['early_stopping_enabled'].value:
            patience = ui_components['patience'].value
            if patience >= epochs:
                errors.append("⚠️ Patience tidak boleh >= total epochs")
                
        # Model inference validation
        conf_thres = ui_components['conf_thres'].value
        iou_thres = ui_components['iou_thres'].value
        
        if conf_thres <= 0 or conf_thres >= 1:
            errors.append("⚠️ Confidence threshold harus antara 0-1")
            
        if iou_thres <= 0 or iou_thres >= 1:
            errors.append("⚠️ IoU threshold harus antara 0-1")
            
        if conf_thres >= iou_thres:
            errors.append("⚠️ Confidence threshold harus < IoU threshold")
            
        max_det = ui_components['max_det'].value
        if max_det < 10 or max_det > 5000:
            errors.append("⚠️ Max detections harus antara 10-5000")
            
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("✅ Validasi form hyperparameters berhasil")
        else:
            logger.warning(f"❌ Form validation gagal: {len(errors)} error(s)")
            
        return is_valid, errors
        
    except Exception as e:
        logger.error(f"❌ Error saat validasi form: {str(e)}")
        return False, [f"Error validasi: {str(e)}"]


def show_validation_errors(errors: List[str]) -> None:
    """Tampilkan validation errors dengan styling 🚨"""
    
    if not errors:
        return
        
    from IPython.display import display, HTML
    
    error_html = "<div style='background: #ffebee; border: 2px solid #f44336; border-radius: 8px; padding: 15px; margin: 10px 0;'>"
    error_html += "<h4 style='color: #f44336; margin: 0 0 10px 0;'>🚨 Validation Errors:</h4>"
    error_html += "<ul style='margin: 0; padding-left: 20px;'>"
    
    for error in errors:
        error_html += f"<li style='color: #d32f2f; margin: 5px 0;'>{error}</li>"
        
    error_html += "</ul></div>"
    
    display(HTML(error_html))


def create_validation_summary() -> Dict[str, Any]:
    """Buat summary widget untuk menampilkan validation status 📊"""
    
    import ipywidgets as widgets
    
    status_widget = widgets.HTML(
        value="<div style='padding: 10px; text-align: center; color: #666;'>Ready untuk validasi...</div>",
        layout=widgets.Layout(
            width='100%',
            border='1px solid #ddd',
            border_radius='4px',
            margin='10px 0'
        )
    )
    
    return {
        'status_widget': status_widget,
        'update_status': lambda is_valid, message: setattr(
            status_widget, 'value',
            f"<div style='padding: 10px; text-align: center; color: {'#4caf50' if is_valid else '#f44336'};'>{message}</div>"
        )
    }