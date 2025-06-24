# File: smartcash/ui/hyperparameters/handlers/config_updater.py
# Deskripsi: Update UI components dari config yang disederhanakan

from typing import Dict, Any
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def update_hyperparameters_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Update UI components dari config hyperparameters 📥"""
    
    try:
        # Training parameters
        training = config.get('training', {})
        ui_components['epochs'].value = training.get('epochs', 100)
        ui_components['batch_size'].value = training.get('batch_size', 16)
        ui_components['learning_rate'].value = training.get('learning_rate', 0.01)
        ui_components['image_size'].value = training.get('image_size', 640)
        ui_components['workers'].value = training.get('workers', 4)
        
        # Optimizer parameters
        optimizer = config.get('optimizer', {})
        ui_components['optimizer_type'].value = optimizer.get('type', 'SGD')
        ui_components['weight_decay'].value = optimizer.get('weight_decay', 0.0005)
        ui_components['momentum'].value = optimizer.get('momentum', 0.937)
        
        # Scheduler parameters
        scheduler = config.get('scheduler', {})
        ui_components['scheduler_type'].value = scheduler.get('type', 'cosine')
        ui_components['warmup_epochs'].value = scheduler.get('warmup_epochs', 3)
        ui_components['min_lr'].value = scheduler.get('min_lr', 0.0001)
        
        # Loss parameters
        loss = config.get('loss', {})
        ui_components['box_loss_gain'].value = loss.get('box_loss_gain', 0.05)
        ui_components['cls_loss_gain'].value = loss.get('cls_loss_gain', 0.5)
        ui_components['obj_loss_gain'].value = loss.get('obj_loss_gain', 1.0)
        
        # Early stopping parameters
        early_stopping = config.get('early_stopping', {})
        ui_components['early_stopping_enabled'].value = early_stopping.get('enabled', True)
        ui_components['patience'].value = early_stopping.get('patience', 15)
        
        # Checkpoint parameters
        checkpoint = config.get('checkpoint', {})
        ui_components['save_best'].value = checkpoint.get('save_best', True)
        ui_components['save_interval'].value = checkpoint.get('save_interval', 10)
        
        # Model inference parameters
        model = config.get('model', {})
        ui_components['conf_thres'].value = model.get('conf_thres', 0.25)
        ui_components['iou_thres'].value = model.get('iou_thres', 0.45)
        ui_components['max_det'].value = model.get('max_det', 1000)
        
        logger.info("✅ UI hyperparameters berhasil di-update dari config")
        
    except Exception as e:
        logger.error(f"❌ Gagal update UI hyperparameters: {str(e)}")
        raise