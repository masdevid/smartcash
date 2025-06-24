# File: smartcash/ui/hyperparameters/handlers/config_extractor.py
# Deskripsi: Extract config dari UI components yang disederhanakan

from typing import Dict, Any
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def extract_hyperparameters_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract konfigurasi hyperparameters dari UI components 📤"""
    
    try:
        config = {
            # Training configuration
            'training': {
                'epochs': ui_components['epochs'].value,
                'batch_size': ui_components['batch_size'].value,
                'learning_rate': ui_components['learning_rate'].value,
                'image_size': ui_components['image_size'].value,
                'device': 'auto',
                'workers': ui_components['workers'].value
            },
            
            # Optimizer configuration
            'optimizer': {
                'type': ui_components['optimizer_type'].value,
                'weight_decay': ui_components['weight_decay'].value,
                'momentum': ui_components['momentum'].value
            },
            
            # Scheduler configuration
            'scheduler': {
                'type': ui_components['scheduler_type'].value,
                'warmup_epochs': ui_components['warmup_epochs'].value,
                'min_lr': ui_components['min_lr'].value
            },
            
            # Loss configuration
            'loss': {
                'box_loss_gain': ui_components['box_loss_gain'].value,
                'cls_loss_gain': ui_components['cls_loss_gain'].value,
                'obj_loss_gain': ui_components['obj_loss_gain'].value
            },
            
            # Early stopping configuration
            'early_stopping': {
                'enabled': ui_components['early_stopping_enabled'].value,
                'patience': ui_components['patience'].value,
                'min_delta': 0.001,  # Fixed value
                'metric': 'mAP_0.5'  # Fixed value
            },
            
            # Checkpoint configuration
            'checkpoint': {
                'save_best': ui_components['save_best'].value,
                'save_interval': ui_components['save_interval'].value,
                'metric': 'mAP_0.5'  # Fixed value
            },
            
            # Model inference configuration
            'model': {
                'conf_thres': ui_components['conf_thres'].value,
                'iou_thres': ui_components['iou_thres'].value,
                'max_det': ui_components['max_det'].value
            },
            
            # Metadata
            'config_version': '2.2',
            'description': 'Simplified hyperparameters untuk SmartCash YOLOv5-EfficientNet',
            'module_name': 'hyperparameters'
        }
        
        logger.info("✅ Config hyperparameters berhasil di-extract dari UI")
        return config
        
    except Exception as e:
        logger.error(f"❌ Gagal extract config hyperparameters: {str(e)}")
        raise