# File: smartcash/ui/hyperparameters/components/ui_form.py
# Deskripsi: Form UI yang disederhanakan untuk hyperparameters dengan fallback handling

import ipywidgets as widgets
import sys
import traceback
from typing import Dict, Any, Optional, Tuple, Callable

from smartcash.common.logger import get_logger
from smartcash.ui.components import (
    create_slider, 
    create_dropdown, 
    create_checkbox, 
    create_log_slider
)
from smartcash.ui.utils.fallback_utils import FallbackConfig, create_fallback_ui

logger = get_logger(__name__)

def _create_fallback_ui(
    error_message: str, 
    exc_info: Optional[Tuple] = None, 
    show_traceback: bool = True, 
    retry_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """Create fallback UI with proper error handling for hyperparameters module
    
    Args:
        error_message: Pesan error yang akan ditampilkan
        exc_info: Optional exception info tuple (type, value, traceback)
        show_traceback: Apakah menampilkan traceback
        retry_callback: Optional callback function untuk tombol retry
        
    Returns:
        Dictionary berisi komponen UI fallback untuk modul hyperparameters
    """
    # Format traceback jika ada
    tb_msg = ""
    if exc_info and show_traceback:
        try:
            tb_msg = "".join(traceback.format_exception(*exc_info))
        except Exception as e:
            tb_msg = f"Error getting traceback: {str(e)}"
    
    # Buat konfigurasi fallback
    config = FallbackConfig(
        title="⚠️ Error in Hyperparameters",
        message=error_message,
        traceback=tb_msg,
        module_name='hyperparameters',
        show_traceback=show_traceback,
        show_retry=retry_callback is not None,
        retry_callback=retry_callback,
        container_style={
            'border': '1px solid #f5c6cb',
            'border_radius': '8px',
            'padding': '15px',
            'margin': '10px 0',
            'background': '#f8d7da',
            'color': '#721c24'
        }
    )
    
    # Buat widget fallback
    error_widget = create_fallback_ui(
        error_message=config.message,
        module_name=config.module_name,
        exc_info=exc_info if show_traceback else None,
        config=config
    )
    
    return {'main_layout': error_widget}


def create_hyperparameters_form(config: Dict[str, Any]) -> Dict[str, Any]:
    """Buat form UI untuk hyperparameters yang disederhanakan 🎛️"""
    try:
        # Validasi config
        if not isinstance(config, dict):
            config = {}
            
        # Training parameters
        training = config.get('training', {})
        epochs = create_slider('epochs', training.get('epochs', 100), 50, 300, 10, 
                            description='🏋️ Epochs', tooltip='Jumlah epoch training')
        batch_size = create_dropdown('batch_size', training.get('batch_size', 16), [8, 16, 32, 64],
                                    description='📦 Batch Size', tooltip='Ukuran batch untuk training')
        learning_rate = create_log_slider('learning_rate', training.get('learning_rate', 0.01), 
                                        1e-4, 1e-1, 0.001, description='📈 Learning Rate', 
                                        tooltip='Learning rate awal')
        image_size = create_dropdown('image_size', training.get('image_size', 640), [416, 512, 608, 640],
                                    description='🖼️ Image Size', tooltip='Resolusi gambar input')
        workers = create_slider('workers', training.get('workers', 4), 0, 8, 1,
                            description='👷 Workers', tooltip='Jumlah worker untuk data loading')
        
        # Optimizer parameters
        optimizer = config.get('optimizer', {})
        optimizer_type = create_dropdown('optimizer_type', optimizer.get('type', 'SGD'), ['SGD', 'Adam'],
                                        description='⚙️ Optimizer', tooltip='Jenis optimizer')
        weight_decay = create_log_slider('weight_decay', optimizer.get('weight_decay', 0.0005), 
                                        1e-6, 1e-2, 0.0001, description='🔄 Weight Decay',
                                        tooltip='L2 regularization strength')
        momentum = create_slider('momentum', optimizer.get('momentum', 0.937), 0.8, 0.99, 0.01,
                                description='🚀 Momentum', tooltip='Momentum untuk SGD optimizer')
        
        # Scheduler parameters
        scheduler = config.get('scheduler', {})
        scheduler_type = create_dropdown('scheduler_type', scheduler.get('type', 'cosine'), 
                                        ['cosine', 'step', 'plateau'],
                                        description='📊 Scheduler', tooltip='Jenis learning rate scheduler')
        warmup_epochs = create_slider('warmup_epochs', scheduler.get('warmup_epochs', 3), 0, 10, 1,
                                    description='🔥 Warmup Epochs', tooltip='Epochs untuk warmup')
        min_lr = create_log_slider('min_lr', scheduler.get('min_lr', 0.0001), 1e-6, 1e-3, 0.00001,
                                description='📉 Min LR', tooltip='Learning rate minimum')
        
        # Loss parameters
        loss = config.get('loss', {})
        box_loss_gain = create_slider('box_loss_gain', loss.get('box_loss_gain', 0.05), 0.01, 0.5, 0.01,
                                    description='📦 Box Loss', tooltip='Bobot untuk box loss')
        cls_loss_gain = create_slider('cls_loss_gain', loss.get('cls_loss_gain', 0.5), 0.1, 1.0, 0.1,
                                    description='🏷️ Class Loss', tooltip='Bobot untuk class loss')
        obj_loss_gain = create_slider('obj_loss_gain', loss.get('obj_loss_gain', 1.0), 0.1, 2.0, 0.1,
                                    description='🎯 Obj Loss', tooltip='Bobot untuk objectness loss')
        
        # Early stopping & checkpointing
        early_stopping = config.get('early_stopping', {})
        early_stopping_enabled = create_checkbox('early_stopping_enabled', 
                                            early_stopping.get('enabled', True),
                                            description='⏹️ Early Stopping',
                                            tooltip='Aktifkan early stopping')
        patience = create_slider('patience', early_stopping.get('patience', 50), 10, 200, 5,
                            description='⏳ Patience', tooltip='Jumlah epoch tanpa perbaikan')
        save_best = create_checkbox('save_best', True, 
                                description='💾 Save Best',
                                tooltip='Simpan model terbaik')
        save_interval = create_slider('save_interval', 10, 1, 50, 1,
                                    description='📅 Save Interval',
                                    tooltip='Interval penyimpanan model (epoch)')
        
        # Model inference parameters
        inference = config.get('inference', {})
        conf_thres = create_slider('conf_thres', inference.get('conf_thres', 0.25), 0.01, 0.99, 0.01,
                                description='🎯 Conf. Threshold', tooltip='Ambang keyakinan deteksi')
        iou_thres = create_slider('iou_thres', inference.get('iou_thres', 0.45), 0.1, 0.9, 0.05,
                                description='📏 IoU Threshold', tooltip='Ambang IoU untuk NMS')
        max_det = create_slider('max_det', inference.get('max_det', 300), 10, 500, 10,
                            description='🔢 Max Detections', tooltip='Max deteksi per gambar')
        
        logger.info("✅ Form hyperparameters berhasil dibuat dengan parameter essential")
        
        return {
            # Training components
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'image_size': image_size,
            'workers': workers,
            
            # Optimizer components
            'optimizer_type': optimizer_type,
            'weight_decay': weight_decay,
            'momentum': momentum,
            
            # Scheduler components
            'scheduler_type': scheduler_type,
            'warmup_epochs': warmup_epochs,
            'min_lr': min_lr,
            
            # Loss components
            'box_loss_gain': box_loss_gain,
            'cls_loss_gain': cls_loss_gain,
            'obj_loss_gain': obj_loss_gain,
            
            # Early stopping & checkpoint components
            'early_stopping_enabled': early_stopping_enabled,
            'patience': patience,
            'save_best': save_best,
            'save_interval': save_interval,
            
            # Model inference components
            'conf_thres': conf_thres,
            'iou_thres': iou_thres,
            'max_det': max_det
        }
        
    except Exception as e:
        logger.error(f"❌ Gagal membuat form hyperparameters: {str(e)}", exc_info=True)
        return _create_fallback_ui(
            f"Gagal memuat form hyperparameters: {str(e)}",
            exc_info=sys.exc_info(),
            show_traceback=True,
            retry_callback=lambda: create_hyperparameters_form(config)
        )