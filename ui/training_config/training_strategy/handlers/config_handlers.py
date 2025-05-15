"""
File: smartcash/ui/training_config/training_strategy/handlers/config_handlers.py
Deskripsi: Handler untuk konfigurasi strategi pelatihan model
"""

from typing import Dict, Any, Optional, List, Callable
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_info_alert, create_status_indicator
from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager

logger = get_logger(__name__)

def get_default_config() -> Dict[str, Any]:
    """
    Mendapatkan konfigurasi default untuk strategi pelatihan.
    
    Returns:
        Dict berisi konfigurasi default
    """
    return {
        # Parameter validasi
        'validation': {
            'frequency': 1,
            'iou_thres': 0.6,
            'conf_thres': 0.001
        },
        
        # Parameter multi-scale training
        'multi_scale': True,
        
        # Konfigurasi tambahan untuk proses training
        'training_utils': {
            'experiment_name': 'efficientnet_b4_training',
            'checkpoint_dir': '/content/runs/train/checkpoints',
            'tensorboard': True,
            'log_metrics_every': 10,
            'visualize_batch_every': 100,
            'gradient_clipping': 1.0,
            'mixed_precision': True,
            'layer_mode': 'single'
        }
    }

def update_config_from_ui(ui_components: Dict[str, Any], config_to_use: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Update konfigurasi dari nilai UI.
    
    Args:
        ui_components: Dictionary komponen UI
        config_to_use: Konfigurasi yang akan digunakan (opsional)
        
    Returns:
        Konfigurasi yang diupdate
    """
    # Dapatkan config manager
    config_manager = get_config_manager()
    
    # Dapatkan konfigurasi yang akan digunakan
    if config_to_use is None:
        # Gunakan config dari ui_components jika ada
        if 'config' in ui_components:
            current_config = ui_components['config']
        else:
            # Jika tidak ada, ambil dari config manager
            current_config = config_manager.get_module_config('training_strategy', {})
    else:
        current_config = config_to_use
    
    # Update parameter validasi
    if 'validation' not in current_config:
        current_config['validation'] = {}
        
    current_config['validation']['frequency'] = ui_components['validation_frequency'].value
    current_config['validation']['iou_thres'] = ui_components['iou_threshold'].value
    current_config['validation']['conf_thres'] = ui_components['conf_threshold'].value
    
    # Update parameter multi-scale training
    current_config['multi_scale'] = ui_components['multi_scale'].value
    
    # Update parameter training_utils
    if 'training_utils' not in current_config:
        current_config['training_utils'] = {}
        
    current_config['training_utils']['experiment_name'] = ui_components['experiment_name'].value
    current_config['training_utils']['checkpoint_dir'] = ui_components['checkpoint_dir'].value
    current_config['training_utils']['tensorboard'] = ui_components['tensorboard'].value
    current_config['training_utils']['log_metrics_every'] = ui_components['log_metrics_every'].value
    current_config['training_utils']['visualize_batch_every'] = ui_components['visualize_batch_every'].value
    current_config['training_utils']['gradient_clipping'] = ui_components['gradient_clipping'].value
    current_config['training_utils']['mixed_precision'] = ui_components['mixed_precision'].value
    current_config['training_utils']['layer_mode'] = ui_components['layer_mode'].value
    
    # Simpan konfigurasi di ui_components
    ui_components['config'] = current_config
    
    return current_config

def update_ui_from_config(ui_components: Dict[str, Any], config_to_use: Optional[Dict[str, Any]] = None) -> None:
    """
    Update UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config_to_use: Konfigurasi yang akan digunakan (opsional)
    """
    # Dapatkan config manager
    config_manager = get_config_manager()
    
    # Dapatkan konfigurasi yang akan digunakan
    if config_to_use is None:
        # Gunakan config dari ui_components jika ada
        if 'config' in ui_components:
            current_config = ui_components['config']
        else:
            # Jika tidak ada, ambil dari config manager
            current_config = config_manager.get_module_config('training_strategy', {})
            # Jika masih tidak ada, gunakan default
            if not current_config:
                current_config = get_default_config()
    else:
        current_config = config_to_use
    
    try:
        # Update parameter validasi
        if 'validation' in current_config:
            if 'frequency' in current_config['validation']:
                ui_components['validation_frequency'].value = current_config['validation']['frequency']
            if 'iou_thres' in current_config['validation']:
                ui_components['iou_threshold'].value = current_config['validation']['iou_thres']
            if 'conf_thres' in current_config['validation']:
                ui_components['conf_threshold'].value = current_config['validation']['conf_thres']
        
        # Update parameter multi-scale training
        if 'multi_scale' in current_config:
            ui_components['multi_scale'].value = current_config['multi_scale']
        
        # Update parameter training_utils
        if 'training_utils' in current_config:
            if 'experiment_name' in current_config['training_utils']:
                ui_components['experiment_name'].value = current_config['training_utils']['experiment_name']
            if 'checkpoint_dir' in current_config['training_utils']:
                ui_components['checkpoint_dir'].value = current_config['training_utils']['checkpoint_dir']
            if 'tensorboard' in current_config['training_utils']:
                ui_components['tensorboard'].value = current_config['training_utils']['tensorboard']
            if 'log_metrics_every' in current_config['training_utils']:
                ui_components['log_metrics_every'].value = current_config['training_utils']['log_metrics_every']
            if 'visualize_batch_every' in current_config['training_utils']:
                ui_components['visualize_batch_every'].value = current_config['training_utils']['visualize_batch_every']
            if 'gradient_clipping' in current_config['training_utils']:
                ui_components['gradient_clipping'].value = current_config['training_utils']['gradient_clipping']
            if 'mixed_precision' in current_config['training_utils']:
                ui_components['mixed_precision'].value = current_config['training_utils']['mixed_precision']
            if 'layer_mode' in current_config['training_utils']:
                ui_components['layer_mode'].value = current_config['training_utils']['layer_mode']
        
        # Simpan konfigurasi di ui_components
        ui_components['config'] = current_config
        
        # Update info strategi pelatihan
        update_training_strategy_info(ui_components)
        
        logger.info(f"{ICONS.get('success', '✅')} UI strategi pelatihan diperbarui dari config")
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error update UI: {e}")

def update_training_strategy_info(ui_components: Dict[str, Any]) -> None:
    """
    Update informasi strategi pelatihan pada UI.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Dapatkan nilai dari UI
        experiment_name = ui_components['experiment_name'].value
        checkpoint_dir = ui_components['checkpoint_dir'].value
        tensorboard = ui_components['tensorboard'].value
        validation_frequency = ui_components['validation_frequency'].value
        iou_threshold = ui_components['iou_threshold'].value
        conf_threshold = ui_components['conf_threshold'].value
        multi_scale = ui_components['multi_scale'].value
        mixed_precision = ui_components['mixed_precision'].value
        layer_mode = ui_components['layer_mode'].value
        
        # Buat informasi HTML
        info_html = f"""
        <h4>Ringkasan Strategi Pelatihan</h4>
        <ul>
            <li><b>Experiment Name:</b> {experiment_name}</li>
            <li><b>Checkpoint Dir:</b> {checkpoint_dir}</li>
            <li><b>TensorBoard:</b> {'Aktif' if tensorboard else 'Nonaktif'}</li>
            <li><b>Validasi Setiap:</b> {validation_frequency} epoch</li>
            <li><b>IoU Threshold:</b> {iou_threshold}</li>
            <li><b>Conf Threshold:</b> {conf_threshold}</li>
            <li><b>Multi-scale Training:</b> {'Aktif' if multi_scale else 'Nonaktif'}</li>
            <li><b>Mixed Precision:</b> {'Aktif' if mixed_precision else 'Nonaktif'}</li>
            <li><b>Layer Mode:</b> {layer_mode}</li>
        </ul>
        <p><i>Catatan: Pastikan parameter pelatihan sesuai dengan kebutuhan dan kapasitas hardware.</i></p>
        """
        
        ui_components['training_strategy_info'].value = info_html
    except Exception as e:
        ui_components['training_strategy_info'].value = f"<p style='color:red'>{ICONS.get('error', '❌')} Error: {str(e)}</p>"
