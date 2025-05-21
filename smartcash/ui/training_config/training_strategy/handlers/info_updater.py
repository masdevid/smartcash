"""
File: smartcash/ui/training_config/training_strategy/handlers/info_updater.py
Deskripsi: Fungsi untuk mengupdate panel informasi training strategy
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

logger = get_logger(__name__)

def update_training_strategy_info(ui_components: Dict[str, Any], message: str = None) -> None:
    """
    Update info panel dengan informasi training strategy yang dipilih.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan tambahan yang akan ditampilkan (opsional)
    """
    try:
        if 'training_strategy_info' not in ui_components:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Info panel tidak ditemukan")
            return
            
        info_panel = ui_components['training_strategy_info']
        
        # Get current values from UI
        experiment_name = ui_components.get('experiment_name', widgets.Text()).value
        checkpoint_dir = ui_components.get('checkpoint_dir', widgets.Text()).value
        tensorboard = ui_components.get('tensorboard', widgets.Checkbox()).value
        log_metrics_every = ui_components.get('log_metrics_every', widgets.IntSlider()).value
        visualize_batch_every = ui_components.get('visualize_batch_every', widgets.IntSlider()).value
        gradient_clipping = ui_components.get('gradient_clipping', widgets.FloatSlider()).value
        mixed_precision = ui_components.get('mixed_precision', widgets.Checkbox()).value
        layer_mode = ui_components.get('layer_mode', widgets.RadioButtons()).value
        
        # Validasi
        validation_frequency = ui_components.get('validation_frequency', widgets.IntSlider()).value
        iou_threshold = ui_components.get('iou_threshold', widgets.FloatSlider()).value
        conf_threshold = ui_components.get('conf_threshold', widgets.FloatSlider()).value
        
        # Multi-scale
        multi_scale = ui_components.get('multi_scale', widgets.Checkbox()).value
                
        # Update info panel dengan informasi training strategy
        with info_panel:
            clear_output(wait=True)
            html_content = f"""
            <h3>{ICONS.get('info', 'ℹ️')} Informasi Strategi Pelatihan</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; grid-gap: 10px;">
                <div>
                    <h4>Parameter Utilitas</h4>
                    <ul>
                        <li><b>Experiment Name:</b> {experiment_name}</li>
                        <li><b>Checkpoint Dir:</b> {checkpoint_dir}</li>
                        <li><b>TensorBoard:</b> {'Aktif' if tensorboard else 'Nonaktif'}</li>
                        <li><b>Log Metrics Every:</b> {log_metrics_every} batch</li>
                        <li><b>Visualize Batch Every:</b> {visualize_batch_every} batch</li>
                        <li><b>Gradient Clipping:</b> {gradient_clipping}</li>
                        <li><b>Mixed Precision:</b> {'Aktif' if mixed_precision else 'Nonaktif'}</li>
                        <li><b>Layer Mode:</b> {layer_mode}</li>
                    </ul>
                </div>
                <div>
                    <h4>Validasi & Multi-scale</h4>
                    <ul>
                        <li><b>Validation Frequency:</b> {validation_frequency} epoch</li>
                        <li><b>IoU Threshold:</b> {iou_threshold}</li>
                        <li><b>Conf Threshold:</b> {conf_threshold}</li>
                        <li><b>Multi-scale Training:</b> {'Aktif' if multi_scale else 'Nonaktif'}</li>
                    </ul>
                </div>
            </div>
            
            <div style="margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-left: 4px solid #17a2b8;">
                <p><b>{ICONS.get('info', 'ℹ️')} Catatan:</b> Konfigurasi strategi pelatihan ini akan digunakan untuk melatih model YOLOv5 dengan EfficientNet backbone.</p>
            </div>
            """
            
            # Add message if provided
            if message:
                html_content += f"""
                <div style="margin-top: 10px; padding: 10px; background-color: #d4edda; border-left: 4px solid #28a745; border-radius: 4px;">
                    <p>{ICONS.get('success', '✅')} {message}</p>
                </div>
                """
                
            display(widgets.HTML(html_content))
        
        logger.info(f"{ICONS.get('success', '✅')} Info panel berhasil diupdate")
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat update info panel: {str(e)}")
        if 'training_strategy_info' in ui_components:
            with ui_components['training_strategy_info']:
                clear_output(wait=True)
                display(widgets.HTML(f"<div style='color:red'>{ICONS.get('error', '❌')} Error saat update info panel: {str(e)}</div>"))