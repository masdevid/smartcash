"""
File: smartcash/ui/training_config/training_strategy/components/info_panel_components.py
Deskripsi: Komponen panel informasi untuk konfigurasi strategi pelatihan model
"""

from typing import Dict, Any, Optional, Tuple
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def create_training_strategy_info_panel() -> Tuple[widgets.HTML, Any]:
    """
    Membuat panel informasi untuk strategi pelatihan.
    
    Returns:
        Tuple berisi HTML widget dan fungsi update
    """
    info_panel = widgets.HTML(
        value="<p style='padding: 10px; background-color: #f8f9fa; border-left: 4px solid #17a2b8;'><b>ℹ️ Info:</b> Konfigurasi strategi pelatihan dasar untuk model YOLOv5 dengan EfficientNet backbone.</p>"
    )
    
    def update_training_strategy_info(ui_components: Optional[Dict[str, Any]] = None):
        """
        Update informasi strategi pelatihan di panel info.
        
        Args:
            ui_components: Komponen UI
        """
        if not ui_components:
            info_panel.value = f"<p style='padding: 10px; background-color: #f8f9fa; border-left: 4px solid #17a2b8;'><b>{ICONS.get('info', 'ℹ️')} Info:</b> Tidak ada informasi yang tersedia.</p>"
            return
        
        try:
            # Dapatkan nilai dari komponen UI
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
            
            # Buat konten HTML
            html_content = f"""
            <div style='padding: 10px; background-color: #f8f9fa; border-left: 4px solid #17a2b8;'>
                <h4 style='margin-top: 0;'>{ICONS.get('info', 'ℹ️')} Informasi Strategi Pelatihan</h4>
                <div style='display: grid; grid-template-columns: 1fr 1fr; grid-gap: 10px;'>
                    <div>
                        <h5>Parameter Utilitas</h5>
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
                        <h5>Validasi & Multi-scale</h5>
                        <ul>
                            <li><b>Validation Frequency:</b> {validation_frequency} epoch</li>
                            <li><b>IoU Threshold:</b> {iou_threshold}</li>
                            <li><b>Conf Threshold:</b> {conf_threshold}</li>
                            <li><b>Multi-scale Training:</b> {'Aktif' if multi_scale else 'Nonaktif'}</li>
                        </ul>
                    </div>
                </div>
                <p style='margin-top: 10px;'><b>Catatan:</b> Konfigurasi strategi pelatihan ini akan digunakan untuk melatih model YOLOv5 dengan EfficientNet backbone.</p>
            </div>
            """
            
            info_panel.value = html_content
        except Exception as e:
            logger.error(f"{ICONS.get('error', '❌')} Error update info panel: {str(e)}")
            info_panel.value = f"<p style='padding: 10px; background-color: #f8f9fa; border-left: 4px solid #dc3545;'><b>{ICONS.get('error', '❌')} Error:</b> Terjadi error saat memperbarui informasi: {str(e)}</p>"
    
    return info_panel, update_training_strategy_info
