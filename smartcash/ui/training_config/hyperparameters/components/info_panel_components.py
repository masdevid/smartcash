"""
File: smartcash/ui/training_config/hyperparameters/components/info_panel_components.py
Deskripsi: Komponen panel informasi untuk konfigurasi hyperparameter
"""

from typing import Dict, Any, Optional, Tuple
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger
from smartcash.ui.info_boxes.hyperparameters_info import (
    get_hyperparameters_info,
    get_basic_hyperparameters_info,
    get_optimization_hyperparameters_info,
    get_advanced_hyperparameters_info
)

logger = get_logger(__name__)

def create_hyperparameters_info_panel() -> Tuple[widgets.Output, Any]:
    """
    Membuat panel informasi untuk hyperparameter.
    
    Returns:
        Tuple berisi output widget dan fungsi update
    """
    info_panel = widgets.Output(layout=widgets.Layout(
        width='100%',
        border='1px solid #ddd',
        padding='10px',
        margin='10px 0'
    ))
    
    def update_hyperparameters_info(ui_components: Optional[Dict[str, Any]] = None):
        """
        Update informasi hyperparameter di panel info.
        
        Args:
            ui_components: Komponen UI
        """
        with info_panel:
            clear_output(wait=True)
            
            if not ui_components:
                display(widgets.HTML(
                    f"<h3>{ICONS.get('info', 'ℹ️')} Informasi Hyperparameter</h3>"
                    f"<p>Tidak ada informasi yang tersedia.</p>"
                ))
                return
            
            # Dapatkan nilai dari komponen UI
            try:
                batch_size = ui_components.get('batch_size_slider', widgets.IntSlider()).value
                image_size = ui_components.get('image_size_slider', widgets.IntSlider()).value
                epochs = ui_components.get('epochs_slider', widgets.IntSlider()).value
                optimizer = ui_components.get('optimizer_dropdown', widgets.Dropdown()).value
                learning_rate = ui_components.get('learning_rate_slider', widgets.FloatLogSlider()).value
                scheduler = ui_components.get('scheduler_dropdown', widgets.Dropdown()).value
                
                # Tampilkan informasi
                html_content = f"""
                <h3>{ICONS.get('info', 'ℹ️')} Informasi Hyperparameter</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; grid-gap: 10px;">
                    <div>
                        <h4>Parameter Dasar</h4>
                        <ul>
                            <li><b>Batch Size:</b> {batch_size}</li>
                            <li><b>Image Size:</b> {image_size}</li>
                            <li><b>Epochs:</b> {epochs}</li>
                        </ul>
                    </div>
                    <div>
                        <h4>Optimasi</h4>
                        <ul>
                            <li><b>Optimizer:</b> {optimizer}</li>
                            <li><b>Learning Rate:</b> {learning_rate:.6f}</li>
                            <li><b>Scheduler:</b> {scheduler}</li>
                        </ul>
                    </div>
                </div>
                """
                
                # Tampilkan informasi tambahan jika ada
                if 'momentum_slider' in ui_components and not ui_components['momentum_slider'].disabled:
                    momentum = ui_components['momentum_slider'].value
                    html_content += f"""
                    <div>
                        <h4>Parameter Tambahan</h4>
                        <ul>
                            <li><b>Momentum:</b> {momentum:.4f}</li>
                    """
                    
                    if 'weight_decay_slider' in ui_components and not ui_components['weight_decay_slider'].disabled:
                        weight_decay = ui_components['weight_decay_slider'].value
                        html_content += f"<li><b>Weight Decay:</b> {weight_decay:.6f}</li>"
                    
                    html_content += "</ul></div>"
                
                # Tampilkan informasi early stopping jika diaktifkan
                if 'early_stopping_checkbox' in ui_components and ui_components['early_stopping_checkbox'].value:
                    patience = ui_components.get('patience_slider', widgets.IntSlider()).value
                    min_delta = ui_components.get('min_delta_slider', widgets.FloatSlider()).value
                    
                    html_content += f"""
                    <div>
                        <h4>Early Stopping</h4>
                        <ul>
                            <li><b>Patience:</b> {patience} epochs</li>
                            <li><b>Min Delta:</b> {min_delta:.4f}</li>
                        </ul>
                    </div>
                    """
                
                # Tampilkan informasi augmentasi jika diaktifkan
                if 'augment_checkbox' in ui_components and ui_components['augment_checkbox'].value:
                    html_content += f"""
                    <div>
                        <h4>Augmentasi</h4>
                        <p>Augmentasi data diaktifkan untuk meningkatkan generalisasi model.</p>
                    </div>
                    """
                
                # Tampilkan catatan
                html_content += f"""
                <div style="margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-left: 4px solid #17a2b8;">
                    <p><b>{ICONS.get('info', 'ℹ️')} Catatan:</b> Pastikan hyperparameter disesuaikan dengan dataset dan kebutuhan model Anda.</p>
                </div>
                """
                
                display(widgets.HTML(html_content))
            except Exception as e:
                logger.error(f"{ICONS.get('error', '❌')} Error update info panel: {str(e)}")
                display(widgets.HTML(
                    f"<h3>{ICONS.get('error', '❌')} Error</h3>"
                    f"<p>Terjadi error saat memperbarui informasi: {str(e)}</p>"
                ))
    
    return info_panel, update_hyperparameters_info
