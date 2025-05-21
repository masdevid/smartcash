"""
File: smartcash/ui/training_config/hyperparameters/handlers/info_panel_updater.py
Deskripsi: Handler untuk memperbarui info panel di UI konfigurasi hyperparameters
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, clear_output

from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.training_config.hyperparameters.handlers.config_manager import get_hyperparameters_config
from smartcash.ui.training_config.hyperparameters.handlers.default_config import get_default_hyperparameters_config

logger = get_logger(__name__)

def update_hyperparameters_info(ui_components: Dict[str, Any]) -> None:
    """
    Update info panel dengan informasi hyperparameters yang dipilih.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        info_panel = ui_components.get('info_panel')
        if not info_panel:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Info panel tidak ditemukan")
            return
            
        # Get training parameters from UI components
        batch_size = ui_components.get('batch_size_slider', widgets.IntSlider()).value
        image_size = ui_components.get('image_size_slider', widgets.IntSlider()).value
        epochs = ui_components.get('epochs_slider', widgets.IntSlider()).value
        
        # Get optimizer parameters
        optimizer = ui_components.get('optimizer_dropdown', widgets.Dropdown()).value
        learning_rate = ui_components.get('learning_rate_slider', widgets.FloatLogSlider()).value
        
        # Get scheduler parameters
        scheduler_enabled = ui_components.get('scheduler_checkbox', widgets.Checkbox()).value
        scheduler_type = ui_components.get('scheduler_dropdown', widgets.Dropdown()).value if scheduler_enabled else "Tidak Aktif"
        
        # Get early stopping parameters
        early_stopping_enabled = ui_components.get('early_stopping_checkbox', widgets.Checkbox()).value
        patience = ui_components.get('patience_slider', widgets.IntSlider()).value if early_stopping_enabled else 0
        
        # Get augmentation parameters
        augmentation_enabled = ui_components.get('augment_checkbox', widgets.Checkbox()).value
        
        # Update info panel dengan informasi hyperparameters dalam format yang lebih terstruktur
        # Gunakan CSS grid untuk layout yang lebih baik
        html_content = f"""
        <div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            <h3 style="color: {COLORS.get('primary', '#007bff')}; margin-bottom: 15px;">{ICONS.get('info', 'ℹ️')} Ringkasan Konfigurasi Hyperparameter</h3>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; grid-gap: 20px;">
                <div style="border: 1px solid #eee; border-radius: 8px; padding: 15px; background-color: #f8f9fa;">
                    <h4 style="color: {COLORS.get('dark', '#343a40')}; margin-top: 0;">{ICONS.get('settings', '⚙️')} Parameter Training</h4>
                    <ul style="padding-left: 20px; margin-bottom: 0;">
                        <li><b>Batch Size:</b> {batch_size}</li>
                        <li><b>Image Size:</b> {image_size}px</li>
                        <li><b>Epochs:</b> {epochs}</li>
                    </ul>
                </div>
                
                <div style="border: 1px solid #eee; border-radius: 8px; padding: 15px; background-color: #f8f9fa;">
                    <h4 style="color: {COLORS.get('dark', '#343a40')}; margin-top: 0;">{ICONS.get('optimization', '🔄')} Optimasi</h4>
                    <ul style="padding-left: 20px; margin-bottom: 0;">
                        <li><b>Optimizer:</b> {optimizer}</li>
                        <li><b>Learning Rate:</b> {learning_rate:.6f}</li>
                        <li><b>Scheduler:</b> {scheduler_type}</li>
                    </ul>
                </div>
            </div>
            
            <div style="margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr; grid-gap: 20px;">
        """
        
        # Tambahkan informasi early stopping jika diaktifkan
        if early_stopping_enabled:
            html_content += f"""
                <div style="border: 1px solid #eee; border-radius: 8px; padding: 15px; background-color: #f8f9fa;">
                    <h4 style="color: {COLORS.get('dark', '#343a40')}; margin-top: 0;">{ICONS.get('time', '⏱️')} Early Stopping</h4>
                    <ul style="padding-left: 20px; margin-bottom: 0;">
                        <li><b>Patience:</b> {patience} epochs</li>
                        <li><b>Min Delta:</b> {ui_components.get('min_delta_slider', widgets.FloatSlider()).value:.4f}</li>
                    </ul>
                </div>
            """
        
        # Tambahkan informasi augmentasi jika diaktifkan
        if augmentation_enabled:
            html_content += f"""
                <div style="border: 1px solid #eee; border-radius: 8px; padding: 15px; background-color: #f8f9fa;">
                    <h4 style="color: {COLORS.get('dark', '#343a40')}; margin-top: 0;">{ICONS.get('chart', '📊')} Augmentasi Data</h4>
                    <p style="margin: 0;">Augmentasi data diaktifkan untuk meningkatkan generalisasi model.</p>
                </div>
            """
        
        # Tutup div grid
        html_content += """
            </div>
            
            <div style="margin-top: 20px; padding: 12px 15px; background-color: #e9f7fe; border-left: 4px solid #17a2b8; border-radius: 4px;">
                <p style="margin: 0;"><b>{} Catatan:</b> Parameter yang optimal dapat bervariasi tergantung pada dataset dan model yang digunakan.</p>
            </div>
        </div>
        """.format(ICONS.get('info', 'ℹ️'))
        
        # Update informasi ke info panel
        with info_panel:
            clear_output(wait=True)
            display(widgets.HTML(html_content))
            
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat update info panel: {str(e)}")
        if 'info_panel' in ui_components:
            with ui_components['info_panel']:
                clear_output(wait=True)
                display(widgets.HTML(
                    f"""<div style="color: {COLORS.get('error', '#dc3545')}; padding: 10px; border: 1px solid {COLORS.get('error', '#dc3545')}; border-radius: 5px;">
                    {ICONS.get('error', '❌')} <b>Error saat memperbarui informasi:</b> {str(e)}
                    </div>"""
                ))

def create_hyperparameters_info_panel() -> tuple:
    """
    Buat panel informasi untuk hyperparameter.
    
    Returns:
        Tuple berisi panel dan fungsi untuk memperbarui panel
    """
    # Buat panel output
    info_panel = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            border='1px solid #ddd',
            padding='10px',
            margin='10px 0',
            min_height='150px'
        )
    )
    
    # Tampilkan pesan awal
    with info_panel:
        display(widgets.HTML(
            f"""<div style="text-align: center; padding: 20px;">
            {ICONS.get('info', 'ℹ️')} <b>Informasi Hyperparameter</b>
            <p>Informasi akan ditampilkan disini setelah mengubah parameter.</p>
            </div>"""
        ))
    
    return info_panel, update_hyperparameters_info