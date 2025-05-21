"""
File: smartcash/ui/training_config/backbone/handlers/info_panel.py
Deskripsi: Modul untuk memperbarui panel informasi backbone model
"""

from typing import Dict, Any, Optional
from IPython.display import display, HTML
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.training_config.backbone.handlers.default_config import get_default_backbone_config

logger = get_logger(__name__)

def update_backbone_info_panel(ui_components: Dict[str, Any], message: str = None) -> None:
    """
    Update info panel dengan informasi backbone yang dipilih.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan tambahan yang akan ditampilkan (opsional)
    """
    try:
        info_panel = ui_components.get('info_panel')
        if not info_panel:
            logger.warning(f"{ICONS.get('warning', '⚠️')} Info panel tidak ditemukan")
            return
            
        # Get current config from UI components
        backbone_config = _get_config_from_ui(ui_components)
        
        # Create HTML for info panel
        info_html = _generate_info_panel_html(backbone_config, message)
        
        # Update info panel
        with info_panel:
            info_panel.clear_output(wait=True)
            display(HTML(info_html))
        
        logger.info(f"{ICONS.get('success', '✅')} Info panel berhasil diupdate")
        
    except Exception as e:
        logger.error(f"{ICONS.get('error', '❌')} Error saat update info panel: {str(e)}")
        if 'info_panel' in ui_components:
            with ui_components['info_panel']:
                info_panel.clear_output(wait=True)
                display(HTML(f"<div style='color: red;'>{ICONS.get('error', '❌')} Error saat update info panel: {str(e)}</div>"))

def _get_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ekstrak informasi konfigurasi dari UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi backbone
    """
    # Default config sebagai fallback
    default_config = get_default_backbone_config()['model']
    
    # Ekstrak nilai dari UI components
    config = {
        'backbone': ui_components.get('backbone_dropdown', {}).value if hasattr(ui_components.get('backbone_dropdown', {}), 'value') else default_config['backbone'],
        'model_type': ui_components.get('model_type_dropdown', {}).value if hasattr(ui_components.get('model_type_dropdown', {}), 'value') else default_config['model_type'],
        'use_attention': ui_components.get('use_attention_checkbox', {}).value if hasattr(ui_components.get('use_attention_checkbox', {}), 'value') else default_config['use_attention'],
        'use_residual': ui_components.get('use_residual_checkbox', {}).value if hasattr(ui_components.get('use_residual_checkbox', {}), 'value') else default_config['use_residual'],
        'use_ciou': ui_components.get('use_ciou_checkbox', {}).value if hasattr(ui_components.get('use_ciou_checkbox', {}), 'value') else default_config['use_ciou'],
    }
    
    return config

def _generate_info_panel_html(config: Dict[str, Any], message: str = None) -> str:
    """
    Generate HTML untuk info panel.
    
    Args:
        config: Dictionary konfigurasi backbone
        message: Pesan tambahan yang akan ditampilkan (opsional)
        
    Returns:
        String HTML untuk info panel
    """
    # Map nilai konfigurasi ke label yang lebih deskriptif
    backbone_map = {
        'efficientnet_b4': 'EfficientNet-B4',
        'cspdarknet_s': 'CSPDarknet-S'
    }
    
    model_type_map = {
        'efficient_basic': 'EfficientNet Basic',
        'yolov5s': 'YOLOv5s'
    }
    
    backbone_label = backbone_map.get(config['backbone'], config['backbone'])
    model_type_label = model_type_map.get(config['model_type'], config['model_type'])
    
    # Buat HTML
    info_html = f"""
    <div style='font-family: monospace; padding: 10px; background-color: #f8f9fa; border-radius: 4px;'>
        <h4 style='color: #495057;'>{ICONS.get('info', 'ℹ️')} Konfigurasi Backbone:</h4>
        <ul style='list-style-type: none; padding-left: 20px;'>
            <li><b>Backbone:</b> {backbone_label} ({config['backbone']})</li>
            <li><b>Tipe Model:</b> {model_type_label} ({config['model_type']})</li>
            <li><b>Fitur Optimasi:</b>
                <ul style='list-style-type: none; padding-left: 20px;'>
                    <li>{ICONS.get('check' if config['use_attention'] else 'cross', '✓' if config['use_attention'] else '✗')} FeatureAdapter (Attention)</li>
                    <li>{ICONS.get('check' if config['use_residual'] else 'cross', '✓' if config['use_residual'] else '✗')} ResidualAdapter (Residual)</li>
                    <li>{ICONS.get('check' if config['use_ciou'] else 'cross', '✓' if config['use_ciou'] else '✗')} CIoU Loss</li>
                </ul>
            </li>
        </ul>
    """
    
    # Tambahkan deskripsi berdasarkan backbone
    if config['backbone'] == 'efficientnet_b4':
        info_html += f"""
        <div style='margin-top: 10px; padding: 10px; background-color: #e6f7ff; border-left: 4px solid #1890ff; border-radius: 4px;'>
            <p><b>{ICONS.get('bulb', '💡')} EfficientNet-B4:</b> Model backbone yang menggunakan compound scaling untuk menyeimbangkan kedalaman, lebar, dan resolusi. Memberikan performa yang baik dengan parameter yang lebih sedikit.</p>
        </div>
        """
    elif config['backbone'] == 'cspdarknet_s':
        info_html += f"""
        <div style='margin-top: 10px; padding: 10px; background-color: #e6f7ff; border-left: 4px solid #1890ff; border-radius: 4px;'>
            <p><b>{ICONS.get('bulb', '💡')} CSPDarknet-S:</b> Backbone yang digunakan pada YOLOv5s, menawarkan komputasi yang efisien dengan Cross Stage Partial (CSP) connections untuk menyeimbangkan kecepatan dan akurasi.</p>
        </div>
        """
    
    # Tambahkan pesan tambahan jika ada
    if message:
        info_html += f"""
        <div style='margin-top: 10px; padding: 10px; background-color: #fff7e6; border-left: 4px solid #fa8c16; border-radius: 4px;'>
            <p><b>{ICONS.get('warning', '⚠️')} Catatan:</b> {message}</p>
        </div>
        """
    
    info_html += "</div>"
    
    return info_html