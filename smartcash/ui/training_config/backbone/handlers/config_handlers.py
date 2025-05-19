"""
File: smartcash/ui/training_config/backbone/handlers/config_handlers.py
Deskripsi: Handler untuk konfigurasi pada UI pemilihan backbone model SmartCash
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import clear_output, display

from smartcash.ui.utils.constants import ICONS
from smartcash.common.config import get_config_manager
from smartcash.common.logger import get_logger

# Setup logger dengan level CRITICAL untuk mengurangi log
logger = get_logger(__name__)
logger.setLevel("CRITICAL")

def update_config_from_ui(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi dari komponen UI.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    
    Returns:
        Dictionary berisi konfigurasi yang diperbarui
    """
    # Dapatkan ConfigManager singleton
    config_manager = get_config_manager()
    
    # Dapatkan konfigurasi saat ini atau buat baru jika belum ada
    current_config = config_manager.get_module_config('model', {})
    
    # Register UI components untuk persistensi
    config_manager.register_ui_components('backbone', ui_components)
    
    # Dapatkan nilai dari komponen UI
    model_type = ui_components.get('model_type_dropdown').value
    backbone = ui_components.get('backbone_dropdown').value
    use_attention = ui_components.get('use_attention_checkbox').value
    use_residual = ui_components.get('use_residual_checkbox').value
    use_ciou = ui_components.get('use_ciou_checkbox').value
    
    # Update konfigurasi model
    if 'model' not in current_config:
        current_config['model'] = {}
    
    current_config['model']['type'] = model_type
    current_config['model']['backbone'] = backbone
    current_config['model']['use_attention'] = use_attention
    current_config['model']['use_residual'] = use_residual
    current_config['model']['use_ciou'] = use_ciou
    
    # Pastikan konfigurasi eksperimen juga diperbarui
    if 'experiments' not in current_config:
        current_config['experiments'] = {
            'backbones': [],
            'scenarios': []
        }
    
    # Pastikan backbone yang dipilih ada dalam daftar backbones
    backbone_exists = False
    for backbone_config in current_config['experiments']['backbones']:
        if backbone_config.get('config', {}).get('backbone') == backbone:
            backbone_exists = True
            break
    
    if not backbone_exists:
        backbone_name = 'EfficientNet-B4' if backbone == 'efficientnet_b4' else 'CSPDarknet-S'
        backbone_desc = 'EfficientNet-B4 backbone' if backbone == 'efficientnet_b4' else 'YOLOv5s default backbone'
        
        current_config['experiments']['backbones'].append({
            'name': backbone,
            'description': backbone_desc,
            'config': {
                'backbone': backbone,
                'pretrained': True
            }
        })
    
    # Pastikan skenario yang sesuai dengan konfigurasi saat ini ada dalam daftar scenarios
    scenario_exists = False
    for scenario_config in current_config['experiments']['scenarios']:
        if (scenario_config.get('config', {}).get('type') == model_type and
            scenario_config.get('config', {}).get('backbone') == backbone and
            scenario_config.get('config', {}).get('use_attention') == use_attention and
            scenario_config.get('config', {}).get('use_residual') == use_residual and
            scenario_config.get('config', {}).get('use_ciou') == use_ciou):
            scenario_exists = True
            break
    
    if not scenario_exists:
        scenario_name = f"{model_type}_{backbone}"
        if use_attention or use_residual or use_ciou:
            scenario_name += "_optimized"
        
        scenario_desc = f"Model dengan backbone {backbone}"
        if use_attention or use_residual or use_ciou:
            optimizations = []
            if use_attention:
                optimizations.append("FeatureAdapter")
            if use_residual:
                optimizations.append("ResidualAdapter")
            if use_ciou:
                optimizations.append("CIoU")
            
            scenario_desc += f" dengan optimasi: {', '.join(optimizations)}"
        
        current_config['experiments']['scenarios'].append({
            'name': scenario_name,
            'description': scenario_desc,
            'config': {
                'type': model_type,
                'backbone': backbone,
                'use_attention': use_attention,
                'use_residual': use_residual,
                'use_ciou': use_ciou
            }
        })
    
    return current_config

def update_ui_from_config(ui_components: Dict[str, Any], config_to_use: Optional[Dict[str, Any]] = None) -> None:
    """
    Update komponen UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config_to_use: Dictionary berisi konfigurasi yang akan digunakan (opsional)
    """
    # Dapatkan ConfigManager singleton
    config_manager = get_config_manager()
    
    # Dapatkan konfigurasi saat ini atau gunakan yang diberikan
    current_config = config_to_use if config_to_use else config_manager.get_module_config('model', {})
    
    # Register UI components untuk persistensi
    config_manager.register_ui_components('backbone', ui_components)
    
    # Dapatkan nilai dari konfigurasi
    model_config = current_config.get('model', {})
    model_type = model_config.get('type', 'efficient_basic')
    backbone = model_config.get('backbone', 'efficientnet_b4')
    use_attention = model_config.get('use_attention', False)
    use_residual = model_config.get('use_residual', False)
    use_ciou = model_config.get('use_ciou', False)
    
    # Validasi nilai
    valid_model_types = ['efficient_basic', 'yolov5s']
    valid_backbones = ['efficientnet_b4', 'cspdarknet_s']
    
    if model_type not in valid_model_types:
        logger.warning(f"{ICONS.get('warning', '⚠️')} Model type tidak valid: {model_type}, menggunakan default: efficient_basic")
        model_type = 'efficient_basic'
    
    if backbone not in valid_backbones:
        logger.warning(f"{ICONS.get('warning', '⚠️')} Backbone tidak valid: {backbone}, menggunakan default: efficientnet_b4")
        backbone = 'efficientnet_b4'
    
    # Update komponen UI
    model_type_dropdown = ui_components.get('model_type_dropdown')
    backbone_dropdown = ui_components.get('backbone_dropdown')
    use_attention_checkbox = ui_components.get('use_attention_checkbox')
    use_residual_checkbox = ui_components.get('use_residual_checkbox')
    use_ciou_checkbox = ui_components.get('use_ciou_checkbox')
    
    if model_type_dropdown:
        model_type_dropdown.value = model_type
    
    if backbone_dropdown:
        backbone_dropdown.value = backbone
    
    if use_attention_checkbox:
        use_attention_checkbox.value = use_attention
        use_attention_checkbox.disabled = (backbone == 'cspdarknet_s')
    
    if use_residual_checkbox:
        use_residual_checkbox.value = use_residual
        use_residual_checkbox.disabled = (backbone == 'cspdarknet_s')
    
    if use_ciou_checkbox:
        use_ciou_checkbox.value = use_ciou
        use_ciou_checkbox.disabled = (backbone == 'cspdarknet_s')
    
    # Update info panel
    update_backbone_info(ui_components)

def update_backbone_info(ui_components: Dict[str, Any]):
    """
    Update panel informasi backbone.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    # Dapatkan panel informasi
    info_panel = ui_components.get('info_panel')
    if not info_panel:
        logger.critical(f"{ICONS.get('error', '❌')} Info panel tidak ditemukan")
        return
        
    # Dapatkan ConfigManager singleton
    config_manager = get_config_manager()
    
    # Pastikan UI components teregistrasi untuk persistensi
    config_manager.register_ui_components('backbone', ui_components)
    
    # Dapatkan nilai dari komponen UI
    model_type = ui_components.get('model_type_dropdown').value
    backbone = ui_components.get('backbone_dropdown').value
    use_attention = ui_components.get('use_attention_checkbox').value
    use_residual = ui_components.get('use_residual_checkbox').value
    use_ciou = ui_components.get('use_ciou_checkbox').value
    
    with info_panel:
        clear_output(wait=True)
        
        # Tampilkan informasi backbone
        if backbone == 'efficientnet_b4':
            display(widgets.HTML("""
            <div style="padding: 10px; border-left: 3px solid #4CAF50; background-color: #f8f9fa;">
                <h4 style="margin-top: 0;">EfficientNet-B4</h4>
                <p><strong>Deskripsi:</strong> Backbone yang dioptimalkan untuk deteksi mata uang dengan performa tinggi.</p>
                <p><strong>Karakteristik:</strong></p>
                <ul>
                    <li>Width Coefficient: 1.4</li>
                    <li>Depth Coefficient: 1.8</li>
                    <li>Resolution: 380px</li>
                    <li>Parameter: ~19M</li>
                </ul>
                <p><strong>Kelebihan:</strong> Akurasi tinggi, efisien untuk deteksi objek kecil seperti mata uang.</p>
            </div>
            """))
        elif backbone == 'cspdarknet_s':
            display(widgets.HTML("""
            <div style="padding: 10px; border-left: 3px solid #2196F3; background-color: #f8f9fa;">
                <h4 style="margin-top: 0;">CSPDarknet-S</h4>
                <p><strong>Deskripsi:</strong> Backbone default YOLOv5s yang ringan dan cepat.</p>
                <p><strong>Karakteristik:</strong></p>
                <ul>
                    <li>Depth Multiple: 0.33</li>
                    <li>Width Multiple: 0.50</li>
                    <li>Parameter: ~7M</li>
                </ul>
                <p><strong>Kelebihan:</strong> Kecepatan inferensi tinggi, cocok untuk perangkat dengan komputasi terbatas.</p>
            </div>
            """))
        
        # Tampilkan informasi optimasi
        if backbone == 'efficientnet_b4':
            optimizations = []
            if use_attention:
                optimizations.append("""
                <div style="margin-top: 10px; padding: 10px; border-left: 3px solid #FF9800; background-color: #f8f9fa;">
                    <h5 style="margin-top: 0;">FeatureAdapter (Attention)</h5>
                    <p>Mekanisme attention untuk meningkatkan fokus pada fitur penting mata uang.</p>
                    <p><strong>Efek:</strong> Meningkatkan akurasi deteksi +3-5%, terutama untuk mata uang yang rusak atau terlipat.</p>
                </div>
                """)
            
            if use_residual:
                optimizations.append("""
                <div style="margin-top: 10px; padding: 10px; border-left: 3px solid #9C27B0; background-color: #f8f9fa;">
                    <h5 style="margin-top: 0;">ResidualAdapter</h5>
                    <p>Koneksi residual tambahan untuk mempertahankan informasi resolusi tinggi.</p>
                    <p><strong>Efek:</strong> Meningkatkan deteksi detail halus pada mata uang seperti hologram dan watermark.</p>
                </div>
                """)
            
            if use_ciou:
                optimizations.append("""
                <div style="margin-top: 10px; padding: 10px; border-left: 3px solid #E91E63; background-color: #f8f9fa;">
                    <h5 style="margin-top: 0;">CIoU Loss</h5>
                    <p>Complete-IoU Loss untuk regresi bounding box yang lebih akurat.</p>
                    <p><strong>Efek:</strong> Meningkatkan presisi lokalisasi mata uang, terutama saat tumpang tindih.</p>
                </div>
                """)
            
            if optimizations:
                display(widgets.HTML("<h4>Optimasi yang Diaktifkan</h4>"))
                for opt in optimizations:
                    display(widgets.HTML(opt))
            else:
                display(widgets.HTML("""
                <div style="margin-top: 10px; padding: 10px; border-left: 3px solid #607D8B; background-color: #f8f9fa;">
                    <p>Tidak ada optimasi khusus yang diaktifkan. Aktifkan optimasi untuk meningkatkan performa model.</p>
                </div>
                """))
        
        # Tampilkan informasi model type
        display(widgets.HTML("<h4>Tipe Model</h4>"))
        if model_type == 'efficient_basic':
            display(widgets.HTML("""
            <div style="padding: 10px; border-left: 3px solid #3F51B5; background-color: #f8f9fa;">
                <h5 style="margin-top: 0;">EfficientNet Basic</h5>
                <p>Model dasar dengan EfficientNet sebagai backbone dan YOLOv5 sebagai detektor.</p>
                <p><strong>Karakteristik:</strong> Seimbang antara akurasi dan kecepatan, optimal untuk deteksi mata uang.</p>
            </div>
            """))
        elif model_type == 'yolov5s':
            display(widgets.HTML("""
            <div style="padding: 10px; border-left: 3px solid #00BCD4; background-color: #f8f9fa;">
                <h5 style="margin-top: 0;">YOLOv5s</h5>
                <p>Model YOLOv5s standar dengan CSPDarknet sebagai backbone.</p>
                <p><strong>Karakteristik:</strong> Kecepatan tinggi, ukuran model kecil, cocok untuk deployment di perangkat terbatas.</p>
            </div>
            """))
