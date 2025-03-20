"""
File: smartcash/ui/training_config/backbone_selection_component.py
Deskripsi: Komponen UI untuk pemilihan backbone dan layer model yang kompatibel dengan ModelManager
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_backbone_selection_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Buat komponen UI untuk pemilihan backbone dan konfigurasi layer model.
    
    Args:
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi widget UI
    """
    # Import komponen UI
    from smartcash.ui.utils.headers import create_header
    from smartcash.ui.utils.alerts import create_info_box
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Header
    header = create_header(
        f"{ICONS['model']} Backbone & Model Configuration",
        "Pemilihan model, arsitektur backbone dan konfigurasi layer untuk SmartCash"
    )
    
    # Model type selection section 
    model_section = widgets.HTML(
        f"<h3 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['folder']} Model Selection</h3>"
    )
    
    # Mendapatkan opsi model dari ModelManager
    model_options_list = [
        'efficient_basic - Model dasar tanpa optimasi khusus',
        'efficient_optimized - Model dengan EfficientNet-B4 dan FeatureAdapter',
        'efficient_advanced - Model dengan semua optimasi: FeatureAdapter, ResidualAdapter, dan CIoU',
        'yolov5s - YOLOv5s dengan CSPDarknet sebagai backbone (model pembanding)',
        'efficient_experiment - Model penelitian dengan konfigurasi khusus'
    ]
    
    model_options = widgets.VBox([
        widgets.Dropdown(
            options=model_options_list,
            value=model_options_list[1],  # Default: efficient_optimized
            description='Model Type:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='80%')
        ),
        widgets.HTML(
            value="""<div style="padding: 10px; color: black; background-color: #e3f2fd; border-radius: 5px; margin: 10px 0;">
                    <p><b>💡 Catatan:</b> Pemilihan model akan menentukan backbone dan fitur optimasi secara otomatis.</p>
                    </div>"""
        )
    ])
    
    # Backbone options section
    backbone_section = widgets.HTML(
        f"<h3 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['folder']} Backbone Settings</h3>"
    )
    
    backbone_options = widgets.VBox([
        widgets.Dropdown(
            options=[
                'efficientnet_b0 - Versi ringan',
                'efficientnet_b1 - Ukuran kecil',
                'efficientnet_b2 - Ukuran sedang',
                'efficientnet_b3 - Keseimbangan performa',
                'efficientnet_b4 - Rekomendasi SmartCash',
                'efficientnet_b5 - Performa tinggi',
                'cspdarknet_s - YOLOv5s',
                'cspdarknet_m - YOLOv5m',
                'cspdarknet_l - YOLOv5l'
            ],
            value='efficientnet_b4 - Rekomendasi SmartCash',
            description='Backbone:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='80%'),
            disabled=True  # Disabled karena dipilih otomatis oleh model type
        ),
        widgets.Checkbox(
            value=True,
            description='Use pretrained weights',
            style={'description_width': 'initial'}
        ),
        widgets.Checkbox(
            value=True,
            description='Freeze backbone initially',
            style={'description_width': 'initial'}
        )
    ])
    
    # Advanced features section
    features_section = widgets.HTML(
        f"<h3 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['settings']} Advanced Features</h3>"
    )
    
    features_options = widgets.VBox([
        widgets.Checkbox(
            value=True,
            description='Gunakan FeatureAdapter/Attention Mechanism',
            style={'description_width': 'initial'},
            disabled=True  # Disabled karena ditentukan oleh model type
        ),
        widgets.Checkbox(
            value=False,
            description='Gunakan ResidualAdapter untuk fitur yang lebih baik',
            style={'description_width': 'initial'},
            disabled=True  # Disabled karena ditentukan oleh model type
        ),
        widgets.Checkbox(
            value=False,
            description='Gunakan CIoU Loss untuk deteksi yang lebih akurat',
            style={'description_width': 'initial'},
            disabled=True  # Disabled karena ditentukan oleh model type
        ),
        widgets.IntSlider(
            value=3,
            min=1,
            max=5,
            description='Jumlah Residual Blocks:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='70%'),
            disabled=True  # Disabled karena ditentukan oleh model type
        )
    ])
    
    # Layer configuration section
    layer_section = widgets.HTML(
        f"<h3 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['folder']} Layer Configuration</h3>"
    )
    
    layer_config = widgets.VBox([
        # Banknote layer
        widgets.HBox([
            widgets.Checkbox(
                value=True,
                description='Banknote Layer',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='150px')
            ),
            widgets.FloatText(
                value=0.25,
                description='Threshold:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='200px')
            ),
            widgets.Label('Classes: 001, 002, 005, 010, 020, 050, 100', 
                         style={'description_width': 'initial'})
        ]),
        
        # Nominal layer
        widgets.HBox([
            widgets.Checkbox(
                value=True,
                description='Nominal Layer',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='150px')
            ),
            widgets.FloatText(
                value=0.30,
                description='Threshold:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='200px')
            ),
            widgets.Label('Classes: l2_001, l2_002, l2_005, l2_010, l2_020, l2_050, l2_100', 
                         style={'description_width': 'initial'})
        ]),
        
        # Security layer
        widgets.HBox([
            widgets.Checkbox(
                value=True,
                description='Security Layer',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='150px')
            ),
            widgets.FloatText(
                value=0.35,
                description='Threshold:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='200px')
            ),
            widgets.Label('Classes: l3_sign, l3_text, l3_thread', 
                         style={'description_width': 'initial'})
        ])
    ])
    
    # Layer configuration summary
    layer_summary = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            min_height='100px',
            margin='10px 0',
            padding='10px',    
            overflow='auto'
        )
    )
    
    # Tombol aksi
    from smartcash.ui.training_config.config_buttons import create_config_buttons
    buttons_container = create_config_buttons("Konfigurasi Model")
    
    # Status output
    status = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            overflow='auto'
        )
    )
    
    # Info box with additional details
    info_box = create_info_box(
        "Model & Backbone Details",
        """
        <p><strong>Tipe Model SmartCash:</strong></p>
        <ul>
            <li><strong>efficient_basic</strong>: Model dasar dengan backbone EfficientNet</li>
            <li><strong>efficient_optimized</strong>: Model dengan EfficientNet-B4 dan FeatureAdapter untuk performa lebih baik</li>
            <li><strong>efficient_advanced</strong>: Model dengan semua optimasi termasuk ResidualAdapter dan CIoU Loss</li>
            <li><strong>yolov5s</strong>: Model bawaan YOLOv5s dengan CSPDarknet sebagai baseline perbandingan</li>
            <li><strong>efficient_experiment</strong>: Model penelitian dengan konfigurasi khusus</li>
        </ul>
        
        <p><strong>Fitur EfficientNet:</strong></p>
        <ul>
            <li><strong>FeatureAdapter</strong>: Meningkatkan kualitas fitur dengan mekanisme attention</li>
            <li><strong>ResidualAdapter</strong>: Menambahkan koneksi residual untuk training yang lebih stabil</li>
            <li><strong>CIoU Loss</strong>: Complete-IoU loss yang memperhitungkan aspect ratio dan alignment</li>
        </ul>
        
        <p><strong>Layer Deteksi:</strong></p>
        <ul>
            <li><strong>Banknote</strong>: Deteksi uang kertas utuh</li>
            <li><strong>Nominal</strong>: Deteksi area nominal pada uang</li>
            <li><strong>Security</strong>: Deteksi fitur keamanan uang</li>
        </ul>
        """,
        'info',
        collapsed=True
    )
    
    # Rakit komponen UI
    ui = widgets.VBox([
        header,
        model_section,
        model_options,
        backbone_section,
        backbone_options,
        features_section,
        features_options,
        widgets.HTML("<hr style='margin: 15px 0; border: 0; border-top: 1px solid #eee;'>"),
        layer_section,
        layer_config,
        layer_summary,
        widgets.HTML("<hr style='margin: 15px 0; border: 0; border-top: 1px solid #eee;'>"),
        buttons_container,
        status,
        info_box
    ])
    
    # Dictionary untuk akses ke komponen dari luar
    ui_components = {
        'ui': ui,
        'header': header,
        'model_options': model_options,
        'backbone_options': backbone_options,
        'features_options': features_options,
        'layer_config': layer_config,
        'layer_summary': layer_summary,
        'save_button': buttons_container.children[0],
        'reset_button': buttons_container.children[1],
        'status': status,
        'module_name': 'backbone_selection'
    }
    
    return ui_components