"""
File: smartcash/ui/training_config/backbone_selection_component.py
Deskripsi: Komponen UI untuk pemilihan backbone dan layer model
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
    from smartcash.ui.components.headers import create_header
    from smartcash.ui.components.alerts import create_info_box
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Header
    header = create_header(
        f"{ICONS['model']} Backbone & Layer Selection",
        "Pemilihan arsitektur backbone dan konfigurasi layer model SmartCash"
    )
    
    # Backbone selection section
    backbone_section = widgets.HTML(
        f"<h3 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['folder']} Backbone Selection</h3>"
    )
    
    backbone_options = widgets.VBox([
        widgets.RadioButtons(
            options=['EfficientNet-B4 (Recommended)', 'CSPDarknet'],
            value='EfficientNet-B4 (Recommended)',
            description='Backbone:',
            style={'description_width': 'initial'},
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
    buttons_container = create_config_buttons("Konfigurasi Backbone")
    
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
        "Model Architecture Details",
        """
        <p><strong>EfficientNet-B4</strong> adalah model arsitektur CNN yang menggunakan:</p>
        <ul>
            <li>Compound Scaling untuk optimasi ukuran dan akurasi</li>
            <li>Mobile Inverted Bottleneck Convolution (MBConv) blocks</li>
            <li>Squeeze-and-Excitation blocks untuk attention</li>
        </ul>
        <p><strong>CSPDarknet</strong> adalah backbone original dari YOLOv5 dengan fitur:</p>
        <ul>
            <li>Cross Stage Partial Networks (CSP) untuk mengurangi bottlenecks</li>
            <li>Fast processing untuk real-time detection</li>
            <li>Memory efficient dengan parameter yang lebih sedikit</li>
        </ul>
        <p><strong>Layer Configuration</strong> memungkinkan model mendeteksi:</p>
        <ul>
            <li>Banknote: Deteksi uang kertas utuh</li>
            <li>Nominal: Deteksi area nominal pada uang</li>
            <li>Security: Deteksi fitur keamanan uang</li>
        </ul>
        """,
        'info',
        collapsed=True
    )
    
    # Rakit komponen UI
    ui = widgets.VBox([
        header,
        backbone_section,
        backbone_options,
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
        'backbone_options': backbone_options,
        'layer_config': layer_config,
        'layer_summary': layer_summary,
        'save_button': buttons_container.children[0],
        'reset_button': buttons_container.children[1],
        'status': status,
        'module_name': 'backbone_selection'
    }
    
    return ui_components