"""
File: smartcash/ui_components/backbone_selection.py
Author: Refactor
Deskripsi: Komponen UI untuk pemilihan backbone dan layer model (optimized).
"""

import ipywidgets as widgets
from smartcash.utils.ui_utils import (
    create_component_header, 
    create_section_title,
    create_info_alert,
    create_info_box
)

def create_backbone_selection_ui():
    """Buat komponen UI untuk pemilihan backbone dan konfigurasi layer model."""
    # Container utama
    main_container = widgets.VBox(layout=widgets.Layout(width='100%', padding='10px'))
    
    # Header
    header = create_component_header(
        "Backbone & Layer Selection",
        "Pemilihan arsitektur backbone dan konfigurasi layer model SmartCash",
        "🦴"
    )
    
    # Backbone selection section
    backbone_section = create_section_title("Backbone Selection", "🧠")
    
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
    
    backbone_info = create_info_alert(
        "EfficientNet-B4 memberikan balance yang baik antara akurasi dan kecepatan untuk deteksi mata uang.",
        "info", "ℹ️"
    )
    
    # Layer configuration section
    layer_section = create_section_title("Layer Configuration", "🔍")
    
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
    
    layer_info = create_info_alert(
        "Model menggunakan pendekatan multi-layer untuk deteksi banknote, nominal, dan fitur keamanan.",
        "info", "ℹ️"
    )
    
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
    
    # Buttons container
    buttons_container = widgets.HBox([
        widgets.Button(
            description='Save Configuration',
            button_style='primary',
            icon='save',
            layout=widgets.Layout(margin='0 10px 0 0')
        ),
        widgets.Button(
            description='Reset to Default',
            button_style='warning',
            icon='refresh',
            layout=widgets.Layout(margin='0')
        )
    ])
    
    # Status output
    status_output = widgets.Output(
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
        <p style="color: #0c5460"><strong>EfficientNet-B4</strong> adalah model arsitektur CNN yang menggunakan:</p>
        <ul style="color: #0c5460">
            <li>Compound Scaling untuk optimasi ukuran dan akurasi</li>
            <li>Mobile Inverted Bottleneck Convolution (MBConv) blocks</li>
            <li>Squeeze-and-Excitation blocks untuk attention</li>
        </ul>
        <p style="color: #0c5460"><strong>CSPDarknet</strong> adalah backbone original dari YOLOv5 dengan fitur:</p>
        <ul style="color: #0c5460">
            <li>Cross Stage Partial Networks (CSP) untuk mengurangi bottlenecks</li>
            <li>Fast processing untuk real-time detection</li>
            <li>Memory efficient dengan parameter yang lebih sedikit</li>
        </ul>
        <p style="color: #0c5460"><strong>Layer Configuration</strong> memungkinkan model mendeteksi:</p>
        <ul style="color: #0c5460">
            <li>Banknote: Deteksi uang kertas utuh</li>
            <li>Nominal: Deteksi area nominal pada uang</li>
            <li>Security: Deteksi fitur keamanan uang</li>
        </ul>
        """,
        'info',
        collapsed=True
    )
    
    # Pasang semua komponen
    main_container.children = [
        header,
        backbone_section,
        backbone_options,
        backbone_info,
        widgets.HTML("<hr style='margin: 20px 0px;'>"),
        layer_section,
        layer_config,
        layer_info,
        layer_summary,
        widgets.HTML("<hr style='margin: 20px 0px;'>"),
        buttons_container,
        status_output,
        info_box
    ]
    
    # Dictionary untuk akses ke komponen dari luar
    ui_components = {
        'ui': main_container,
        'backbone_options': backbone_options,
        'layer_config': layer_config,
        'save_button': buttons_container.children[0],
        'reset_button': buttons_container.children[1],
        'status_output': status_output,
        'layer_summary': layer_summary
    }
    
    return ui_components